from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


PLUGIN_ROOT = Path(os.environ.get("KIMODO_UE_PLUGIN_ROOT", Path(__file__).resolve().parents[3]))
PROJECT_ROOT = Path(os.environ.get("KIMODO_UE_PROJECT_ROOT", PLUGIN_ROOT.parents[1] if len(PLUGIN_ROOT.parents) > 1 else PLUGIN_ROOT))
PYTHON_ROOT = Path(os.environ.get("KIMODO_UE_PYTHON_ROOT", PLUGIN_ROOT / "Resources" / "Python"))
VENDOR_ROOT = Path(os.environ.get("KIMODO_VENDOR_ROOT", PYTHON_ROOT / "Vendor" / "kimodo-main"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("KIMODO_UE_OUTPUT_ROOT", PROJECT_ROOT / "Saved" / "EmbodiedSpaceStudio" / "Jobs"))
DEFAULT_LOG_ROOT = Path(os.environ.get("KIMODO_UE_LOG_ROOT", PLUGIN_ROOT / "Saved" / "EmbodiedSpaceStudio" / "Logs"))
BRIDGE_OUTPUT_ROOT_TEXT = os.environ.get("KIMODO_BRIDGE_OUTPUT_ROOT", "").strip()
BRIDGE_OUTPUT_ROOT = Path(BRIDGE_OUTPUT_ROOT_TEXT) if BRIDGE_OUTPUT_ROOT_TEXT else None
BRIDGE_OUTPUT_HOST_ROOT = os.environ.get("KIMODO_BRIDGE_OUTPUT_HOST_ROOT", "").strip()
SERVICE_VERSION = "0.1.0"
UE_PREVIEW_MANIFEST_NAME = "ue_preview_manifest.json"
UE_MOTION_DIAGNOSTICS_NAME = "ue_motion_diagnostics.json"
CHECKPOINT_ROOT = ""
TEXT_ENCODERS_ROOT = ""
TEXT_ENCODER_CONNECT_TIMEOUT_SECONDS = 5.0
TEXT_ENCODER_READY_TIMEOUT_SECONDS = 180.0
TEXT_ENCODER_READY_RETRY_DELAY_SECONDS = 1.5
TEXT_ENCODER_SEMANTIC_MIN_MEAN_ABS_DIFF = 1.0e-4
TEXT_ENCODER_SEMANTIC_MAX_COSINE = 0.9999
TEXT_ENCODER_SEMANTIC_CACHE_TTL_SECONDS = 600.0
TEXT_ENCODER_SEMANTIC_CACHE_LOCK = threading.Lock()
TEXT_ENCODER_SEMANTIC_CACHE: dict[str, tuple[float, str]] = {}

if str(VENDOR_ROOT) not in sys.path:
	sys.path.insert(0, str(VENDOR_ROOT))

from kimodo.geometry import matrix_to_quaternion  # noqa: E402
from kimodo.constraints import (  # noqa: E402
	FullBodyConstraintSet,
	LeftFootConstraintSet,
	LeftHandConstraintSet,
	RightFootConstraintSet,
	RightHandConstraintSet,
	Root2DConstraintSet,
	save_constraints_lst,
)
from kimodo.exports.motion_io import load_kimodo_npz_as_torch  # noqa: E402
from kimodo.model.registry import get_model_info, get_models_for_demo  # noqa: E402
from kimodo.skeleton.registry import build_skeleton  # noqa: E402


class GenerateJobRequest(BaseModel):
	sessionId: str = ""
	prompt: str = ""
	prompts: list[str] = Field(default_factory=list)
	duration: float = 5.0
	durations: list[float] = Field(default_factory=list)
	promptCategory: str = ""
	promptSequenceId: str = ""
	promptSequence: str = ""
	promptScene: str = ""
	model: str = "kimodo-soma-rp"
	bUseKimodoDefaultSourceCharacter: bool = True
	sourceSkeletalMeshPath: str = ""
	sourceSkeletonPath: str = ""
	sourceCharacterLabel: str = "SOMA Human Body"
	targetBoneNames: list[str] = Field(default_factory=list)
	targetRootBoneIndex: int = 0
	numSamples: int = 1
	diffusionSteps: int = 100
	bStabilizeFootContacts: bool = False
	footContactStabilizationLevel: int = 2
	numTransitionFrames: int = 5
	postprocessRootMargin: float = 0.04
	postprocessAboveGroundOffset: float = 0.0
	bDisablePostprocess: bool = True
	bSaveExampleDir: bool = True
	bUseMultiPromptInput: bool = False
	constraintsPath: str = ""
	nativeConstraints: list["ConstraintItem"] = Field(default_factory=list)
	constraintSourceMotionPath: str = ""
	outputDir: str = ""
	outputName: str = "motion"
	textEncoderMode: str = "local"
	textEncoderDevice: str = "auto"
	bUseSeed: bool = False
	seed: int = 0
	cfgType: str = ""
	cfgWeight: list[float] = Field(default_factory=list)

	@model_validator(mode="after")
	def validate_prompt_payload(self) -> "GenerateJobRequest":
		if not self.prompt.strip() and not self.prompts:
			raise ValueError("Either prompt or prompts must be provided.")
		if self.durations and self.prompts and len(self.durations) != len(self.prompts):
			raise ValueError("Durations length must match prompts length when using multi-prompt mode.")
		return self


class GenerateJobAccepted(BaseModel):
	jobId: str
	status: str


class JobStatusResponse(BaseModel):
	jobId: str
	status: str
	command: str = ""
	error: str = ""
	outputDir: str = ""
	stdOutLog: str = ""
	stdErrLog: str = ""
	createdAt: str = ""
	startedAt: str = ""
	finishedAt: str = ""
	generatedFiles: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
	status: str
	serviceVersion: str
	outputRoot: str
	vendorRoot: str
	bMotionCorrectionInstalled: bool
	activeJobs: int
	activeSessions: int


class SessionCreateRequest(BaseModel):
	model: str = ""


class SessionModelUpdateRequest(BaseModel):
	model: str = ""


class PromptSegment(BaseModel):
	prompt: str = ""
	startFrame: int = 0
	endFrame: int = 149


class ConstraintItem(BaseModel):
	track: str = ""
	startFrame: int = 0
	endFrame: int = 0
	bHasEditedTransform: bool = False
	editedLocationCm: Any = Field(default_factory=dict)
	editedRotation: Any = Field(default_factory=dict)
	editedJointTransforms: list[dict[str, Any]] = Field(default_factory=list)


class SessionAuthoringStateRequest(BaseModel):
	prompts: list[PromptSegment] = Field(default_factory=list)
	constraints: list[ConstraintItem] = Field(default_factory=list)
	constraintsPath: str = ""
	framesPerSecond: float = 30.0


GenerateJobRequest.model_rebuild()


class ExampleDescriptor(BaseModel):
	exampleName: str
	model: str


class ExampleLoadRequest(BaseModel):
	model: str = ""
	exampleName: str = ""
	framesPerSecond: float = 30.0


class ExampleLoadResponse(BaseModel):
	exampleName: str
	model: str
	prompts: list[PromptSegment] = Field(default_factory=list)
	constraintsPath: str = ""
	previewManifestPath: str = ""


class SessionResponse(BaseModel):
	sessionId: str
	status: str
	createdAt: str = ""
	updatedAt: str = ""
	model: str = ""
	activeJobId: str = ""


class ModelDescriptor(BaseModel):
	shortKey: str
	displayName: str
	family: str
	skeleton: str
	dataset: str
	version: str
	repoId: str


@dataclass
class JobRecord:
	job_id: str
	request: GenerateJobRequest
	status: str = "queued"
	command: str = ""
	error: str = ""
	output_dir: str = ""
	stdout_log: str = ""
	stderr_log: str = ""
	created_at: str = field(default_factory=lambda: now_utc())
	started_at: str = ""
	finished_at: str = ""
	generated_files: list[str] = field(default_factory=list)
	cancel_requested: bool = False
	process: subprocess.Popen[str] | None = field(default=None, repr=False)
	lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class SessionRecord:
	session_id: str
	status: str = "active"
	created_at: str = field(default_factory=lambda: now_utc())
	updated_at: str = field(default_factory=lambda: now_utc())
	model: str = ""
	active_job_id: str = ""
	prompts: list[PromptSegment] = field(default_factory=list)
	constraints: list[ConstraintItem] = field(default_factory=list)
	constraints_path: str = ""
	frames_per_second: float = 30.0


def now_utc() -> str:
	return datetime.now(timezone.utc).isoformat()


def detect_motion_correction() -> bool:
	try:
		import motion_correction  # noqa: F401
	except Exception:
		return False
	return True


def build_session_response(record: SessionRecord) -> SessionResponse:
	return SessionResponse(
		sessionId=record.session_id,
		status=record.status,
		createdAt=record.created_at,
		updatedAt=record.updated_at,
		model=record.model,
		activeJobId=record.active_job_id,
	)


def build_models() -> list[ModelDescriptor]:
	models: list[ModelDescriptor] = []
	for info in get_models_for_demo():
		if not is_model_locally_available(info.short_key):
			continue
		models.append(
			ModelDescriptor(
				shortKey=info.short_key,
				displayName=info.display_name,
				family=info.family,
				skeleton=info.skeleton,
				dataset=info.dataset,
				version=info.version,
				repoId=info.repo_id,
			)
		)
	return models


def get_checkpoint_root() -> Path | None:
	checkpoint_root = CHECKPOINT_ROOT or os.environ.get("CHECKPOINT_DIR", "")
	trimmed_root = checkpoint_root.strip()
	if not trimmed_root:
		return None
	return Path(trimmed_root)


def resolve_local_model_checkpoint_path(short_key: str) -> Path | None:
	trimmed_short_key = short_key.strip()
	if not trimmed_short_key:
		return None

	checkpoint_root = get_checkpoint_root()
	if checkpoint_root is None:
		return None

	info = get_model_info(trimmed_short_key)
	checkpoint_folder_name = info.display_name if info is not None else trimmed_short_key
	preferred_path = checkpoint_root / checkpoint_folder_name
	if preferred_path.exists():
		return preferred_path

	fallback_path = checkpoint_root / trimmed_short_key
	if fallback_path.exists():
		return fallback_path

	return None


def is_model_locally_available(short_key: str) -> bool:
	checkpoint_root = get_checkpoint_root()
	if checkpoint_root is None:
		return True
	return resolve_local_model_checkpoint_path(short_key) is not None


def has_model_descriptor(short_key: str) -> bool:
	trimmed_short_key = short_key.strip()
	if not trimmed_short_key:
		return False
	return any(descriptor.shortKey == trimmed_short_key for descriptor in build_models())


def get_examples_root() -> Path:
	return VENDOR_ROOT / "kimodo" / "assets" / "demo" / "examples"


def get_model_examples_dir(model_name: str) -> Path:
	return get_examples_root() / model_name


def list_examples_for_model(model_name: str) -> list[ExampleDescriptor]:
	if not model_name.strip():
		return []
	examples_dir = get_model_examples_dir(model_name.strip())
	if not examples_dir.exists():
		return []
	results: list[ExampleDescriptor] = []
	for path in sorted(examples_dir.iterdir()):
		if not path.is_dir():
			continue
		if not (path / "meta.json").exists():
			continue
		results.append(ExampleDescriptor(exampleName=path.name, model=model_name.strip()))
	return results


def load_prompts_from_meta(meta_path: Path, frames_per_second: float) -> list[PromptSegment]:
	if not meta_path.exists():
		return []
	meta = json.loads(meta_path.read_text(encoding="utf-8"))
	segments: list[PromptSegment] = []
	frame_cursor = 0
	if isinstance(meta.get("texts"), list):
		texts = [str(value) for value in meta.get("texts", [])]
		durations = [float(value) for value in meta.get("durations", [])]
		for prompt_text, duration in zip(texts, durations):
			frame_count = max(int(round(duration * frames_per_second)), 1)
			end_frame = frame_cursor + frame_count - 1
			segments.append(PromptSegment(prompt=prompt_text, startFrame=frame_cursor, endFrame=end_frame))
			frame_cursor = end_frame + 1
	elif isinstance(meta.get("text"), str):
		duration = float(meta.get("duration", 5.0))
		frame_count = max(int(round(duration * frames_per_second)), 1)
		segments.append(PromptSegment(prompt=meta["text"], startFrame=0, endFrame=frame_count - 1))
	return segments


def build_generation_request_from_session(session: SessionRecord, request: GenerateJobRequest) -> GenerateJobRequest:
	if request.model.strip():
		session.model = request.model.strip()
	elif session.model.strip():
		request.model = session.model.strip()

	if session.prompts and not request.prompts and not request.prompt.strip():
		request.bUseMultiPromptInput = len(session.prompts) > 1
		request.prompts = [segment.prompt for segment in session.prompts]
		request.durations = [max((segment.endFrame - segment.startFrame + 1) / max(session.frames_per_second, 1.0), 0.0) for segment in session.prompts]
		if len(session.prompts) == 1:
			request.prompt = session.prompts[0].prompt
			request.duration = request.durations[0] if request.durations else request.duration
		else:
			request.duration = request.durations[0] if request.durations else request.duration

	if not request.nativeConstraints and not request.constraintsPath.strip() and session.constraints_path.strip():
		request.constraintsPath = session.constraints_path

	if (
		not request.constraintsPath.strip()
		and not request.nativeConstraints
		and session.constraints
		and request.constraintSourceMotionPath.strip()
	):
		request.nativeConstraints = list(session.constraints)

	return request


JOBS: dict[str, JobRecord] = {}
SESSIONS: dict[str, SessionRecord] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EmbodiedSpaceStudioJobs")
APP = FastAPI(title="EmbodiedSpaceStudio Bridge Service", version=SERVICE_VERSION)


def write_text(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")


def write_json(path: Path, content: Any) -> None:
	write_text(path, json.dumps(content, indent=2))


def normalize_path_text(path_text: str) -> str:
	return path_text.replace("\\", "/").strip()


def append_display_path(base_path: str, *relative_parts: str) -> str:
	trimmed_base = base_path.rstrip("/\\")
	if not trimmed_base:
		return "/".join(part.strip("/\\") for part in relative_parts if part)
	return trimmed_base + "/" + "/".join(part.strip("/\\") for part in relative_parts if part)


def get_project_relative_saved_path(path_text: str) -> Path | None:
	normalized = normalize_path_text(path_text)
	if not normalized:
		return None
	parts = [part for part in normalized.split("/") if part]
	for index, part in enumerate(parts):
		if part.lower() == "saved":
			return Path(*parts[index:])
	return None


def resolve_bridge_file_path(path_text: str) -> Path:
	normalized = normalize_path_text(path_text)
	if not normalized:
		return Path()

	direct_path = Path(normalized)
	if direct_path.exists():
		return direct_path

	project_relative_saved_path = get_project_relative_saved_path(normalized)
	if project_relative_saved_path is not None:
		project_saved_path = PROJECT_ROOT / project_relative_saved_path
		if project_saved_path.exists():
			return project_saved_path

	return direct_path


def resolve_output_paths(output_dir_text: str, output_name: str, job_id: str) -> tuple[Path, str]:
	"""Return the Docker write path plus the path string UE should read.

	In the release Docker bridge, generation output is written under the
	KimodoBridge output bind mount instead of a specific Unreal project. UE only
	needs a host-readable path in the API response.
	"""
	output_folder_name = output_name.strip() or job_id
	if BRIDGE_OUTPUT_ROOT is not None:
		response_base = BRIDGE_OUTPUT_HOST_ROOT or str(BRIDGE_OUTPUT_ROOT)
		return BRIDGE_OUTPUT_ROOT / output_folder_name, append_display_path(response_base, output_folder_name)

	trimmed_output_dir = output_dir_text.strip()
	if not trimmed_output_dir:
		write_base = DEFAULT_OUTPUT_ROOT
		return write_base / output_folder_name, str(write_base / output_folder_name)

	normalized_output_dir = normalize_path_text(trimmed_output_dir)
	project_relative_saved_path = get_project_relative_saved_path(normalized_output_dir)
	if project_relative_saved_path is not None:
		write_base = PROJECT_ROOT / project_relative_saved_path
		return write_base / output_folder_name, append_display_path(trimmed_output_dir, output_folder_name)

	write_base = Path(trimmed_output_dir)
	return write_base / output_folder_name, str(write_base / output_folder_name)


def discover_outputs(job_root: Path, response_root: str = "") -> list[str]:
	results: list[str] = []
	for path in sorted(job_root.rglob("*")):
		if path.is_file() and path.name not in {"stdout.log", "stderr.log", "meta.json", "constraints.json"}:
			if response_root:
				try:
					relative_path = path.relative_to(job_root)
					results.append(append_display_path(response_root, *relative_path.parts))
				except ValueError:
					results.append(str(path))
			else:
				results.append(str(path))
	return results


def build_environment(request: GenerateJobRequest, output_dir: Path) -> dict[str, str]:
	environment = os.environ.copy()
	environment.setdefault("PYTHONPATH", str(VENDOR_ROOT))
	if environment["PYTHONPATH"] != str(VENDOR_ROOT):
		environment["PYTHONPATH"] = str(VENDOR_ROOT) + os.pathsep + environment["PYTHONPATH"]
	environment["TEXT_ENCODER_MODE"] = request.textEncoderMode or "local"
	environment["TEXT_ENCODER_DEVICE"] = request.textEncoderDevice or "auto"
	environment["LOCAL_CACHE"] = "true"
	environment["KIMODO_STRICT_LOCAL"] = "true"
	environment["HF_HUB_OFFLINE"] = "1"
	environment["HUGGINGFACE_HUB_OFFLINE"] = "1"
	environment["TRANSFORMERS_OFFLINE"] = "1"
	checkpoint_root = CHECKPOINT_ROOT or os.environ.get("CHECKPOINT_DIR", "")
	if checkpoint_root:
		environment["CHECKPOINT_DIR"] = checkpoint_root
	if TEXT_ENCODERS_ROOT:
		environment["TEXT_ENCODERS_DIR"] = TEXT_ENCODERS_ROOT
	environment["KIMODO_UE_OUTPUT_DIR"] = str(output_dir)
	return environment


def kimodo_generate_supports_root_margin() -> bool:
	generate_script = VENDOR_ROOT / "kimodo" / "scripts" / "generate.py"
	try:
		return '"--root_margin"' in generate_script.read_text(encoding="utf-8", errors="ignore")
	except OSError:
		return False


def kimodo_generate_supports_above_ground_offset() -> bool:
	generate_script = VENDOR_ROOT / "kimodo" / "scripts" / "generate.py"
	try:
		return '"--above_ground_offset"' in generate_script.read_text(encoding="utf-8", errors="ignore")
	except OSError:
		return False


def should_wait_for_text_encoder(environment: dict[str, str]) -> bool:
	mode = environment.get("TEXT_ENCODER_MODE", "auto").strip().lower()
	url = environment.get("TEXT_ENCODER_URL", "").strip()
	return mode in {"api", "auto"} and bool(url)


def get_cached_text_encoder_semantic_result(url: str) -> str | None:
	if not url:
		return None
	with TEXT_ENCODER_SEMANTIC_CACHE_LOCK:
		cached = TEXT_ENCODER_SEMANTIC_CACHE.get(url)
		if cached is None:
			return None
		cached_at, detail = cached
		if time.monotonic() - cached_at > TEXT_ENCODER_SEMANTIC_CACHE_TTL_SECONDS:
			TEXT_ENCODER_SEMANTIC_CACHE.pop(url, None)
			return None
		return detail


def cache_text_encoder_semantic_success(url: str, detail: str) -> None:
	if not url:
		return
	with TEXT_ENCODER_SEMANTIC_CACHE_LOCK:
		TEXT_ENCODER_SEMANTIC_CACHE[url] = (time.monotonic(), detail)


def is_text_encoder_ready(environment: dict[str, str]) -> tuple[bool, str]:
	url = environment.get("TEXT_ENCODER_URL", "").strip()
	if not url:
		return True, ""

	request = urllib.request.Request(url, method="GET")
	try:
		with urllib.request.urlopen(request, timeout=TEXT_ENCODER_CONNECT_TIMEOUT_SECONDS) as response:
			status_code = getattr(response, "status", 200)
			if 200 <= status_code < 500:
				return True, ""
			return False, f"HTTP {status_code}"
	except urllib.error.URLError as error:
		reason = getattr(error, "reason", error)
		return False, f"{type(reason).__name__}: {reason}"
	except Exception as error:
		return False, f"{type(error).__name__}: {error}"


def is_text_encoder_semantically_ready(environment: dict[str, str]) -> tuple[bool, str]:
	url = environment.get("TEXT_ENCODER_URL", "").strip()
	if not url:
		return True, ""

	cached_detail = get_cached_text_encoder_semantic_result(url)
	if cached_detail is not None:
		return True, cached_detail

	try:
		from gradio_client import Client

		client = Client(url, verbose=False)
		prompts = [
			"A person walks forward.",
			"A person jumps upward and lands.",
		]
		embeddings = []
		for index, prompt in enumerate(prompts):
			filename = f"ue_bridge_probe_{os.getpid()}_{index}.npy"
			result = client.predict(text=prompt, filename=filename, api_name="/DemoWrapper")
			output_path = result[0]["value"]
			embeddings.append(np.load(output_path).reshape(-1).astype(np.float32))
	except Exception as error:
		return False, f"semantic probe failed: {type(error).__name__}: {error}"

	if len(embeddings) < 2:
		return False, "semantic probe returned fewer than two embeddings"

	first, second = embeddings[0], embeddings[1]
	if first.size == 0 or second.size == 0:
		return False, "semantic probe returned an empty embedding"

	mean_abs_diff = float(np.mean(np.abs(first - second)))
	first_norm = float(np.linalg.norm(first))
	second_norm = float(np.linalg.norm(second))
	if first_norm <= 0.0 or second_norm <= 0.0:
		return False, "semantic probe returned a zero-norm embedding"

	cosine = float(np.dot(first, second) / (first_norm * second_norm))
	if mean_abs_diff < TEXT_ENCODER_SEMANTIC_MIN_MEAN_ABS_DIFF or cosine > TEXT_ENCODER_SEMANTIC_MAX_COSINE:
		return False, (
			"text encoder embeddings collapsed "
			f"(mean_abs_diff={mean_abs_diff:.3e}, cosine={cosine:.6f}); "
			"check TEXT_ENCODER_BASE_MODEL_HOST and TEXT_ENCODER_ADAPTER_HOST"
		)

	detail = f"semantic probe ok (mean_abs_diff={mean_abs_diff:.3e}, cosine={cosine:.6f})"
	cache_text_encoder_semantic_success(url, detail)
	return True, detail


def wait_for_text_encoder(environment: dict[str, str]) -> tuple[bool, str]:
	if not should_wait_for_text_encoder(environment):
		return True, ""

	url = environment.get("TEXT_ENCODER_URL", "").strip()
	deadline = time.monotonic() + TEXT_ENCODER_READY_TIMEOUT_SECONDS
	last_error = ""
	while time.monotonic() < deadline:
		is_ready, detail = is_text_encoder_ready(environment)
		if is_ready:
			semantic_ready, semantic_detail = is_text_encoder_semantically_ready(environment)
			if semantic_ready:
				return True, ""
			last_error = semantic_detail
		else:
			last_error = detail
		time.sleep(TEXT_ENCODER_READY_RETRY_DELAY_SECONDS)

	return False, f"Text encoder service at {url} did not become ready within {TEXT_ENCODER_READY_TIMEOUT_SECONDS:.0f}s ({last_error or 'no response'})."


def current_text_encoder_readiness() -> tuple[bool, str]:
	environment = os.environ.copy()
	if not should_wait_for_text_encoder(environment):
		return True, "not required"
	is_ready, detail = is_text_encoder_ready(environment)
	if is_ready:
		semantic_ready, semantic_detail = is_text_encoder_semantically_ready(environment)
		if semantic_ready:
			return True, semantic_detail or "ready"
		return False, semantic_detail
	return False, f"not ready: {detail or 'no response'}"


def normalize_constraint_track(track: str) -> str:
	normalized = "".join(char for char in track.lower() if char.isalnum())
	return {
		"2droot": "root2d",
		"root2d": "root2d",
		"fullbody": "fullbody",
		"lefthand": "left-hand",
		"righthand": "right-hand",
		"leftfoot": "left-foot",
		"rightfoot": "right-foot",
	}.get(normalized, "")


def should_force_postprocess_for_constraints(request: "GenerateJobRequest") -> bool:
	if request.nativeConstraints:
		return True
	return bool(request.constraintsPath.strip())


def is_postprocess_enabled(request: "GenerateJobRequest") -> bool:
	return (not request.bDisablePostprocess) or should_force_postprocess_for_constraints(request)


def build_prompt_metadata_fields(request: "GenerateJobRequest") -> dict[str, str]:
	fields: dict[str, str] = {}
	if request.promptCategory.strip():
		fields["category"] = request.promptCategory.strip()
	if request.promptSequenceId.strip():
		fields["sequence_id"] = request.promptSequenceId.strip()
	if request.promptSequence.strip():
		fields["sequence"] = request.promptSequence.strip()
	if request.promptScene.strip():
		fields["scene"] = request.promptScene.strip()
	return fields


def read_ue_vector(value: Any) -> np.ndarray | None:
	if isinstance(value, dict):
		try:
			return np.array(
				[
					float(value.get("x", value.get("X", 0.0))),
					float(value.get("y", value.get("Y", 0.0))),
					float(value.get("z", value.get("Z", 0.0))),
				],
				dtype=np.float32,
			)
		except (TypeError, ValueError):
			return None
	if isinstance(value, (list, tuple)) and len(value) >= 3:
		try:
			return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
		except (TypeError, ValueError):
			return None
	return None


def read_ue_quaternion_xyzw(value: Any) -> np.ndarray | None:
	if isinstance(value, dict):
		try:
			return np.array(
				[
					float(value.get("x", value.get("X", 0.0))),
					float(value.get("y", value.get("Y", 0.0))),
					float(value.get("z", value.get("Z", 0.0))),
					float(value.get("w", value.get("W", 1.0))),
				],
				dtype=np.float32,
			)
		except (TypeError, ValueError):
			return None
	if isinstance(value, (list, tuple)) and len(value) >= 4:
		try:
			return np.array([float(value[0]), float(value[1]), float(value[2]), float(value[3])], dtype=np.float32)
		except (TypeError, ValueError):
			return None
	return None


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray | None:
	norm = float(np.linalg.norm(quaternion))
	if norm <= 1e-8:
		return None
	x, y, z, w = (quaternion / norm).astype(np.float32)
	return np.array(
		[
			[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
			[2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
			[2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
		],
		dtype=np.float32,
	)


def ue_preview_rotation_to_kimodo_matrix(rotation_xyzw: np.ndarray) -> np.ndarray | None:
	rotation_ue = quaternion_xyzw_to_matrix(rotation_xyzw)
	if rotation_ue is None:
		return None
	basis = kimodo_to_ue_coord_rotation_matrix()
	return (basis.T @ rotation_ue @ basis).astype(np.float32)


def ue_preview_location_cm_to_kimodo_m(location_cm: np.ndarray, first_root_position_m: torch.Tensor) -> torch.Tensor:
	location_ue_m = location_cm.astype(np.float32) / 100.0
	location_kimodo_root_relative = kimodo_to_ue_coord_rotation_matrix().T @ location_ue_m
	root_origin = first_root_position_m.detach().cpu().numpy().astype(np.float32)
	return torch.tensor(location_kimodo_root_relative + root_origin, device=first_root_position_m.device, dtype=first_root_position_m.dtype)


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
	norm = torch.linalg.norm(vector)
	if norm <= torch.finfo(vector.dtype).eps:
		return vector
	return vector / norm


def _rotation_between_vectors(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
	current_normal = _normalize_vector(current)
	desired_normal = _normalize_vector(desired)
	dot_value = torch.clamp(torch.dot(current_normal, desired_normal), -1.0, 1.0)
	identity = torch.eye(3, device=current.device, dtype=current.dtype)
	if dot_value > 0.999999:
		return identity

	if dot_value < -0.999999:
		candidate_axis = torch.tensor([1.0, 0.0, 0.0], device=current.device, dtype=current.dtype)
		if torch.abs(current_normal[0]) > 0.9:
			candidate_axis = torch.tensor([0.0, 1.0, 0.0], device=current.device, dtype=current.dtype)
		axis = _normalize_vector(torch.cross(current_normal, candidate_axis, dim=0))
		return 2.0 * torch.outer(axis, axis) - identity

	axis = torch.cross(current_normal, desired_normal, dim=0)
	k_matrix = torch.zeros((3, 3), device=current.device, dtype=current.dtype)
	k_matrix[0, 1] = -axis[2]
	k_matrix[0, 2] = axis[1]
	k_matrix[1, 0] = axis[2]
	k_matrix[1, 2] = -axis[0]
	k_matrix[2, 0] = -axis[1]
	k_matrix[2, 1] = axis[0]
	return identity + k_matrix + (k_matrix @ k_matrix) * (1.0 / (1.0 + dot_value))


def _is_joint_descendant_or_self(parent_indices: list[int], joint_index: int, ancestor_index: int) -> bool:
	current_index = joint_index
	while current_index >= 0:
		if current_index == ancestor_index:
			return True
		current_index = parent_indices[current_index] if current_index < len(parent_indices) else -1
	return False


def _build_joint_chain_to_root(parent_indices: list[int], joint_index: int) -> list[int]:
	chain = []
	current_index = joint_index
	while current_index >= 0 and current_index < len(parent_indices):
		chain.append(current_index)
		current_index = parent_indices[current_index]
	chain.reverse()
	return chain


def _apply_rotation_to_pose_subtree(
	positions: torch.Tensor,
	rotations: torch.Tensor,
	parent_indices: list[int],
	pivot_joint_index: int,
	rotation_delta: torch.Tensor,
	*,
	rotate_pivot_position: bool,
) -> None:
	pivot_position = positions[pivot_joint_index].clone()
	for joint_index in range(int(positions.shape[0])):
		if not _is_joint_descendant_or_self(parent_indices, joint_index, pivot_joint_index):
			continue
		if rotate_pivot_position or joint_index != pivot_joint_index:
			positions[joint_index] = pivot_position + rotation_delta @ (positions[joint_index] - pivot_position)
		rotations[joint_index] = rotation_delta @ rotations[joint_index]


def apply_ue_constraint_edit_to_kimodo_pose(
	skeleton,
	positions: torch.Tensor,
	rotations: torch.Tensor,
	joint_index: int,
	desired_position_m: torch.Tensor | None,
	desired_rotation_matrix: np.ndarray | None,
) -> None:
	if joint_index < 0 or joint_index >= int(positions.shape[0]):
		return

	parent_indices = [int(value) for value in skeleton.joint_parents.detach().cpu().tolist()]
	if desired_position_m is not None:
		if joint_index == int(skeleton.root_idx):
			translation_delta = desired_position_m - positions[joint_index]
			positions += translation_delta.view(1, 3)
		else:
			chain = _build_joint_chain_to_root(parent_indices, joint_index)
			if len(chain) >= 2:
				for _ in range(12):
					if torch.sum((positions[joint_index] - desired_position_m) ** 2) <= 0.0025**2:
						break
					for pivot_joint_index in reversed(chain[:-1]):
						current_vector = positions[joint_index] - positions[pivot_joint_index]
						desired_vector = desired_position_m - positions[pivot_joint_index]
						if torch.sum(current_vector**2) <= torch.finfo(positions.dtype).eps or torch.sum(desired_vector**2) <= torch.finfo(positions.dtype).eps:
							continue
						rotation_delta = _rotation_between_vectors(current_vector, desired_vector)
						_apply_rotation_to_pose_subtree(
							positions,
							rotations,
							parent_indices,
							pivot_joint_index,
							rotation_delta,
							rotate_pivot_position=True,
						)
			else:
				translation_delta = desired_position_m - positions[joint_index]
				positions += translation_delta.view(1, 3)

	if desired_rotation_matrix is not None:
		rotation_delta = torch.tensor(desired_rotation_matrix, device=rotations.device, dtype=rotations.dtype)
		if not torch.allclose(rotation_delta, torch.eye(3, device=rotations.device, dtype=rotations.dtype), atol=1.0e-4):
			_apply_rotation_to_pose_subtree(
				positions,
				rotations,
				parent_indices,
				joint_index,
				rotation_delta,
				rotate_pivot_position=False,
			)


def get_constraint_primary_joint_index(skeleton, constraint_type: str) -> int:
	if constraint_type in {"root2d", "fullbody"}:
		return int(skeleton.root_idx)
	joint_name_by_type = {
		"left-hand": "LeftHand",
		"right-hand": "RightHand",
		"left-foot": "LeftFoot",
		"right-foot": "RightFoot",
	}
	return int(skeleton.bone_index.get(joint_name_by_type.get(constraint_type, ""), -1))


def build_constraints_from_source_motion(
	source_motion_path: Path,
	native_constraints: list[ConstraintItem],
	destination_path: Path,
	max_frame_count: int | None = None,
) -> str:
	if not native_constraints:
		return ""

	if not source_motion_path.exists():
		raise FileNotFoundError(f"Constraint source motion not found: {source_motion_path}")

	motion_dict, joint_count = load_kimodo_npz_as_torch(str(source_motion_path), ensure_complete=True)
	skeleton = build_skeleton(joint_count)
	posed_joints = motion_dict["posed_joints"]
	global_rot_mats = motion_dict["global_rot_mats"]
	smooth_root_pos = motion_dict.get("smooth_root_pos", motion_dict["root_positions"])
	frame_count = int(posed_joints.shape[0])
	if max_frame_count is not None and max_frame_count > 0:
		frame_count = min(frame_count, int(max_frame_count))
	if frame_count <= 0:
		return ""

	grouped_constraints: dict[str, dict[int, ConstraintItem]] = {}
	for constraint in native_constraints:
		constraint_type = normalize_constraint_track(constraint.track)
		if not constraint_type:
			continue
		start_frame = max(min(constraint.startFrame, frame_count - 1), 0)
		end_frame = max(start_frame, min(constraint.endFrame, frame_count - 1))
		track_constraints = grouped_constraints.setdefault(constraint_type, {})
		for frame_idx in range(start_frame, end_frame + 1):
			track_constraints[frame_idx] = constraint

	constraint_sets = []
	for constraint_type, constraints_by_frame in grouped_constraints.items():
		if not constraints_by_frame:
			continue

		frame_list = sorted(constraints_by_frame.keys())
		frame_indices = torch.tensor(frame_list, device=posed_joints.device, dtype=torch.long)
		sampled_positions = posed_joints.index_select(0, frame_indices)
		sampled_rotations = global_rot_mats.index_select(0, frame_indices)
		sampled_root_2d = smooth_root_pos.index_select(0, frame_indices)[:, [0, 2]]

		for local_frame_idx, frame_idx in enumerate(frame_list):
			constraint = constraints_by_frame[frame_idx]
			if not constraint.bHasEditedTransform:
				continue

			full_pose_snapshot = constraint_type == "fullbody" and len(constraint.editedJointTransforms) >= sampled_positions.shape[1]
			if full_pose_snapshot:
				for edited_joint in constraint.editedJointTransforms:
					try:
						edited_joint_index = int(edited_joint.get("jointIndex", -1))
					except (TypeError, ValueError):
						edited_joint_index = -1
					if not (0 <= edited_joint_index < sampled_positions.shape[1]):
						continue

					edited_joint_location_cm = read_ue_vector(edited_joint.get("locationCm", {}))
					if edited_joint_location_cm is not None:
						sampled_positions[local_frame_idx, edited_joint_index] = ue_preview_location_cm_to_kimodo_m(
							edited_joint_location_cm,
							posed_joints[0, skeleton.root_idx],
						)

					edited_joint_rotation_xyzw = read_ue_quaternion_xyzw(edited_joint.get("rotation", {}))
					if edited_joint_rotation_xyzw is not None:
						edited_joint_rotation_matrix = ue_preview_rotation_to_kimodo_matrix(edited_joint_rotation_xyzw)
						if edited_joint_rotation_matrix is not None:
							sampled_rotations[local_frame_idx, edited_joint_index] = torch.as_tensor(
								edited_joint_rotation_matrix,
								device=sampled_rotations.device,
								dtype=sampled_rotations.dtype,
							)

				sampled_root_2d[local_frame_idx] = sampled_positions[local_frame_idx, skeleton.root_idx, [0, 2]]
				continue

			# Preserve source-motion root rotation so that non-root-joint IK calls (which
			# walk the CCD chain all the way up to the root) don't inadvertently replace
			# the body's global orientation with a large IK-driven rotation.  We restore it
			# after all per-joint edits so the constraint keeps a natural body orientation
			# while the requested joint positions are still encoded.
			original_root_rotation = sampled_rotations[local_frame_idx, skeleton.root_idx].clone()

			use_primary_transform = len(constraint.editedJointTransforms) == 0
			edited_location_cm = read_ue_vector(constraint.editedLocationCm) if use_primary_transform else None
			joint_index = get_constraint_primary_joint_index(skeleton, constraint_type)
			edited_position_m = (
				ue_preview_location_cm_to_kimodo_m(edited_location_cm, posed_joints[0, skeleton.root_idx])
				if edited_location_cm is not None and 0 <= joint_index < sampled_positions.shape[1]
				else None
			)
			edited_rotation_xyzw = read_ue_quaternion_xyzw(constraint.editedRotation) if use_primary_transform else None
			edited_rotation_matrix = ue_preview_rotation_to_kimodo_matrix(edited_rotation_xyzw) if edited_rotation_xyzw is not None else None
			if edited_position_m is not None or edited_rotation_matrix is not None:
				apply_ue_constraint_edit_to_kimodo_pose(
					skeleton,
					sampled_positions[local_frame_idx],
					sampled_rotations[local_frame_idx],
					joint_index,
					edited_position_m,
					edited_rotation_matrix,
				)
				if constraint_type == "root2d" or joint_index == skeleton.root_idx:
					sampled_root_2d[local_frame_idx] = sampled_positions[local_frame_idx, skeleton.root_idx, [0, 2]]
			for edited_joint in constraint.editedJointTransforms:
				try:
					edited_joint_index = int(edited_joint.get("jointIndex", -1))
				except (TypeError, ValueError):
					edited_joint_index = -1
				edited_joint_location_cm = read_ue_vector(edited_joint.get("locationCm", {}))
				if not (0 <= edited_joint_index < sampled_positions.shape[1]):
					continue
				edited_joint_position_m = (
					ue_preview_location_cm_to_kimodo_m(edited_joint_location_cm, posed_joints[0, skeleton.root_idx])
					if edited_joint_location_cm is not None
					else None
				)
				edited_joint_rotation_xyzw = read_ue_quaternion_xyzw(edited_joint.get("rotation", {}))
				edited_joint_rotation_matrix = ue_preview_rotation_to_kimodo_matrix(edited_joint_rotation_xyzw) if edited_joint_rotation_xyzw is not None else None
				apply_ue_constraint_edit_to_kimodo_pose(
					skeleton,
					sampled_positions[local_frame_idx],
					sampled_rotations[local_frame_idx],
					edited_joint_index,
					edited_joint_position_m,
					edited_joint_rotation_matrix,
				)
				if constraint_type == "root2d" or edited_joint_index == skeleton.root_idx:
					sampled_root_2d[local_frame_idx] = sampled_positions[local_frame_idx, skeleton.root_idx, [0, 2]]

			# If non-root IK calls modified the root rotation (CCD chain reaches root),
			# restore it to preserve the source-motion body orientation.
			# Skip restore only if the user explicitly edited the root joint itself.
			explicitly_edited_root = any(
				int(ej.get("jointIndex", -1)) == int(skeleton.root_idx)
				for ej in constraint.editedJointTransforms
			)
			if not explicitly_edited_root:
				sampled_rotations[local_frame_idx, skeleton.root_idx] = original_root_rotation

		if constraint_type == "root2d":
			constraint_sets.append(Root2DConstraintSet(skeleton, frame_indices, sampled_root_2d))
		elif constraint_type == "fullbody":
			constraint_sets.append(
				FullBodyConstraintSet(
					skeleton,
					frame_indices,
					sampled_positions,
					sampled_rotations,
					smooth_root_2d=sampled_root_2d,
				)
			)
		elif constraint_type == "left-hand":
			constraint_sets.append(LeftHandConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))
		elif constraint_type == "right-hand":
			constraint_sets.append(RightHandConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))
		elif constraint_type == "left-foot":
			constraint_sets.append(LeftFootConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))
		elif constraint_type == "right-foot":
			constraint_sets.append(RightFootConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))

	if not constraint_sets:
		return ""

	save_constraints_lst(str(destination_path), constraint_sets)
	return str(destination_path)


def build_constraints_from_native_targets(
	native_constraints: list[ConstraintItem],
	destination_path: Path,
	target_bone_names: list[str] | None = None,
	target_root_bone_index: int = 0,
	max_frame_count: int | None = None,
) -> str:
	if not native_constraints:
		return ""

	max_joint_index = 0
	for constraint in native_constraints:
		for edited_joint in constraint.editedJointTransforms:
			try:
				max_joint_index = max(max_joint_index, int(edited_joint.get("jointIndex", -1)))
			except (TypeError, ValueError):
				pass
	joint_count = max(len(target_bone_names or []), max_joint_index + 1, 77)
	skeleton = build_skeleton(joint_count)
	frame_count = max_frame_count if max_frame_count is not None and max_frame_count > 0 else 1

	grouped_constraints: dict[str, dict[int, ConstraintItem]] = {}
	for constraint in native_constraints:
		constraint_type = normalize_constraint_track(constraint.track)
		if not constraint_type:
			continue
		start_frame = max(min(constraint.startFrame, frame_count - 1), 0)
		end_frame = max(start_frame, min(constraint.endFrame, frame_count - 1))
		track_constraints = grouped_constraints.setdefault(constraint_type, {})
		for frame_idx in range(start_frame, end_frame + 1):
			track_constraints[frame_idx] = constraint

	root_idx = int(target_root_bone_index) if 0 <= int(target_root_bone_index) < joint_count else int(skeleton.root_idx)
	constraint_sets = []
	for constraint_type, constraints_by_frame in grouped_constraints.items():
		frame_list = sorted(constraints_by_frame.keys())
		if not frame_list:
			continue

		frame_indices = torch.tensor(frame_list, dtype=torch.long)
		sampled_positions = torch.zeros((len(frame_list), joint_count, 3), dtype=torch.float32)
		sampled_rotations = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(len(frame_list), joint_count, 1, 1)
		root_origin = torch.zeros(3, dtype=torch.float32)

		for local_frame_idx, frame_idx in enumerate(frame_list):
			constraint = constraints_by_frame[frame_idx]
			if not constraint.bHasEditedTransform:
				continue

			if constraint.editedJointTransforms:
				edited_joints = constraint.editedJointTransforms
			else:
				edited_joints = [
					{
						"jointIndex": get_constraint_primary_joint_index(skeleton, constraint_type),
						"locationCm": constraint.editedLocationCm,
						"rotation": constraint.editedRotation,
					}
				]

			for edited_joint in edited_joints:
				try:
					edited_joint_index = int(edited_joint.get("jointIndex", -1))
				except (TypeError, ValueError):
					edited_joint_index = -1
				if not (0 <= edited_joint_index < joint_count):
					continue

				edited_joint_location_cm = read_ue_vector(edited_joint.get("locationCm", {}))
				if edited_joint_location_cm is not None:
					sampled_positions[local_frame_idx, edited_joint_index] = ue_preview_location_cm_to_kimodo_m(edited_joint_location_cm, root_origin)

				edited_joint_rotation_xyzw = read_ue_quaternion_xyzw(edited_joint.get("rotation", {}))
				edited_joint_rotation_matrix = ue_preview_rotation_to_kimodo_matrix(edited_joint_rotation_xyzw) if edited_joint_rotation_xyzw is not None else None
				if edited_joint_rotation_matrix is not None:
					sampled_rotations[local_frame_idx, edited_joint_index] = torch.tensor(edited_joint_rotation_matrix, dtype=torch.float32)

		sampled_root_2d = sampled_positions[:, root_idx, [0, 2]]
		if constraint_type == "root2d":
			constraint_sets.append(Root2DConstraintSet(skeleton, frame_indices, sampled_root_2d))
		elif constraint_type == "fullbody":
			constraint_sets.append(FullBodyConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, smooth_root_2d=sampled_root_2d))
		elif constraint_type == "left-hand":
			constraint_sets.append(LeftHandConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))
		elif constraint_type == "right-hand":
			constraint_sets.append(RightHandConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))
		elif constraint_type == "left-foot":
			constraint_sets.append(LeftFootConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))
		elif constraint_type == "right-foot":
			constraint_sets.append(RightFootConstraintSet(skeleton, frame_indices, sampled_positions, sampled_rotations, sampled_root_2d))

	if not constraint_sets:
		return ""

	save_constraints_lst(str(destination_path), constraint_sets)
	return str(destination_path)


def estimate_request_frame_count(request: GenerateJobRequest, fps: float = 30.0) -> int:
	if request.prompts:
		durations = request.durations or [request.duration for _ in request.prompts]
		return max(1, int(sum(max(float(duration), 0.0) for duration in durations) * fps))
	return max(1, int(max(float(request.duration), 0.0) * fps))


def build_input_folder(job_root: Path, request: GenerateJobRequest, should_cancel=None) -> Path:
	input_dir = job_root / "input"
	if input_dir.exists():
		shutil.rmtree(input_dir)
	input_dir.mkdir(parents=True, exist_ok=True)
	if request.prompts:
		meta = {
			"texts": request.prompts,
			"durations": request.durations or [request.duration for _ in request.prompts],
		}
	else:
		meta = {
			"text": request.prompt,
			"duration": request.duration,
		}
	meta.update(build_prompt_metadata_fields(request))
	postprocess_enabled = is_postprocess_enabled(request)
	meta["num_samples"] = request.numSamples
	meta["diffusion_steps"] = request.diffusionSteps
	meta["foot_contact_stabilization"] = {
		"enabled": request.bStabilizeFootContacts,
		"level": max(1, min(int(request.footContactStabilizationLevel), 2)),
	}
	meta["postprocess"] = {
		"enabled": postprocess_enabled,
		"root_margin": request.postprocessRootMargin,
		"above_ground_offset": request.postprocessAboveGroundOffset,
	}
	if postprocess_enabled and request.bDisablePostprocess and should_force_postprocess_for_constraints(request):
		meta["postprocess"]["forced_for_constraints"] = True
	meta["source_character"] = {
		"use_kimodo_default": request.bUseKimodoDefaultSourceCharacter,
		"label": request.sourceCharacterLabel or "SOMA Human Body",
		"skeletal_mesh_path": request.sourceSkeletalMeshPath,
		"skeleton_path": request.sourceSkeletonPath,
	}
	if request.bUseSeed:
		meta["seed"] = request.seed
	if request.cfgType:
		if request.cfgType == "nocfg":
			meta["cfg"] = {"enabled": False}
		elif request.cfgType == "regular" and len(request.cfgWeight) == 1:
			meta["cfg"] = {
				"enabled": True,
				"text_weight": request.cfgWeight[0],
				"constraint_weight": request.cfgWeight[0],
			}
		elif request.cfgType == "separated" and len(request.cfgWeight) == 2:
			meta["cfg"] = {
				"enabled": True,
				"text_weight": request.cfgWeight[0],
				"constraint_weight": request.cfgWeight[1],
			}
	write_text(input_dir / "meta.json", json.dumps(meta, indent=2))
	constraints_output_path = input_dir / "constraints.json"
	if request.constraintsPath:
		constraints_src = Path(request.constraintsPath)
		if constraints_src.exists():
			constraints_output_path.write_bytes(constraints_src.read_bytes())
			write_text(
				input_dir / "constraints_summary.json",
				json.dumps(
					{
						"mode": "external",
						"constraintsPath": str(constraints_src),
						"constraintsWritten": str(constraints_output_path),
					},
					indent=2,
				),
			)
	elif request.nativeConstraints and request.constraintSourceMotionPath.strip():
		estimated_frame_count = estimate_request_frame_count(request)
		constraint_source_motion_path = resolve_bridge_file_path(request.constraintSourceMotionPath)
		constraints_summary_path = input_dir / "constraints_summary.json"
		write_text(
			constraints_summary_path,
			json.dumps(
				{
					"mode": "nativeTimeline",
					"status": "preprocessing",
					"constraintSourceMotionPath": request.constraintSourceMotionPath,
					"resolvedConstraintSourceMotionPath": str(constraint_source_motion_path),
					"nativeConstraintItems": len(request.nativeConstraints),
					"estimatedGenerationFrames": estimated_frame_count,
					"tracks": sorted({constraint.track for constraint in request.nativeConstraints}),
				},
				indent=2,
			),
		)
		if should_cancel is not None and should_cancel():
			return input_dir
		written_constraints_path = build_constraints_from_source_motion(
			constraint_source_motion_path,
			request.nativeConstraints,
			constraints_output_path,
			max_frame_count=estimated_frame_count,
		)
		write_text(
			constraints_summary_path,
			json.dumps(
				{
					"mode": "nativeTimeline",
					"status": "completed",
					"constraintSourceMotionPath": request.constraintSourceMotionPath,
					"resolvedConstraintSourceMotionPath": str(constraint_source_motion_path),
					"nativeConstraintItems": len(request.nativeConstraints),
					"estimatedGenerationFrames": estimated_frame_count,
					"constraintsWritten": written_constraints_path,
					"tracks": sorted({constraint.track for constraint in request.nativeConstraints}),
				},
				indent=2,
			),
		)
	elif request.nativeConstraints:
		estimated_frame_count = estimate_request_frame_count(request)
		constraints_summary_path = input_dir / "constraints_summary.json"
		write_text(
			constraints_summary_path,
			json.dumps(
				{
					"mode": "nativeDirect",
					"status": "preprocessing",
					"nativeConstraintItems": len(request.nativeConstraints),
					"estimatedGenerationFrames": estimated_frame_count,
					"tracks": sorted({constraint.track for constraint in request.nativeConstraints}),
				},
				indent=2,
			),
		)
		if should_cancel is not None and should_cancel():
			return input_dir
		written_constraints_path = build_constraints_from_native_targets(
			request.nativeConstraints,
			constraints_output_path,
			target_bone_names=request.targetBoneNames,
			target_root_bone_index=request.targetRootBoneIndex,
			max_frame_count=estimated_frame_count,
		)
		write_text(
			constraints_summary_path,
			json.dumps(
				{
					"mode": "nativeDirect",
					"status": "completed",
					"nativeConstraintItems": len(request.nativeConstraints),
					"estimatedGenerationFrames": estimated_frame_count,
					"constraintsWritten": written_constraints_path,
					"tracks": sorted({constraint.track for constraint in request.nativeConstraints}),
				},
				indent=2,
			),
		)
	return input_dir


def kimodo_to_ue_coord_rotation_matrix() -> np.ndarray:
	"""Rotate Kimodo's Y-up, +Z forward convention into the UE glTF-imported SOMA mesh convention.

	The Kimodo demo previews SOMA directly in its native Y-up space. The SOMA
	glTF source asset is imported by UE with X preserved, Y mapped to Z-up, and
	the original +Z forward axis landing on UE +Y. Matching that convention keeps
	the generated root trajectory aligned with the imported SOMA skinned mesh.
	"""
	return np.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0],
			[0.0, 1.0, 0.0],
		],
		dtype=np.float32,
	)


def transform_points_to_ue(points: np.ndarray) -> np.ndarray:
	basis = kimodo_to_ue_coord_rotation_matrix()
	return np.einsum("ab,...b->...a", basis, points).astype(np.float32)


def transform_rotations_to_ue(rotations: np.ndarray) -> np.ndarray:
	basis = kimodo_to_ue_coord_rotation_matrix()
	return np.einsum("ab,...bc,dc->...ad", basis, rotations, basis).astype(np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
	length = float(np.linalg.norm(vector))
	if length <= 1e-8:
		return np.zeros_like(vector, dtype=np.float32)
	return (vector / length).astype(np.float32)


def horizontal_normal(vector: np.ndarray) -> np.ndarray:
	horizontal = np.array([vector[0], vector[1], 0.0], dtype=np.float32)
	return normalize_vector(horizontal)


def build_motion_diagnostics(
	motion_path: Path,
	bone_names: list[str],
	root_bone_index: int,
	raw_joint_positions: np.ndarray,
	ue_joint_positions: np.ndarray,
) -> dict[str, object]:
	name_to_index = {name: index for index, name in enumerate(bone_names)}

	def joint_position_summary(joint_positions: np.ndarray) -> dict[str, list[float]]:
		result: dict[str, list[float]] = {}
		for name in ["Hips", "Head", "LeftShoulder", "RightShoulder", "LeftFoot", "RightFoot"]:
			index = name_to_index.get(name)
			if index is not None:
				result[name] = joint_positions[0, index].astype(float).tolist()
		return result

	def anatomy_forward(joint_positions: np.ndarray) -> list[float]:
		required = ["Hips", "Head", "LeftShoulder", "RightShoulder"]
		if any(name not in name_to_index for name in required):
			return [0.0, 0.0, 0.0]
		hips = joint_positions[0, name_to_index["Hips"]]
		head = joint_positions[0, name_to_index["Head"]]
		left_shoulder = joint_positions[0, name_to_index["LeftShoulder"]]
		right_shoulder = joint_positions[0, name_to_index["RightShoulder"]]
		up_axis = normalize_vector(head - hips)
		left_axis = normalize_vector(left_shoulder - right_shoulder)
		forward_axis = normalize_vector(np.cross(left_axis, up_axis))
		return forward_axis.astype(float).tolist()

	raw_root_delta = raw_joint_positions[-1, root_bone_index] - raw_joint_positions[0, root_bone_index]
	ue_root_delta = ue_joint_positions[-1, root_bone_index] - ue_joint_positions[0, root_bone_index]
	ue_forward = np.asarray(anatomy_forward(ue_joint_positions), dtype=np.float32)
	ue_root_heading = horizontal_normal(ue_root_delta)
	forward_heading_dot = float(np.dot(horizontal_normal(ue_forward), ue_root_heading))
	return {
		"sourceMotionPath": str(motion_path),
		"rootBoneIndex": root_bone_index,
		"rootBoneName": bone_names[root_bone_index] if 0 <= root_bone_index < len(bone_names) else "",
		"rawKimodoRootDeltaMeters": raw_root_delta.astype(float).tolist(),
		"ueRootDeltaCm": ue_root_delta.astype(float).tolist(),
		"rawKimodoFirstFrameJointsMeters": joint_position_summary(raw_joint_positions),
		"ueFirstFrameJointsCm": joint_position_summary(ue_joint_positions),
		"rawKimodoAnatomicalForward": anatomy_forward(raw_joint_positions),
		"ueAnatomicalForward": ue_forward.astype(float).tolist(),
		"ueRootHeading": ue_root_heading.astype(float).tolist(),
		"ueForwardHeadingDotRootHeading": forward_heading_dot,
	}


def find_primary_motion_npz_files(output_root: Path) -> list[Path]:
	candidates: list[Path] = []
	for path in sorted(output_root.glob("*.npz")):
		if "amass" not in path.stem.lower():
			candidates.append(path)

	motion_dir = output_root / "motion"
	if motion_dir.exists():
		for path in sorted(motion_dir.glob("*.npz")):
			if "amass" not in path.stem.lower():
				candidates.append(path)

	unique_paths: list[Path] = []
	seen: set[Path] = set()
	for path in candidates:
		if path not in seen:
			seen.add(path)
			unique_paths.append(path)
	return unique_paths


def _runs_from_bool_sequence(values: np.ndarray) -> list[tuple[bool, int, int]]:
	if values.size == 0:
		return []
	runs: list[tuple[bool, int, int]] = []
	start = 0
	current = bool(values[0])
	for index in range(1, int(values.shape[0])):
		next_value = bool(values[index])
		if next_value != current:
			runs.append((current, start, index - 1))
			start = index
			current = next_value
	runs.append((current, start, int(values.shape[0]) - 1))
	return runs


def _clean_contact_side(values: np.ndarray, max_spike_frames: int = 2, max_gap_frames: int = 2) -> np.ndarray:
	cleaned = np.asarray(values, dtype=bool).copy()
	for is_contact, start, end in _runs_from_bool_sequence(cleaned):
		length = end - start + 1
		is_interior = start > 0 and end + 1 < cleaned.shape[0]
		if not is_interior:
			continue
		if is_contact and length <= max_spike_frames:
			cleaned[start : end + 1] = False
		elif not is_contact and length <= max_gap_frames:
			cleaned[start : end + 1] = True
	return cleaned


def _side_contact_indices(channel_count: int) -> tuple[list[int], list[int]]:
	if channel_count >= 6:
		return [0, 1, 2], [3, 4, 5]
	if channel_count >= 4:
		return [0, 1], [2, 3]
	return [], []


def _stabilize_foot_contacts_array(foot_contacts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
	if foot_contacts.ndim != 2:
		return foot_contacts, np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), False
	left_indices, right_indices = _side_contact_indices(int(foot_contacts.shape[1]))
	if not left_indices or not right_indices:
		return foot_contacts, np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), False

	left_contact = np.any(foot_contacts[:, left_indices] > 0.5, axis=1)
	right_contact = np.any(foot_contacts[:, right_indices] > 0.5, axis=1)
	left_clean = _clean_contact_side(left_contact)
	right_clean = _clean_contact_side(right_contact)

	stabilized = foot_contacts.copy()
	stabilized[:, left_indices] = left_clean[:, None]
	stabilized[:, right_indices] = right_clean[:, None]
	return stabilized.astype(foot_contacts.dtype, copy=False), left_clean, right_clean, bool(
		np.any(left_clean != left_contact) or np.any(right_clean != right_contact)
	)


def _apply_xy_foot_lock_for_sliding(arrays: dict[str, np.ndarray], left_contact: np.ndarray, right_contact: np.ndarray) -> tuple[bool, list[dict[str, object]]]:
	posed_joints = arrays.get("posed_joints")
	if posed_joints is None or posed_joints.ndim != 3 or posed_joints.shape[1] < 77:
		return False, []
	frame_count = int(posed_joints.shape[0])
	if frame_count == 0 or left_contact.shape[0] != frame_count or right_contact.shape[0] != frame_count:
		return False, []

	left_foot_indices = [69, 70, 71]
	right_foot_indices = [74, 75, 76]
	min_segment_frames = 3
	fade_frames = 2
	fps = 30.0
	drift_threshold_m = 0.025
	speed_threshold_mps = 0.08

	def sole_xz(frame_index: int, b_left: bool) -> np.ndarray:
		indices = left_foot_indices if b_left else right_foot_indices
		return posed_joints[frame_index, indices][:, [0, 2]].mean(axis=0)

	dominant_per_frame: list[str | None] = [None] * frame_count
	dominant_side: str | None = None
	prev_left = False
	prev_right = False

	for frame_index in range(frame_count):
		left = bool(left_contact[frame_index])
		right = bool(right_contact[frame_index])
		left_landed = left and not prev_left
		right_landed = right and not prev_right

		if dominant_side == "left" and not left:
			dominant_side = None
		elif dominant_side == "right" and not right:
			dominant_side = None

		if dominant_side is None:
			if left:
				dominant_side = "left"
			elif right:
				dominant_side = "right"
		elif left and right:
			if dominant_side == "left" and right_landed:
				dominant_side = "right"
			elif dominant_side == "right" and left_landed:
				dominant_side = "left"

		dominant_per_frame[frame_index] = dominant_side

		prev_left = left
		prev_right = right

	offsets = np.zeros((frame_count, 2), dtype=np.float32)
	segments: list[dict[str, object]] = []
	frame_index = 0
	while frame_index < frame_count:
		side = dominant_per_frame[frame_index]
		if side is None:
			frame_index += 1
			continue
		segment_start = frame_index
		while frame_index + 1 < frame_count and dominant_per_frame[frame_index + 1] == side:
			frame_index += 1
		segment_end = frame_index
		frame_index += 1

		segment_length = segment_end - segment_start + 1
		if segment_length < min_segment_frames:
			continue

		b_left_side = side == "left"
		positions = np.stack([sole_xz(index, b_left_side) for index in range(segment_start, segment_end + 1)], axis=0)
		anchor_index = min(segment_start + 1, segment_end)
		anchor = sole_xz(anchor_index, b_left_side)
		drifts = np.linalg.norm(positions - anchor[None, :], axis=1)
		step_speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps if segment_length > 1 else np.zeros((0,), dtype=np.float32)
		max_drift = float(np.max(drifts)) if drifts.size else 0.0
		max_speed = float(np.max(step_speeds)) if step_speeds.size else 0.0
		mean_speed = float(np.mean(step_speeds)) if step_speeds.size else 0.0
		should_lock = max_drift >= drift_threshold_m or max_speed >= speed_threshold_mps
		segments.append(
			{
				"side": side,
				"startFrame": segment_start,
				"endFrame": segment_end,
				"frames": segment_length,
				"maxDriftMeters": max_drift,
				"maxSpeedMetersPerSecond": max_speed,
				"meanSpeedMetersPerSecond": mean_speed,
				"locked": should_lock,
			}
		)
		if not should_lock:
			continue

		for local_index, absolute_index in enumerate(range(segment_start, segment_end + 1)):
			edge_distance = min(local_index + 1, segment_length - local_index)
			fade_alpha = min(1.0, edge_distance / max(1, fade_frames))
			offsets[absolute_index] = (anchor - positions[local_index]) * fade_alpha

	if not np.any(np.abs(offsets) > 1e-6):
		return False, segments

	arrays["posed_joints"] = posed_joints.copy()
	arrays["posed_joints"][:, :, 0] += offsets[:, 0:1]
	arrays["posed_joints"][:, :, 2] += offsets[:, 1:2]
	for key in ("root_positions", "smooth_root_pos"):
		if key in arrays and arrays[key].ndim == 2 and arrays[key].shape[0] == frame_count and arrays[key].shape[1] >= 3:
			arrays[key] = arrays[key].copy()
			arrays[key][:, 0] += offsets[:, 0]
			arrays[key][:, 2] += offsets[:, 1]
	return True, segments


def stabilize_motion_npz_file(motion_path: Path, level: int) -> dict[str, object]:
	with np.load(motion_path, allow_pickle=False) as loaded:
		arrays = {key: loaded[key] for key in loaded.files}
	foot_contacts = arrays.get("foot_contacts")
	if foot_contacts is None:
		return {"path": str(motion_path), "changed": False, "reason": "missing foot_contacts"}

	stabilized_contacts, left_contact, right_contact, contacts_changed = _stabilize_foot_contacts_array(foot_contacts)
	if left_contact.size == 0 or right_contact.size == 0:
		return {"path": str(motion_path), "changed": False, "reason": "unsupported foot contact shape"}
	arrays["foot_contacts"] = stabilized_contacts
	pinning_changed = False
	sliding_segments: list[dict[str, object]] = []
	if level >= 2:
		pinning_changed, sliding_segments = _apply_xy_foot_lock_for_sliding(arrays, left_contact, right_contact)
	if not contacts_changed and not pinning_changed:
		return {
			"path": str(motion_path),
			"changed": False,
			"level": level,
			"contactsChanged": contacts_changed,
			"slidingSegments": sliding_segments,
		}

	temp_path = motion_path.with_name(motion_path.stem + ".stabilized.tmp.npz")
	np.savez(temp_path, **arrays)
	temp_path.replace(motion_path)
	return {
		"path": str(motion_path),
		"changed": True,
		"level": level,
		"contactsChanged": contacts_changed,
		"pinningChanged": pinning_changed,
		"slidingSegments": sliding_segments,
	}


def preserve_raw_smooth_root_npz_file(motion_path: Path) -> dict[str, object]:
	with np.load(motion_path, allow_pickle=False) as loaded:
		arrays = {key: loaded[key] for key in loaded.files}
	if "raw_smooth_root_pos" in arrays:
		return {"path": str(motion_path), "changed": False, "reason": "already present"}
	if "smooth_root_pos" not in arrays:
		return {"path": str(motion_path), "changed": False, "reason": "missing smooth_root_pos"}
	arrays["raw_smooth_root_pos"] = arrays["smooth_root_pos"].copy()
	temp_path = motion_path.with_name(motion_path.stem + ".raw_smooth.tmp.npz")
	np.savez(temp_path, **arrays)
	temp_path.replace(motion_path)
	return {"path": str(motion_path), "changed": True}


def preserve_raw_smooth_root_outputs(output_root: Path) -> None:
	reports: list[dict[str, object]] = []
	for motion_path in sorted(output_root.rglob("*.npz")):
		try:
			reports.append(preserve_raw_smooth_root_npz_file(motion_path))
		except Exception as error:
			reports.append({"path": str(motion_path), "changed": False, "error": f"{type(error).__name__}: {error}"})
	if reports:
		write_json(output_root / "raw_smooth_root_preservation.json", {"reports": reports})


def stabilize_motion_outputs(output_root: Path, request: GenerateJobRequest) -> None:
	if not request.bStabilizeFootContacts:
		return
	level = max(1, min(int(request.footContactStabilizationLevel), 2))
	reports: list[dict[str, object]] = []
	for motion_path in sorted(output_root.rglob("*.npz")):
		if "amass" in motion_path.stem.lower():
			continue
		try:
			reports.append(stabilize_motion_npz_file(motion_path, level))
		except Exception as exc:
			reports.append(
				{
					"path": str(motion_path),
					"changed": False,
					"error": f"{type(exc).__name__}: {exc}",
				}
			)
	write_text(
		output_root / "foot_contact_stabilization.json",
		json.dumps(
			{
				"enabled": True,
				"level": level,
				"files": reports,
			},
			indent=2,
		),
	)


def normalize_target_bone_mapping(target_bone_names: list[str], target_root_bone_index: int, joint_count: int) -> tuple[list[str], int]:
	if joint_count <= 0 or len(target_bone_names) != joint_count:
		return [], 0

	normalized_bone_names: list[str] = []
	for bone_name in target_bone_names:
		trimmed_bone_name = str(bone_name).strip()
		if not trimmed_bone_name:
			return [], 0
		normalized_bone_names.append(trimmed_bone_name)

	return normalized_bone_names, max(0, min(int(target_root_bone_index), joint_count - 1))


def build_preview_sample(
	motion_path: Path,
	target_bone_names: list[str] | None = None,
	target_root_bone_index: int = 0,
) -> tuple[dict[str, object], list[str], int, list[str], int]:
	motion_dict, joint_count = load_kimodo_npz_as_torch(str(motion_path), ensure_complete=True)
	skeleton = build_skeleton(joint_count)
	if joint_count == 30 and hasattr(skeleton, "output_to_SOMASkeleton77"):
		motion_dict = skeleton.output_to_SOMASkeleton77(motion_dict)
		skeleton = skeleton.somaskel77
	normalized_target_bone_names, normalized_target_root_bone_index = normalize_target_bone_mapping(
		target_bone_names or [],
		target_root_bone_index,
		len(skeleton.bone_order_names),
	)
	joint_positions_tensor = motion_dict["posed_joints"]
	global_rot_mats_tensor = motion_dict["global_rot_mats"]
	smooth_root_tensor = motion_dict.get("smooth_root_pos", motion_dict["root_positions"])
	raw_smooth_root_tensor = motion_dict.get("raw_smooth_root_pos")
	foot_contacts_tensor = motion_dict.get("foot_contacts")
	global_root_heading_tensor = motion_dict.get("global_root_heading")
	joint_positions = joint_positions_tensor.detach().cpu().numpy().astype(np.float32)
	global_rot_mats = global_rot_mats_tensor.detach().cpu().numpy().astype(np.float32)
	smooth_root_pos = smooth_root_tensor.detach().cpu().numpy().astype(np.float32)
	raw_smooth_root_pos = raw_smooth_root_tensor.detach().cpu().numpy().astype(np.float32) if raw_smooth_root_tensor is not None else None
	joint_positions_ue = transform_points_to_ue(joint_positions)
	joint_positions_ue = (joint_positions_ue - joint_positions_ue[0:1, skeleton.root_idx : skeleton.root_idx + 1, :]) * 100.0
	smooth_root_pos_ue = transform_points_to_ue(smooth_root_pos)
	smooth_root_pos_ue = (smooth_root_pos_ue - transform_points_to_ue(joint_positions[0:1, skeleton.root_idx : skeleton.root_idx + 1, :])[0, 0]) * 100.0
	raw_smooth_root_pos_ue = None
	if raw_smooth_root_pos is not None:
		raw_smooth_root_pos_ue = transform_points_to_ue(raw_smooth_root_pos)
		raw_smooth_root_pos_ue = (raw_smooth_root_pos_ue - transform_points_to_ue(joint_positions[0:1, skeleton.root_idx : skeleton.root_idx + 1, :])[0, 0]) * 100.0
	global_rot_mats_ue = transform_rotations_to_ue(global_rot_mats)
	global_quats_wxyz = matrix_to_quaternion(torch.from_numpy(global_rot_mats_ue)).cpu().numpy().astype(np.float32)
	global_quats_xyzw = np.concatenate([global_quats_wxyz[..., 1:], global_quats_wxyz[..., 0:1]], axis=-1)
	foot_contacts = None
	if foot_contacts_tensor is not None:
		foot_contacts = foot_contacts_tensor.detach().cpu().numpy().astype(np.float32)
	global_root_heading = None
	if global_root_heading_tensor is not None:
		global_root_heading = global_root_heading_tensor.detach().cpu().numpy().astype(np.float32)

	sample: dict[str, object] = {
		"label": motion_path.stem,
		"sourceMotionPath": str(motion_path),
		"framesPerSecond": 30.0,
		"jointPositionsCm": joint_positions_ue.tolist(),
		"globalRotationsXyzw": global_quats_xyzw.tolist(),
		"smoothRootPositionsCm": smooth_root_pos_ue.tolist(),
	}
	if raw_smooth_root_pos_ue is not None:
		sample["rawSmoothRootPositionsCm"] = raw_smooth_root_pos_ue.tolist()
	if foot_contacts is not None:
		sample["footContacts"] = foot_contacts.tolist()
	if global_root_heading is not None:
		sample["globalRootHeading"] = global_root_heading.tolist()

	return (
		sample,
		skeleton.bone_order_names,
		skeleton.root_idx,
		normalized_target_bone_names,
		normalized_target_root_bone_index,
	)


def write_preview_manifest(
	output_path: Path,
	motion_paths: list[Path],
	target_bone_names: list[str] | None = None,
	target_root_bone_index: int = 0,
) -> str:
	if not motion_paths:
		return ""

	bone_names: list[str] = []
	root_bone_index = 0
	preview_target_bone_names: list[str] = []
	preview_target_root_bone_index = 0
	samples: list[dict[str, object]] = []
	diagnostics: list[dict[str, object]] = []
	for motion_path in motion_paths:
		sample, sample_bone_names, sample_root_bone_index, sample_target_bone_names, sample_target_root_bone_index = build_preview_sample(
			motion_path,
			target_bone_names=target_bone_names,
			target_root_bone_index=target_root_bone_index,
		)
		if not bone_names:
			bone_names = list(sample_bone_names)
			root_bone_index = sample_root_bone_index
			preview_target_bone_names = list(sample_target_bone_names)
			preview_target_root_bone_index = sample_target_root_bone_index
		elif sample_bone_names != bone_names:
			continue
		samples.append(sample)
		try:
			motion_dict, diagnostic_joint_count = load_kimodo_npz_as_torch(str(motion_path), ensure_complete=True)
			diagnostic_skeleton = build_skeleton(diagnostic_joint_count)
			if diagnostic_joint_count == 30 and hasattr(diagnostic_skeleton, "output_to_SOMASkeleton77"):
				motion_dict = diagnostic_skeleton.output_to_SOMASkeleton77(motion_dict)
			raw_positions = motion_dict["posed_joints"].detach().cpu().numpy().astype(np.float32)
			ue_positions = np.asarray(sample["jointPositionsCm"], dtype=np.float32)
			diagnostics.append(build_motion_diagnostics(motion_path, sample_bone_names, sample_root_bone_index, raw_positions, ue_positions))
		except Exception as exc:
			diagnostics.append({"sourceMotionPath": str(motion_path), "error": f"{type(exc).__name__}: {exc}"})

	if not samples:
		return ""

	manifest = {
		"version": 1,
		"sourceCoordinateSystem": "Kimodo_Yup_Zforward_Meters",
		"targetCoordinateSystem": "UE_Zup_KimodoSomaGltfImport_Centimeters",
		"rootBoneIndex": root_bone_index,
		"boneNames": bone_names,
		"clips": samples,
	}
	if preview_target_bone_names:
		manifest["targetRootBoneIndex"] = preview_target_root_bone_index
		manifest["targetBoneNames"] = preview_target_bone_names
	write_text(output_path, json.dumps(manifest))
	if diagnostics:
		write_text(
			output_path.with_name(UE_MOTION_DIAGNOSTICS_NAME),
			json.dumps(
			{
				"version": 1,
				"sourceCoordinateSystem": "Kimodo_Yup_Zforward_Meters",
				"targetCoordinateSystem": "UE_Zup_KimodoSomaGltfImport_Centimeters",
				"clips": diagnostics,
			},
				indent=2,
			),
		)
	return str(output_path)


def try_write_preview_manifest_for_output_root(output_root: Path, request: GenerateJobRequest | None = None) -> str:
	try:
		motion_paths = find_primary_motion_npz_files(output_root)
		return write_preview_manifest(
			output_root / UE_PREVIEW_MANIFEST_NAME,
			motion_paths,
			target_bone_names=request.targetBoneNames if request is not None else None,
			target_root_bone_index=request.targetRootBoneIndex if request is not None else 0,
		)
	except Exception:
		write_text(output_root / "ue_preview_error.log", traceback.format_exc())
		return ""


def build_job_status_response(record: JobRecord) -> JobStatusResponse:
	return JobStatusResponse(
		jobId=record.job_id,
		status=record.status,
		command=record.command,
		error=record.error,
		outputDir=record.output_dir,
		stdOutLog=record.stdout_log,
		stdErrLog=record.stderr_log,
		createdAt=record.created_at,
		startedAt=record.started_at,
		finishedAt=record.finished_at,
		generatedFiles=record.generated_files,
	)


def finish_cancelled_job(record: JobRecord, message: str = "Generation cancelled by user.") -> None:
	record.status = "cancelled"
	record.error = message
	record.finished_at = now_utc()
	if record.request.sessionId:
		session = SESSIONS.get(record.request.sessionId)
		if session is not None and session.active_job_id == record.job_id:
			session.active_job_id = ""
			session.updated_at = now_utc()


def terminate_generation_process(record: JobRecord) -> None:
	process = record.process
	if process is None or process.poll() is not None:
		return
	process.terminate()
	try:
		process.wait(timeout=10.0)
	except subprocess.TimeoutExpired:
		process.kill()
		process.wait(timeout=5.0)


def run_generation_job(job_id: str) -> None:
	record = JOBS[job_id]
	if record.cancel_requested:
		finish_cancelled_job(record)
		return
	record.status = "running"
	record.started_at = now_utc()
	if record.request.sessionId:
		session = SESSIONS.get(record.request.sessionId)
		if session is not None:
			session.updated_at = now_utc()
			session.active_job_id = job_id
			if record.request.model:
				session.model = record.request.model
	job_root, response_output_dir = resolve_output_paths(record.request.outputDir, record.request.outputName, job_id)
	if job_root.exists():
		shutil.rmtree(job_root)
	job_root.mkdir(parents=True, exist_ok=True)
	output_stem = job_root / record.request.outputName
	stdout_log = job_root / "stdout.log"
	stderr_log = job_root / "stderr.log"
	record.output_dir = response_output_dir
	record.stdout_log = append_display_path(response_output_dir, "stdout.log")
	record.stderr_log = append_display_path(response_output_dir, "stderr.log")
	record.command = "Preparing Kimodo generation inputs"
	try:
		input_dir = build_input_folder(job_root, record.request, should_cancel=lambda: record.cancel_requested)
	except Exception as error:
		write_text(stderr_log, traceback.format_exc())
		record.status = "failed"
		record.error = f"{type(error).__name__}: {error}"
		record.finished_at = now_utc()
		if record.request.sessionId:
			session = SESSIONS.get(record.request.sessionId)
			if session is not None and session.active_job_id == job_id:
				session.active_job_id = ""
				session.updated_at = now_utc()
		return
	if record.cancel_requested:
		write_text(stderr_log, "Generation cancelled while preparing Kimodo generation inputs.\n")
		finish_cancelled_job(record)
		return
	environment = build_environment(record.request, job_root)
	text_encoder_ready, text_encoder_error = wait_for_text_encoder(environment)
	if record.cancel_requested:
		write_text(stderr_log, "Generation cancelled before Kimodo inference started.\n")
		finish_cancelled_job(record)
		return
	if not text_encoder_ready:
		write_text(stderr_log, text_encoder_error + "\n")
		record.status = "failed"
		record.error = text_encoder_error
		record.finished_at = now_utc()
		if record.request.sessionId:
			session = SESSIONS.get(record.request.sessionId)
			if session is not None and session.active_job_id == job_id:
				session.active_job_id = ""
				session.updated_at = now_utc()
		return
	command = [
		sys.executable,
		"-m",
		"kimodo.scripts.generate",
		"--model",
		record.request.model,
		"--input_folder",
		str(input_dir),
		"--output",
		str(output_stem),
		"--num_transition_frames",
		str(record.request.numTransitionFrames),
	]
	if not is_postprocess_enabled(record.request):
		command.append("--no-postprocess")
	else:
		if kimodo_generate_supports_root_margin():
			command.extend(["--root_margin", str(record.request.postprocessRootMargin)])
		if kimodo_generate_supports_above_ground_offset():
			command.extend(["--above_ground_offset", str(record.request.postprocessAboveGroundOffset)])
	if record.request.bSaveExampleDir:
		command.append("--save_example_dir")
	if record.request.cfgType:
		command.extend(["--cfg_type", record.request.cfgType])
		if record.request.cfgWeight:
			command.append("--cfg_weight")
			command.extend([str(value) for value in record.request.cfgWeight])
	record.command = subprocess.list2cmdline(command)
	try:
		with stdout_log.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_log.open(
			"w",
			encoding="utf-8",
			errors="replace",
		) as stderr_file:
			process = subprocess.Popen(
				command,
				stdout=stdout_file,
				stderr=stderr_file,
				text=True,
				env=environment,
			)
			record.process = process
			while process.poll() is None:
				if record.cancel_requested:
					record.status = "cancelling"
					terminate_generation_process(record)
					break
				time.sleep(0.25)
			process.wait()
		stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.exists() else ""
		stderr_text = stderr_log.read_text(encoding="utf-8", errors="replace") if stderr_log.exists() else ""
		if record.cancel_requested:
			finish_cancelled_job(record)
		elif process.returncode != 0:
			record.status = "failed"
			record.error = (stderr_text or "").strip() or (stdout_text or "").strip() or f"Kimodo exited with code {process.returncode}."
		else:
			record.status = "completed"
			preserve_raw_smooth_root_outputs(job_root)
			stabilize_motion_outputs(job_root, record.request)
			try_write_preview_manifest_for_output_root(job_root, record.request)
			record.generated_files = discover_outputs(job_root, response_output_dir)
	except Exception as error:
		write_text(stderr_log, traceback.format_exc())
		record.status = "failed"
		record.error = f"{type(error).__name__}: {error}"
	finally:
		record.process = None
		if not record.finished_at:
			record.finished_at = now_utc()
		if record.request.sessionId:
			session = SESSIONS.get(record.request.sessionId)
			if session is not None and session.active_job_id == job_id:
				session.active_job_id = ""
				session.updated_at = now_utc()


@APP.get("/api/v1/health", response_model=HealthResponse)
@APP.get("/bridge/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
	active_jobs = sum(1 for record in JOBS.values() if record.status in {"queued", "running", "cancelling"})
	text_encoder_ready, text_encoder_detail = current_text_encoder_readiness()
	if not text_encoder_ready:
		raise HTTPException(
			status_code=503,
			detail=f"Docker bridge is up, but text encoder is {text_encoder_detail}. Wait for text-encoder:9550 to finish loading, then Refresh Health.",
		)
	return HealthResponse(
		status="ok",
		serviceVersion=SERVICE_VERSION,
		outputRoot=str(BRIDGE_OUTPUT_ROOT or DEFAULT_OUTPUT_ROOT),
		vendorRoot=str(VENDOR_ROOT),
		bMotionCorrectionInstalled=detect_motion_correction(),
		activeJobs=active_jobs,
		activeSessions=len(SESSIONS),
	)


@APP.post("/api/v1/sessions", response_model=SessionResponse)
@APP.post("/bridge/v1/sessions", response_model=SessionResponse)
def create_session(request: SessionCreateRequest) -> SessionResponse:
	session_id = str(uuid.uuid4())
	record = SessionRecord(session_id=session_id, model=request.model or "")
	SESSIONS[session_id] = record
	return build_session_response(record)


@APP.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
@APP.get("/bridge/v1/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
	record = SESSIONS.get(session_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
	record.updated_at = now_utc()
	return build_session_response(record)


@APP.put("/api/v1/sessions/{session_id}/model", response_model=SessionResponse)
@APP.put("/bridge/v1/sessions/{session_id}/model", response_model=SessionResponse)
def update_session_model(session_id: str, request: SessionModelUpdateRequest) -> SessionResponse:
	record = SESSIONS.get(session_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
	model_key = request.model.strip()
	if not model_key:
		raise HTTPException(status_code=422, detail="A model short key is required.")
	if not has_model_descriptor(model_key):
		checkpoint_root = get_checkpoint_root()
		if checkpoint_root is None:
			raise HTTPException(status_code=404, detail=f"Unknown model short key: {model_key}")
		raise HTTPException(
			status_code=404,
			detail=f"Model '{model_key}' is not available under CHECKPOINT_DIR {checkpoint_root}. Only locally mounted models can be used in Docker bridge mode.",
		)
	record.model = model_key
	record.updated_at = now_utc()
	return build_session_response(record)


@APP.put("/api/v1/sessions/{session_id}/authoring", response_model=SessionResponse)
@APP.put("/bridge/v1/sessions/{session_id}/authoring", response_model=SessionResponse)
def update_session_authoring_state(session_id: str, request: SessionAuthoringStateRequest) -> SessionResponse:
	record = SESSIONS.get(session_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
	record.prompts = list(request.prompts)
	record.constraints = list(request.constraints)
	record.constraints_path = request.constraintsPath.strip()
	record.frames_per_second = max(request.framesPerSecond, 1.0)
	record.updated_at = now_utc()
	return build_session_response(record)


@APP.get("/api/v1/sessions/{session_id}/examples", response_model=list[ExampleDescriptor])
@APP.get("/bridge/v1/sessions/{session_id}/examples", response_model=list[ExampleDescriptor])
def get_session_examples(session_id: str, model: str = "") -> list[ExampleDescriptor]:
	record = SESSIONS.get(session_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
	model_name = model.strip() or record.model.strip()
	if not model_name:
		return []
	return list_examples_for_model(model_name)


@APP.post("/api/v1/sessions/{session_id}/examples/load", response_model=ExampleLoadResponse)
@APP.post("/bridge/v1/sessions/{session_id}/examples/load", response_model=ExampleLoadResponse)
def load_session_example(session_id: str, request: ExampleLoadRequest) -> ExampleLoadResponse:
	record = SESSIONS.get(session_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
	model_name = request.model.strip() or record.model.strip()
	if not model_name:
		raise HTTPException(status_code=422, detail="A model short key is required before loading examples.")
	if not has_model_descriptor(model_name):
		checkpoint_root = get_checkpoint_root()
		if checkpoint_root is None:
			raise HTTPException(status_code=404, detail=f"Unknown model short key: {model_name}")
		raise HTTPException(
			status_code=404,
			detail=f"Model '{model_name}' is not available under CHECKPOINT_DIR {checkpoint_root}. Only locally mounted models can be used in Docker bridge mode.",
		)
	example_name = request.exampleName.strip()
	if not example_name:
		raise HTTPException(status_code=422, detail="An example name is required.")
	example_dir = get_model_examples_dir(model_name) / example_name
	meta_path = example_dir / "meta.json"
	if not meta_path.exists():
		raise HTTPException(status_code=404, detail=f"Example meta.json not found for {example_name}")
	constraints_path = example_dir / "constraints.json"
	preview_manifest_path = ""
	motion_path = example_dir / "motion.npz"
	if motion_path.exists():
		try:
			preview_manifest_path = write_preview_manifest(example_dir / UE_PREVIEW_MANIFEST_NAME, [motion_path])
		except Exception as exc:
			sys.stderr.write(f"[kimodo_ue_service] Failed to build preview manifest for example '{example_name}': {exc}\n")
	loaded_prompts = load_prompts_from_meta(meta_path, max(request.framesPerSecond, 1.0))
	record.model = model_name
	record.prompts = list(loaded_prompts)
	record.constraints = []
	record.constraints_path = str(constraints_path) if constraints_path.exists() else ""
	record.frames_per_second = max(request.framesPerSecond, 1.0)
	record.updated_at = now_utc()
	return ExampleLoadResponse(
		exampleName=example_name,
		model=model_name,
		prompts=loaded_prompts,
		constraintsPath=record.constraints_path,
		previewManifestPath=preview_manifest_path,
	)


@APP.delete("/api/v1/sessions/{session_id}", response_model=SessionResponse)
@APP.delete("/bridge/v1/sessions/{session_id}", response_model=SessionResponse)
def delete_session(session_id: str) -> SessionResponse:
	record = SESSIONS.pop(session_id, None)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
	record.status = "deleted"
	record.updated_at = now_utc()
	record.active_job_id = ""
	return build_session_response(record)


@APP.get("/api/v1/models", response_model=list[ModelDescriptor])
@APP.get("/bridge/v1/models", response_model=list[ModelDescriptor])
def models() -> list[ModelDescriptor]:
	return build_models()


@APP.post("/api/v1/jobs", response_model=GenerateJobAccepted)
@APP.post("/bridge/v1/jobs", response_model=GenerateJobAccepted)
def create_job(request: GenerateJobRequest) -> GenerateJobAccepted:
	if request.sessionId:
		session = SESSIONS.get(request.sessionId)
		if session is None:
			raise HTTPException(status_code=404, detail=f"Unknown session: {request.sessionId}")
		request = build_generation_request_from_session(session, request)
		session.updated_at = now_utc()
		if request.model:
			session.model = request.model

	model_name = request.model.strip()
	if not model_name:
		raise HTTPException(status_code=422, detail="A model short key is required before generating.")
	if not has_model_descriptor(model_name):
		checkpoint_root = get_checkpoint_root()
		if checkpoint_root is None:
			raise HTTPException(status_code=404, detail=f"Unknown model short key: {model_name}")
		raise HTTPException(
			status_code=404,
			detail=f"Model '{model_name}' is not available under CHECKPOINT_DIR {checkpoint_root}. Mounted checkpoints: {', '.join(sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())) or 'none' }.",
		)

	job_id = str(uuid.uuid4())
	record = JobRecord(job_id=job_id, request=request)
	JOBS[job_id] = record
	EXECUTOR.submit(run_generation_job, job_id)
	return GenerateJobAccepted(jobId=job_id, status=record.status)


@APP.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
@APP.get("/bridge/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
	record = JOBS.get(job_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
	return build_job_status_response(record)


@APP.delete("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
@APP.delete("/bridge/v1/jobs/{job_id}", response_model=JobStatusResponse)
def cancel_job(job_id: str) -> JobStatusResponse:
	record = JOBS.get(job_id)
	if record is None:
		raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
	if record.status in {"completed", "failed", "cancelled"}:
		return build_job_status_response(record)
	record.cancel_requested = True
	record.status = "cancelling"
	terminate_generation_process(record)
	return build_job_status_response(record)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the EmbodiedSpaceStudio Kimodo bridge service.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=50127)
	parser.add_argument("--checkpoint-root", default="")
	parser.add_argument("--text-encoders-root", default="")
	return parser.parse_args()


def main() -> int:
	global CHECKPOINT_ROOT
	global TEXT_ENCODERS_ROOT
	args = parse_args()
	DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	if BRIDGE_OUTPUT_ROOT is not None:
		BRIDGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
	CHECKPOINT_ROOT = args.checkpoint_root
	TEXT_ENCODERS_ROOT = args.text_encoders_root
	if args.checkpoint_root:
		os.environ["CHECKPOINT_DIR"] = args.checkpoint_root
	if args.text_encoders_root:
		os.environ["TEXT_ENCODERS_DIR"] = args.text_encoders_root
	uvicorn.run(APP, host=args.host, port=args.port, log_level="info")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
