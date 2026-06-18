<h1 align="left">
    <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="30" height="auto" />
    Viser with Timeline for Motion Synthesis
    <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="30" height="auto" />
</h1>


## Disclaimer

This repository is a fork of the upstream Viser project ([https://github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser)) at commit `567580e0af1ef850b506dc85295e82c53bf55557` and is not intended to replace it. It contains modifications to integrate a timeline UI and keyboard input support specifically for use alongside the NVIDIA Kimodo repository ([https://github.com/nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo)). This fork may not be maintained long‑term and is not intended to be merged back into the main Viser library. Please use the upstream Viser project instead unless you specifically need these fork‑specific changes and accept the maintenance risk.


## Introduction

Viser is a 3D visualization library for computer vision and robotics in Python.

Features include:

- API for visualizing 3D primitives.
- GUI building blocks: buttons, checkboxes, text inputs, sliders, etc.
- Scene interaction tools (clicks, selection, transform gizmos).
- Programmatic camera control and rendering.
- An entirely web-based client, for easy use over SSH!

The goal is to provide primitives that are (1) easy for simple visualization tasks, but (2) can be composed into more elaborate interfaces. For more about design goals, see the [technical report](https://arxiv.org/abs/2507.22885).

Examples and documentation: https://viser.studio

## Installation

You can install `viser` with `pip`:

```bash
pip install viser            # Core dependencies only.
pip install viser[examples]  # To include example dependencies.
```

That's it! To learn more, we recommend looking at the examples in the [documentation](https://viser.studio/).

## Citation

To cite Viser in your work, you can use the BibTeX for our [technical report](https://arxiv.org/abs/2507.22885):

```
@misc{yi2025viser,
      title={Viser: Imperative, Web-based 3D Visualization in Python},
      author={Brent Yi and Chung Min Kim and Justin Kerr and Gina Wu and Rebecca Feng and Anthony Zhang and Jonas Kulhanek and Hongsuk Choi and Yi Ma and Matthew Tancik and Angjoo Kanazawa},
      year={2025},
      eprint={2507.22885},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.22885},
}
```

## Acknowledgements

`viser` is heavily inspired by packages like
[Pangolin](https://github.com/stevenlovegrove/Pangolin),
[Dear ImGui](https://github.com/ocornut/imgui),
[rviz](https://wiki.ros.org/rviz/),
[meshcat](https://github.com/rdeits/meshcat), and
[Gradio](https://github.com/gradio-app/gradio).

The web client is implemented using [React](https://react.dev/), with:

- [Vite](https://vitejs.dev/) / [Rollup](https://rollupjs.org/) for bundling
- [three.js](https://threejs.org/) via [react-three-fiber](https://github.com/pmndrs/react-three-fiber) and [drei](https://github.com/pmndrs/drei)
- [Mantine](https://mantine.dev/) for UI components
- [zustand](https://github.com/pmndrs/zustand) for state management
- [vanilla-extract](https://vanilla-extract.style/) for stylesheets

Thanks to the authors of these projects for open-sourcing their work!
