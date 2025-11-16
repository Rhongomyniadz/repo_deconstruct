# PHALP Documentation

This folder contains a deconstruction of the PHALP repository
(*Tracking People by Predicting 3D Appearance, Location & Pose*).

The goal is to help new contributors quickly understand:

- The **architecture** of the system
- The **data flow** from input video to 3D tracking output
- The roles of **core modules** like `PHALP`, `HMAR`, the tracker,
  and the visualizer
- How to **extend** PHALP with new models or features

## Document Map

- `architecture_overview.md`  
  High-level system design and component relationships.

- `module_breakdown.md`  
  Directory-level and file-level responsibilities.

- `data_flow.md`  
  How data moves from raw video frames through detection, 3D
  estimation, tracking, and visualization.

- `tracking_system.md`  
  Detailed description of the multi-target tracking logic.

- `hmar.md`  
  Human Mesh & Appearance Regression (HMAR) model and its outputs.

- `visualization_pipeline.md`  
  Rendering and visualization of 3D humans and tracklets.

- `execution_flow.md`  
  “What happens when I run the demo?” – step-by-step execution path.

- `extension_points.md`  
  Where and how to plug in new detectors, trackers, models, or
  visualization modes.

- `reflection_template.md`  
  Template for recording lessons learned while deconstructing PHALP.

You can read these in order (top–down) or jump directly to the part
you care about (e.g., tracking, HMAR, or rendering).