---
name: perception-strategy
description: Guide multimodal observation for spacecraft situational awareness. Use for perception-heavy tasks that combine image quality, segmentation, local inspection, and knowledge-base reasoning.
metadata:
  skill_kind: helper
  routing_summary: Add when the mission depends on structured visual evidence gathering and multimodal inspection.
  routing_keywords:
    - perception
    - observe
    - image analysis
    - multimodal evidence
    - visual summary
    - inspect components
    - component status
    - components status
---
# Perception Workflow

1. Start from the currently exposed observation source and the full image.
2. Evaluate image quality with `image_bright()` if visibility is questionable, and use `set_exposure()` when brightness itself is the bottleneck.
3. Inspect visible structure with `part_segmentation()` when segmentation is exposed.
4. Use `image_zoom()` when the target is centered but important details are too small to judge from the full frame.
5. Use `image_crop()` when you need a local view of a specific region such as an antenna, docking port, solar panel root, or damaged area.
6. If the `[Sensor] LiDAR` block shows data and geometry is still ambiguous, use it as an additional structural cue.
7. Query `knowledge_base()` after enough visual evidence exists to connect observed structure to type and function.

# Viewpoint Control

Use `set_attitude()` when a different angle is needed to disambiguate spacecraft status or components.

Use `image_zoom()` or `image_crop()` before changing viewpoint when the needed evidence is already visible but too small in the current image.

# Efficiency Rule

Do not loop on perception tools after the answer is already well-supported. Prefer the currently exposed visual tools first, and terminate once the evidence is sufficient.
