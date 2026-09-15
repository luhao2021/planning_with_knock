# Offline regression fixtures

`recorded_protex_poses.json` and `recorded_protex_layout.json` are the compact pose and planning records from the four-Protex capture `protex_4_20260914_183700`. Tests read these copies so they do not require the local experiments archive or robot access.

`initial_pick_mismatch_061015.json` preserves the initial geometry that exposed the planner/executor grasp-clearance mismatch. `automatic_seed_bounded_correction.json` preserves the automatic-seed preview geometry that requires the 5 mm correction limit to be enforced inside optimization.

Images, captures, learned-model assets, videos and runtime logs remain outside version control.
