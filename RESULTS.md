# Released model results (official BOP19 protocol)

Scored locally with the official bop_toolkit (`eval_bop19_pose.py`,
`test_targets_bop19.json`). Submission CSVs are in `evaluation/`.

| Dataset | Checkpoint | Box protocol | Rows | Pose ms/img | CSV time s/img | AR | MSPD | MSSD | VSD |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| T-LESS | `checkpoints/rvc6d_tless_e39.pth.tar` | GDRNPP detections | 6422 | 5.18 | 0.088 | **75.27** | 80.51 | 78.90 | 66.40 |
| LM-O | `checkpoints/rvc6d_lmo_e36.pth.tar` | GDRNPP (YOLOX) detections | 1445 | 5.74 | 0.087 | **69.55** | 79.52 | 74.93 | 54.22 |

- Pose time = H2D + forward + fused decode + D2H + metric translation
  restore, one batch per image, RTX 5090, FP32 (TF32 off), cuDNN enabled.
- CSV time adds the detector time carried by the detection JSON
  (BOP detector+pose convention).
- Reproduce with `tools/evaluate_bop_detections.py` (see README).
