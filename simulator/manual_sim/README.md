# Manual Offline Eval

`simulator/manual_sim/offline_eval.py` replays a real Waste-E recording session, samples one camera stream, and scores a policy against the robot's actual future trajectory.

This is the first practical step before a full model-backed simulator:
- it works with the current `hardware/recordings/<session>/` format
- it gives you ADE/FDE metrics now
- it leaves a clean hook for a future Runpod policy backend

## Run With `uv`

From the repo root, the easiest local or Runpod flow is:

```bash
bash simulator/manual_sim/setup_runpod.sh
```

Then run:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy constant_velocity \
  --overwrite
```

If you prefer raw `uv` commands:

```bash
uv sync --project simulator/manual_sim
uv run --project simulator/manual_sim \
  simulator/manual_sim/offline_eval.py \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy constant_velocity \
  --overwrite
```

`uv.lock` is optional here. You do not need it to run. It is only useful if you want stricter reproducibility across different pods.

## Inputs

Expected session layout:

```text
hardware/recordings/<session>/
  telemetry.csv
  session.json
  gps_track.geojson
  videos/
    orangepi-video0.mp4
    ...
```

The default primary view is `orangepi-video0.mp4`, since that is the cleanest-aligned stream in your current recordings.

## Quick Start

Run a cheap baseline:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy constant_velocity \
  --overwrite
```

Sanity-check the pipeline with an oracle policy:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy oracle \
  --overwrite
```

Render an overlay video:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy constant_velocity \
  --render-video \
  --overwrite
```

Quick smoke test on a subset:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy constant_velocity \
  --max-samples 40 \
  --render-video \
  --overwrite
```

## Outputs

By default, outputs go to:

```text
hardware/recordings/<session>/offline_eval/<policy>/
```

Artifacts:
- `summary.json`: run config, aggregate metrics, artifact paths
- `per_frame_metrics.csv`: one row per evaluated timestep
- `predictions.jsonl`: predicted and ground-truth future waypoints
- `overlay.mp4`: optional annotated video

## Built-In Policies

- `stop`: predicts zero motion
- `constant_velocity`: projects current speed and yaw-rate forward
- `oracle`: uses the robot's actual future path, useful to validate the evaluator itself
- `external`: calls your own policy hook

## External Policy Hook

For a future Runpod model backend, use:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy external \
  --policy-hook simulator/manual_sim/sample_policy.py:predict \
  --overwrite
```

The hook signature is:

```python
def predict(observation, config):
    ...
    return [[x1, y1], [x2, y2], ...]
```

Where:
- `observation.sample` contains the current telemetry sample
- `observation.history_local_xy` is the recent ego-frame path history
- `observation.frame_rgb` is the sampled RGB frame if you pass `--load-frames`
- `config.steps` and `config.step_seconds` define the output trajectory shape

The returned waypoints are expected in ego frame:
- `x`: meters forward
- `y`: meters left

If your model needs camera frames:

```bash
bash simulator/manual_sim/run_offline_eval.sh \
  hardware/recordings/2026-05-07_08-11-52 \
  --policy external \
  --policy-hook path/to/model_adapter.py:predict \
  --load-frames \
  --overwrite
```

## Runpod Notes

This script is meant to move cleanly to Runpod:
- the evaluator itself runs fine on CPU
- only your future policy hook needs GPU
- you can start with the exact same session folders you already collect locally

For remote or containerized model serving later, the cleanest path is:
1. keep `offline_eval.py` unchanged
2. implement a policy module that calls your local model or a Runpod endpoint
3. run `--policy external --policy-hook ...`

## Dependencies

Required:
- `python`

Optional:
- `opencv-python` and `numpy` for `--render-video` or `--load-frames`
- `ffprobe` for reliable video metadata probing

If you use the provided `uv` setup, those Python dependencies are installed for you.
