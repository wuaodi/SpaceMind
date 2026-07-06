# Testing Guide

End-to-end instructions for running the SpaceMind agent stack with the bundled Three.js environment. No Unreal Engine assets are required; everything below runs out of the box on a normal PC.

## 1. Prerequisites

- Python 3.10+ (tested with Anaconda)
- Node.js 18+ (for the Three.js environment; `npm` must be on PATH)
- A local Redis server listening on `127.0.0.1:6379`
- A vision-language model API key (OpenAI-compatible endpoint)

## 2. Setup

```bash
git clone https://github.com/wuaodi/SpaceMind.git
cd SpaceMind
pip install -r SpaceMind_UE5/requirements.txt

# Configure model credentials
cp .env.example SpaceMind_UE5/.env
# Edit SpaceMind_UE5/.env and fill in your API key / base URL
```

Start Redis before anything else. On Windows, run `redis-server.exe`; on Linux/macOS, `redis-server`.

## 3. Start the environment

```bash
cd SpaceMind_UE5/environments/threejs_env
python launch_env.py
```

The first launch installs npm dependencies and builds the page, which takes a minute. A browser tab opens at `http://127.0.0.1:4173` showing the space scene.

Important: keep the browser tab open. The page is the renderer; if it is closed, the agent will not receive images.

Common options:

```bash
python launch_env.py --skip-build              # skip the frontend rebuild on later runs
python launch_env.py --satellite IBEX          # target: BioSentinel/CAPSTONE/Huygens/IBEX/New_Horizons
python launch_env.py --init_x -11 --init_y 0 --init_z 0   # servicer initial position (m)
```

## 4. Verify the link

With the environment running and the browser tab open, from `threejs_env/`:

```bash
python self_test.py --scenario smoke      # pose + image round trip
python self_test.py --scenario move       # motion command executes
python self_test.py --scenario exposure   # exposure control updates the image
```

Each test prints `... ok` on success. If `smoke` fails, see Troubleshooting below.

## 5. Run the agent

From `SpaceMind_UE5/`:

```bash
python host.py --task rendezvous-hold-front --tool_profile oracle_full
```

Expected behavior: the agent receives images, approaches the target in shrinking steps (about 7 to 10 steps from the default 11 m start), calls `terminate_navigation` near the 2 m hold distance, and the evaluator prints `Mission success: True` together with a delta-v summary.

Other tasks and profiles:

```bash
python host.py --task rendezvous-hold-above --tool_profile oracle_full
python host.py --task rendezvous-hold-front --tool_profile hybrid_nav    # perception from images + LiDAR
python host.py --task rendezvous-hold-front --enable_react               # ReAct reasoning mode
```

Per-run logs are written to `runtime_logs/log/`, and model traces to `runtime_logs/trace/` (open `runtime_logs/viewer/trace_viewer.html` to inspect a trace).

## 6. Challenge scenarios (stress injection)

Restart the environment with an injection switch, then run the agent as usual:

```bash
# E1 rotating target
python launch_env.py --skip-build --spin_deg_s 0.5

# E2 delta-v budget (enforced on the agent side)
python host.py --task rendezvous-hold-front --tool_profile oracle_full --deltav_budget 12

# E3N actuation noise
python launch_env.py --skip-build --noise

# E3F thruster fault (dy commands execute at 50%)
python launch_env.py --skip-build --fault_axis dy --fault_scale 0.5

# E4L intermittent LiDAR dropout
python launch_env.py --skip-build --lidar_dropout 0.3

# E4E mid-flight exposure disturbance (after the 3rd motion command)
python launch_env.py --skip-build --exposure_disturb_step 3 --exposure_disturb_value -3
```

## 7. Parse results

Continuous metrics (delta-v proxy, path efficiency, pointing latency) and the failure taxonomy are computed from run logs:

```bash
cd SpaceMind_UE5
python runtime_logs/result_parser.py                  # parses all runs in runtime_logs/log/
python runtime_logs/result_parser.py --output results.csv
```

Reference per-run results from the paper's challenge campaign are provided in `SpaceMind_UE5/results/phase_e_results.jsonl`.

## 8. Troubleshooting

- `self_test.py` fails on `smoke`: confirm the browser tab at `http://127.0.0.1:4173` is open and `logs/launcher_bridge_stdout.log` contains `Browser connected`; confirm Redis answers on `127.0.0.1:6379`.
- `host.py` hangs at `Waiting for image...`: the environment or the browser tab was closed; restart `launch_env.py` and reload the page.
- `npm install` fails behind a proxy: run `npm config set registry https://registry.npmmirror.com` and retry.
- Model call errors: check the API key and base URL in `SpaceMind_UE5/.env`.
- Port conflicts: the environment uses 4173 (HTTP), 8765 (WebSocket), 6379 (Redis); pass `--http-port` / `--ws-port` to change the first two.

## Notes

- The Three.js environment mirrors the UE5/AirSim Redis interface exactly (images, part-level segmentation, LiDAR, pose truth, exposure, stress injection), so `host.py`, tools, and skills run unmodified. Rendering and geometry are simplified, so the numbers in the paper, which were obtained in UE5, are not expected to match exactly.
- Satellite models are from [NASA 3D Resources](https://science.nasa.gov/3d-resources/) (public domain), scaled to the physical dimensions used in the paper experiments.
