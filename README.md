# Residual Fusion SITL

This repository contains a SITL-based residual-fusion controller for evaluating how different PID/ML recovery strategies behave under simulated GPS and IMU attacks. It keeps the simulator source, the runtime model JSON files, and the experiment scripts needed to reproduce mode comparisons.

Generated training tables, batch experiment outputs, simulator logs, dependency checkouts, papers, and local build artifacts are intentionally ignored.

## Repository Layout

- `simulator/` contains the ArduPilot SITL source tree and the runtime controller integration.
- `simulator/libraries/` contains the runtime fusion code and bundled model JSON files.
- `simulator/Tools/autotest/run_mode_experiments.py` is the public batch experiment entry point.
- `simulator/Tools/autotest/requirements.txt` lists the Python package needed by the batch runner.
- The top-level training utilities are optional and are used only when regenerating gate models from local experiment data.

## Runtime Modes

The batch runner compares these controller modes:

- `0`: pure PID baseline.
- `1`: ML output only.
- `2`: hard switch between PID and ML using a residual detector.
- `3`: one-frame gate model that blends PID and ML.
- `4`: 100-frame LSTM residual-fusion gate that blends PID and ML.

The default copter parameter file starts in mode `4`.

## Supported Environment

Use Linux or WSL. Ubuntu 18.04 is the tested environment for this older ArduPilot SITL tree. Newer Ubuntu versions can work, but they may need manual Python 2.7 package handling because this simulator generation still launches `sim_vehicle.py` with Python 2.7.

The C++ ML dependencies must stay on the older versions used by this codebase:

- FunctionalPlus `v0.2.14-p0` at `aa94989c43bd6680479b3c1cb5f63972d8380d61`
- Eigen `3.3.9` at `0fd6b4f71dd85b2009ee4d1aeb296e2c11fc9d68`
- nlohmann/json `v3.7.3` at `e7b3b40b5a95bc74b9a7f662830a27c49ffc01b4`
- frugally-deep `v0.15.17-p0` at `9159a4bdf25df607c3af188884fed2480142eea2`

`setup.sh` installs those exact revisions into `/usr/local`.

## Install

Clone the repository, then run setup from the repository root:

```bash
git clone <your-fork-url> residual-fusion-sitl
cd residual-fusion-sitl
chmod +x setup.sh
./setup.sh
```

The script installs SITL build packages, Python packages for `sim_vehicle.py` and the experiment runner, then clones and installs the pinned C++ dependencies. The dependency checkout directories are local working directories and are ignored by Git.

Useful setup options:

```bash
SKIP_APT=1 ./setup.sh
SKIP_PYTHON=1 ./setup.sh
FORCE_DEP_CHECKOUT=1 ./setup.sh
```

Use `SKIP_APT` or `SKIP_PYTHON` when the machine already has those prerequisites. Use `FORCE_DEP_CHECKOUT` only when you want setup to reset local dependency checkouts to the pinned revisions.

## Build The Simulator

From the repository root:

```bash
cd simulator
./waf configure --board sitl
./waf copter
```

If Python 2.7 user scripts installed by `pip2 --user` are not on your path, add them before launching SITL:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Run The Simulator Manually

Start SITL from the copter directory:

```bash
cd simulator/ArduCopter
python2.7 ../Tools/autotest/sim_vehicle.py --wipe-eeprom --console --map
```

In the MAVProxy console, load and fly a mission:

```text
wp load ../Tools/autotest/mission/mission-1.txt
mode guided
arm throttle
takeoff 50
mode auto
```

Use `mission-2.txt` for the second bundled route. Runtime logs and generated CSV files are intentionally not tracked.

## Run Batch Mode Experiments

Run experiments from the repository root. The script launches SITL, sets controller and attack parameters, flies the selected missions, records telemetry, and writes aggregate summaries.

Smoke test:

```bash
python3 simulator/Tools/autotest/run_mode_experiments.py \
  --modes 0,2,4 \
  --missions mission-1.txt \
  --attack-combos 0:0,4:0 \
  --seeds 0 \
  --max-runs 6 \
  --force-cleanup \
  --isolated-sitl-state
```

Default comparison:

```bash
python3 simulator/Tools/autotest/run_mode_experiments.py
```

Stress comparison:

```bash
python3 simulator/Tools/autotest/run_mode_experiments.py \
  --attack-grid stress \
  --modes 0,2,4 \
  --seeds 0,1,2 \
  --resume \
  --force-cleanup \
  --isolated-sitl-state
```

Dry-run a plan without launching SITL:

```bash
python3 simulator/Tools/autotest/run_mode_experiments.py \
  --modes 0,2,4 \
  --missions mission-1.txt,mission-2.txt \
  --attack-grid layered \
  --dry-run
```

Analyze an existing experiment directory:

```bash
python3 simulator/Tools/autotest/run_mode_experiments.py \
  --analyze-only simulator/Tools/autotest/mode_experiment_runs/<run-directory>
```

Outputs are written under `simulator/Tools/autotest/mode_experiment_runs/` unless `--output-dir` is provided. Key files include `experiment_manifest.csv`, per-run `telemetry.csv`, per-run `analysis.json`, `summary_by_condition.csv`, `comparison_by_mode.csv`, and `analysis_report.md`.

Use `--isolated-sitl-state` when reusing an old workspace so stale simulator parameters do not affect arming or attack settings.

## Optional Model Training

The training utilities are optional and are meant for regenerating gate models from local experiment data. They are separate from running the simulator with the checked-in runtime JSON models.

Training requires Python 3 packages such as `numpy`, `pandas`, and `tensorflow`. Generated training tables, intermediate analysis folders, and exported experiment artifacts are ignored by Git unless explicitly added.
