#!/usr/bin/env python3
"""
Run PID-Piper mode comparison experiments in SITL.

This script is intentionally independent from the older 360 experiment scripts.
It compares PID-Piper modes over missions, attack scenarios, and seeds, then
summarizes task success and trajectory/control quality metrics.
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

try:
    from pymavlink import mavutil
except Exception:
    mavutil = None


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SIMULATOR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMULATOR_DIR, ".."))
ARDUCOPTER_DIR = os.path.join(SIMULATOR_DIR, "ArduCopter")
MISSION_DIR = os.path.join(SCRIPT_DIR, "mission")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "mode_experiment_runs")
DEFAULT_MAVLINK_ADDR = "udp:127.0.0.1:14550"
DEFAULT_SUCCESS_RADIUS_M = 10.0
DEFAULT_LAND_ALT_M = 3.0
OUTSIDE_SUCCESS_GATE_SUFFIX = "_outside_success_gate"
MODE_LABELS = {
    0: "pid_baseline",
    1: "original_ml",
    2: "hard_switch",
    3: "one_frame_gate",
    4: "lstm_residual_fusion",
}
COMPARISON_BASELINES = [0, 2]
COMPARISON_TARGET = 4
RETRYABLE_LAUNCH_STATUSES = set(["exception", "prearm_timeout", "arm_timeout", "takeoff_timeout"])

MAVLINK_TYPES = [
    "GPS_RAW_INT",
    "GLOBAL_POSITION_INT",
    "ATTITUDE",
    "HEARTBEAT",
    "EXTENDED_SYS_STATE",
    "VFR_HUD",
    "MISSION_CURRENT",
    "NAV_CONTROLLER_OUTPUT",
    "SYS_STATUS",
    "STATUSTEXT",
]

TELEMETRY_COLUMNS = [
    "run_time_s",
    "wall_time",
    "mode",
    "armed",
    "landed_state",
    "lat",
    "lon",
    "relative_alt_m",
    "amsl_m",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "groundspeed_mps",
    "airspeed_mps",
    "heading_deg",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "rollspeed_rad_s",
    "pitchspeed_rad_s",
    "yawspeed_rad_s",
    "mission_seq",
    "nav_bearing",
    "target_bearing",
    "wp_dist_m",
    "alt_error_m",
    "aspd_error_mps",
    "xtrack_error_m",
    "battery_remaining_pct",
    "voltage_battery_v",
    "current_battery_a",
    "gps_fix_type",
    "gps_satellites_visible",
]

STRESS_POST_TAKEOFF_PARAMS = [
    ("FS_EKF_THRESH", 0.0),
    ("FS_CRASH_CHECK", 0.0),
]


def eprint(msg):
    print(msg, file=sys.stderr)


def progress(msg):
    try:
        print(msg)
    except IOError:
        pass


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_int_list(text):
    out = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(raw))
    return out


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def attack_combos_for_grid(name):
    name = name.lower()
    if name == "none":
        return [(0, 0)]
    if name == "base":
        return [(gps, imu) for gps in range(4) for imu in range(4)]
    if name == "layered":
        base = [(gps, imu) for gps in range(4) for imu in range(4)]
        extra = [
            (4, 0),
            (5, 0),
            (0, 4),
            (0, 5),
            (4, 2),
            (5, 2),
            (2, 4),
            (3, 5),
        ]
        return base + extra
    if name == "stress":
        return [
            (6, 0),
            (7, 0),
            (0, 6),
            (0, 7),
            (6, 2),
            (7, 2),
            (2, 6),
            (2, 7),
            (6, 6),
            (6, 7),
            (7, 6),
            (7, 7),
        ]
    if name == "extended":
        return [(gps, imu) for gps in range(8) for imu in range(8)]
    raise ValueError("unknown attack grid: %s" % name)


def parse_attack_combos(text):
    combos = []
    for item in parse_str_list(text):
        if ":" not in item:
            raise ValueError("attack combo must be GPS:IMU, got %r" % item)
        left, right = item.split(":", 1)
        combos.append((int(left), int(right)))
    return combos


def parse_param_assignments(text):
    params = []
    for item in parse_str_list(text):
        if "=" not in item:
            raise ValueError("parameter assignment must be NAME=VALUE, got %r" % item)
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("parameter assignment has empty name: %r" % item)
        params.append((name, float(raw_value.strip())))
    return params


def run_id_for(index, mission, mode, gps, imu, seed):
    stem = os.path.splitext(os.path.basename(mission))[0].replace("_", "-")
    return "%04d_%s_mode%d_gps%d_imu%d_seed%d" % (index, stem, mode, gps, imu, seed)


def condition_key(mission, mode, gps, imu, seed):
    return (str(mission), int(mode), int(gps), int(imu), int(seed))


def condition_key_from_meta(meta):
    return condition_key(meta["mission"], meta["mode"], meta["gps_scn"], meta["imu_scn"], meta["gps_seed"])


def condition_key_from_row(row):
    return condition_key(row["mission"], row["mode"], row["gps_scn"], row["imu_scn"], row["gps_seed"])


def run_dir_has_completed_result(run_dir):
    analysis_path = os.path.join(run_dir, "analysis.json")
    if os.path.isfile(analysis_path):
        payload = read_json_safely(analysis_path)
        if payload and payload.get("status") in RETRYABLE_LAUNCH_STATUSES:
            return False
        if payload and bool(payload.get("attempt_complete", False)) and bool(payload.get("status")):
            return True
    return find_latest_completed_attempt_dir(run_dir) is not None


def read_existing_run_meta(output_dir, completed_only=False):
    existing = {}
    max_index = 0
    for meta_path in sorted(glob.glob(os.path.join(output_dir, "*", "run_meta.json"))):
        run_dir = os.path.dirname(meta_path)
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            key = condition_key_from_meta(meta)
        except Exception:
            continue
        max_index = max(max_index, int(meta.get("run_index") or 0))
        if completed_only and not run_dir_has_completed_result(run_dir):
            continue
        existing[key] = {
            "run_index": int(meta.get("run_index") or 0),
            "run_id": meta.get("run_id") or os.path.basename(run_dir),
            "mission": meta.get("mission"),
            "mode": int(meta.get("mode")),
            "gps_scn": int(meta.get("gps_scn")),
            "imu_scn": int(meta.get("imu_scn")),
            "gps_seed": int(meta.get("gps_seed")),
            "imu_seed": int(meta.get("imu_seed")),
            "run_dir": run_dir,
        }
    return existing, max_index


def build_manifest(args, apply_max_runs=True):
    missions = parse_str_list(args.missions)
    modes = parse_int_list(args.modes)
    seeds = parse_int_list(args.seeds)
    combos = parse_attack_combos(args.attack_combos) if args.attack_combos else attack_combos_for_grid(args.attack_grid)

    rows = []
    index = 0
    for mission in missions:
        for mode in modes:
            for seed in seeds:
                for gps_scn, imu_scn in combos:
                    index += 1
                    run_id = run_id_for(index, mission, mode, gps_scn, imu_scn, seed)
                    rows.append(
                        {
                            "run_index": index,
                            "run_id": run_id,
                            "mission": mission,
                            "mode": mode,
                            "gps_scn": gps_scn,
                            "imu_scn": imu_scn,
                            "gps_seed": seed,
                            "imu_seed": seed,
                            "run_dir": os.path.join(args.output_dir, run_id),
                        }
                    )
    if apply_max_runs and args.max_runs is not None:
        rows = rows[: args.max_runs]
    return rows


def build_append_missing_manifest(args):
    planned = build_manifest(args, apply_max_runs=False)
    all_existing, max_index = read_existing_run_meta(args.output_dir, completed_only=False)
    completed_existing, _ = read_existing_run_meta(args.output_dir, completed_only=True)
    missing_rows = []
    full_rows = []
    index = max_index
    for row in planned:
        key = condition_key_from_row(row)
        if key in completed_existing:
            full_rows.append(completed_existing[key])
            continue
        if key in all_existing:
            full_rows.append(all_existing[key])
            missing_rows.append(all_existing[key])
            continue
        index += 1
        run_id = run_id_for(index, row["mission"], row["mode"], row["gps_scn"], row["imu_scn"], row["gps_seed"])
        row = dict(row)
        row["run_index"] = index
        row["run_id"] = run_id
        row["run_dir"] = os.path.join(args.output_dir, run_id)
        missing_rows.append(row)
        full_rows.append(row)
    if args.max_runs is not None:
        missing_rows = missing_rows[: args.max_runs]
    full_rows = sorted(full_rows, key=lambda item: int(item.get("run_index") or 0))
    return missing_rows, full_rows, len(completed_existing), len(all_existing)


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json_safely(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def make_attempt_dir(run_dir):
    ensure_dir(run_dir)
    base = "attempt_%s" % now_stamp()
    attempt_dir = os.path.join(run_dir, base)
    index = 2
    while os.path.exists(attempt_dir):
        attempt_dir = os.path.join(run_dir, "%s_%02d" % (base, index))
        index += 1
    ensure_dir(attempt_dir)
    return os.path.basename(attempt_dir), attempt_dir


def find_latest_completed_attempt_dir(run_dir):
    candidates = []
    for attempt_dir in glob.glob(os.path.join(run_dir, "attempt_*")):
        result_path = os.path.join(attempt_dir, "run_result.json")
        result = read_json_safely(result_path)
        if not result or not result.get("attempt_complete"):
            continue
        if result.get("status") in RETRYABLE_LAUNCH_STATUSES:
            continue
        candidates.append((os.path.getmtime(result_path), attempt_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def is_stress_run(args):
    return (getattr(args, "attack_grid", "") or "").lower() == "stress" and not getattr(args, "attack_combos", "")


def defer_attack_until_after_takeoff(args):
    if getattr(args, "defer_attack_until_after_takeoff", False):
        return True
    return is_stress_run(args) and not getattr(args, "keep_attack_during_launch", False)


def use_isolated_sitl_state(args):
    if getattr(args, "isolated_sitl_state", False):
        return True
    return is_stress_run(args) and not getattr(args, "shared_sitl_state", False)


def configure_stress_defaults(args):
    if is_stress_run(args):
        if not getattr(args, "allow_stress_mode_subset", False):
            modes = set(parse_int_list(args.modes))
            modes.update([0, 2, 4])
            args.modes = ",".join(str(mode) for mode in sorted(modes))
        if not getattr(args, "keep_failsafes_in_stress", False) and not getattr(args, "disable_failsafes_after_takeoff", False):
            args.disable_failsafes_after_takeoff = True


def post_takeoff_param_list(args):
    params = []
    if getattr(args, "disable_failsafes_after_takeoff", False):
        params.extend(STRESS_POST_TAKEOFF_PARAMS)
    params.extend(parse_param_assignments(getattr(args, "post_takeoff_param", "") or ""))
    return params


def attack_param_list(meta, gps_scn=None, imu_scn=None):
    if gps_scn is None:
        gps_scn = meta["gps_scn"]
    if imu_scn is None:
        imu_scn = meta["imu_scn"]
    return [
        ("SIM_GPS_ATK_SEED", meta["gps_seed"]),
        ("SIM_IMU_ATK_SEED", meta["imu_seed"]),
        ("SIM_GPS_ATK_SCN", gps_scn),
        ("SIM_IMU_ATK_SCN", imu_scn),
    ]


def launch_param_list(meta, defer_attack=False):
    gps_scn = 0 if defer_attack else meta["gps_scn"]
    imu_scn = 0 if defer_attack else meta["imu_scn"]
    params = [
        ("ATC_PIPER_MODE", meta["mode"]),
    ]
    params.extend(attack_param_list(meta, gps_scn=gps_scn, imu_scn=imu_scn))
    return params


def write_param_file(path, params):
    with open(path, "w") as f:
        for name, value in params:
            f.write("%s %s\n" % (name, value))


def prepare_sitl_state_dir(state_dir):
    if not state_dir:
        return
    ensure_dir(state_dir)
    link_path = os.path.join(state_dir, "build")
    target = os.path.join(SIMULATOR_DIR, "build")
    if os.path.exists(link_path):
        return
    try:
        os.symlink(target, link_path)
    except OSError:
        if os.path.isdir(target):
            shutil.copytree(target, link_path, symlinks=True)


def post_takeoff_param_list_for_run(args, meta):
    return list(post_takeoff_param_list(args))


def post_auto_param_list_for_run(args, meta):
    params = []
    if defer_attack_until_after_takeoff(args):
        params.extend(attack_param_list(meta))
    return params


def pad_param(name):
    encoded = name.encode("ascii")
    if len(encoded) > 16:
        raise ValueError("MAVLink param id too long: %s" % name)
    return encoded + b"\x00" * (16 - len(encoded))


def decode_param_id(raw):
    if isinstance(raw, bytes):
        return raw.decode("ascii", "ignore").rstrip("\x00")
    return str(raw).rstrip("\x00")


def set_param(mav, name, value, attempts=3, timeout=2.0):
    if mavutil is None:
        raise RuntimeError("pymavlink is required to set simulator parameters")

    target_system = getattr(mav, "target_system", 1) or 1
    target_component = getattr(mav, "target_component", 1) or 1
    param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    for _ in range(attempts):
        mav.mav.param_set_send(target_system, target_component, pad_param(name), float(value), param_type)
        end = time.time() + timeout
        while time.time() < end:
            msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.25)
            if msg is None:
                continue
            if decode_param_id(msg.param_id) != name:
                continue
            got = safe_float(msg.param_value)
            if got is not None and abs(got - float(value)) < 0.01:
                return True
        mav.mav.param_request_read_send(target_system, target_component, pad_param(name), -1)
    return False


def set_run_params(mav, meta, defer_attack=False):
    params = launch_param_list(meta, defer_attack=defer_attack)
    failed = []
    for name, value in params:
        ok = set_param(mav, name, value)
        if not ok:
            failed.append(name)
    if failed:
        raise RuntimeError("failed to confirm parameter(s): %s" % ", ".join(failed))


def start_sim(args, run_dir, stdout_path, state_dir=None, launch_param_path=None):
    sim_vehicle_path = os.path.join(SCRIPT_DIR, "sim_vehicle.py")
    sim_cmd = [args.sim_vehicle_py, sim_vehicle_path, "--aircraft", args.aircraft]
    if not args.sim_rebuild:
        sim_cmd.append("--no-rebuild")
    if args.show_console:
        sim_cmd.append("--console")
    if args.show_map:
        sim_cmd.append("--map")
    for item in args.extra_sim_arg or []:
        sim_cmd.append(item)
    if state_dir:
        sim_cmd.extend(["--use-dir", os.path.abspath(state_dir), "--wipe-eeprom"])
    if launch_param_path:
        sim_cmd.extend(["--add-param-file", os.path.abspath(launch_param_path)])

    env = os.environ.copy()
    env["PID_PIPER_OUTPUT_DIR"] = os.path.abspath(run_dir)
    env["PID_PIPER_ROOT"] = REPO_ROOT
    env["PID_PIPER_MODEL_DIR"] = os.path.join(SIMULATOR_DIR, "libraries", "PID_Piper", "models")

    stdout_fh = open(stdout_path, "w", buffering=1)
    creationflags = 0
    preexec_fn = None
    if os.name == "posix":
        preexec_fn = os.setsid
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        sim_cmd,
        cwd=ARDUCOPTER_DIR,
        stdin=subprocess.PIPE,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
        preexec_fn=preexec_fn,
        creationflags=creationflags,
    )
    return proc, stdout_fh, sim_cmd


def stop_sim(proc, stdout_fh=None):
    if proc is not None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
    if stdout_fh is not None:
        try:
            stdout_fh.close()
        except Exception:
            pass


def force_cleanup_leftovers():
    if os.name != "posix":
        return
    patterns = [
        "sim_vehicle.py",
        "MAVProxy.py",
        "mavproxy",
        "arducopter",
        "ArduCopter.elf",
        "xterm",
        "gzserver",
        "gzclient",
    ]
    for pattern in patterns:
        try:
            subprocess.call(["pkill", "-TERM", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(1.0)
    for pattern in patterns:
        try:
            subprocess.call(["pkill", "-KILL", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def send_console_command(proc, command, command_log_path=None):
    if proc is None or proc.stdin is None:
        raise RuntimeError("simulator stdin is not available")
    print("[CMD] %s" % command)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    if command_log_path:
        with open(command_log_path, "a") as f:
            f.write("%.3f %s\n" % (time.time(), command))


def send_console_commands(proc, commands, interval_sec, command_log_path):
    for command in commands:
        send_console_command(proc, command, command_log_path)
        time.sleep(interval_sec)


def mission_path_for_console(mission_name):
    return os.path.join(MISSION_DIR, os.path.basename(mission_name))


def parse_mission_points(mission_name):
    path = mission_name
    if not os.path.isabs(path):
        path = os.path.join(MISSION_DIR, mission_name)
    points = []
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("QGC"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            command = safe_int(parts[3])
            lat = safe_float(parts[8])
            lon = safe_float(parts[9])
            alt = safe_float(parts[10])
            if lat is None or lon is None:
                continue
            points.append({"command": command, "lat": lat, "lon": lon, "alt": alt})
    home = None
    target = None
    for p in points:
        if abs(p["lat"]) > 1e-7 and abs(p["lon"]) > 1e-7:
            if home is None:
                home = p
            elif target is None and p.get("command") == 16:
                target = p
                break
    if home is None or target is None:
        return None
    return {"home": home, "target": target}


def update_state_from_msg(state, msg, statustext_fh=None):
    msg_type = msg.get_type()
    if msg_type == "GPS_RAW_INT":
        state["gps_fix_type"] = getattr(msg, "fix_type", None)
        state["gps_satellites_visible"] = getattr(msg, "satellites_visible", None)
    elif msg_type == "GLOBAL_POSITION_INT":
        state["lat"] = msg.lat / 1e7
        state["lon"] = msg.lon / 1e7
        state["relative_alt_m"] = msg.relative_alt / 1000.0
        state["amsl_m"] = msg.alt / 1000.0
        state["vx_mps"] = getattr(msg, "vx", 0) / 100.0
        state["vy_mps"] = getattr(msg, "vy", 0) / 100.0
        state["vz_mps"] = getattr(msg, "vz", 0) / 100.0
    elif msg_type == "ATTITUDE":
        state["roll_rad"] = msg.roll
        state["pitch_rad"] = msg.pitch
        state["yaw_rad"] = msg.yaw
        state["rollspeed_rad_s"] = msg.rollspeed
        state["pitchspeed_rad_s"] = msg.pitchspeed
        state["yawspeed_rad_s"] = msg.yawspeed
    elif msg_type == "HEARTBEAT":
        if mavutil is not None:
            try:
                state["mode"] = mavutil.mode_string_v10(msg)
            except Exception:
                state["mode"] = ""
            state["armed"] = 1 if (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 0
    elif msg_type == "EXTENDED_SYS_STATE":
        state["landed_state"] = getattr(msg, "landed_state", None)
    elif msg_type == "VFR_HUD":
        state["groundspeed_mps"] = getattr(msg, "groundspeed", None)
        state["airspeed_mps"] = getattr(msg, "airspeed", None)
        state["heading_deg"] = getattr(msg, "heading", None)
    elif msg_type == "MISSION_CURRENT":
        state["mission_seq"] = getattr(msg, "seq", None)
    elif msg_type == "NAV_CONTROLLER_OUTPUT":
        state["nav_bearing"] = getattr(msg, "nav_bearing", None)
        state["target_bearing"] = getattr(msg, "target_bearing", None)
        state["wp_dist_m"] = getattr(msg, "wp_dist", None)
        state["alt_error_m"] = getattr(msg, "alt_error", None)
        state["aspd_error_mps"] = getattr(msg, "aspd_error", None)
        state["xtrack_error_m"] = getattr(msg, "xtrack_error", None)
    elif msg_type == "SYS_STATUS":
        state["battery_remaining_pct"] = getattr(msg, "battery_remaining", None)
        voltage_mv = getattr(msg, "voltage_battery", None)
        current_ca = getattr(msg, "current_battery", None)
        if voltage_mv is not None and voltage_mv != 65535:
            state["voltage_battery_v"] = voltage_mv / 1000.0
        if current_ca is not None and current_ca != -1:
            state["current_battery_a"] = current_ca / 100.0
    elif msg_type == "STATUSTEXT" and statustext_fh is not None:
        try:
            text = msg.text
        except Exception:
            text = str(msg)
        statustext_fh.write("%.3f %s\n" % (time.time(), text))
        statustext_fh.flush()


def telemetry_row(state, start_t):
    row = dict((k, "") for k in TELEMETRY_COLUMNS)
    row["run_time_s"] = "%.3f" % (time.time() - start_t)
    row["wall_time"] = "%.3f" % time.time()
    for key in TELEMETRY_COLUMNS:
        if key in ("run_time_s", "wall_time"):
            continue
        value = state.get(key)
        if value is None:
            continue
        row[key] = value
    return row


def is_landed_state_on_ground(state):
    if mavutil is None:
        return False
    landed = state.get("landed_state")
    return landed == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND


def speed_for_landing(state):
    gs = safe_float(state.get("groundspeed_mps"))
    if gs is not None:
        return abs(gs)
    vx = safe_float(state.get("vx_mps"), 0.0)
    vy = safe_float(state.get("vy_mps"), 0.0)
    return math.sqrt(vx * vx + vy * vy)


def final_distance_to_target_from_state(state, mission_points):
    lat = safe_float(state.get("lat"))
    lon = safe_float(state.get("lon"))
    _, final_dist = line_metrics_for_point(lat, lon, mission_points)
    return final_dist


def annotate_success_gate(result, state, mission_points, args):
    rel_alt = safe_float(state.get("relative_alt_m"))
    final_dist = final_distance_to_target_from_state(state, mission_points)
    groundspeed = speed_for_landing(state)
    low_alt = rel_alt is not None and rel_alt <= args.land_alt
    near_target = final_dist is not None and final_dist <= args.success_radius_m
    result["success_relative_alt_m"] = rel_alt
    result["success_final_dist_to_target_m"] = final_dist
    result["success_groundspeed_mps"] = groundspeed
    result["success_land_alt_m"] = args.land_alt
    result["success_radius_m"] = args.success_radius_m
    result["success_low_alt"] = low_alt
    result["success_near_target"] = near_target
    return low_alt and near_target


def finish_with_strict_success(result, status_prefix, state, mission_points, args):
    strict_success = annotate_success_gate(result, state, mission_points, args)
    result["success"] = bool(strict_success)
    if strict_success:
        result["status"] = status_prefix
    else:
        result["status"] = "%s_outside_success_gate" % status_prefix
    return result


def pump_mavlink_until(mav, proc, state, predicate, timeout_sec, statustext_fh, telemetry_writer=None, telemetry_start_t=None, sample_interval=0.2):
    start_t = time.time()
    last_msg_t = time.time()
    last_sample_t = 0.0
    while True:
        now = time.time()
        if predicate(state):
            return True
        if proc is not None and proc.poll() is not None and now - start_t > 5:
            return False
        if now - start_t > timeout_sec:
            return False
        if now - last_msg_t > 20:
            return False

        msg = mav.recv_match(type=MAVLINK_TYPES, blocking=True, timeout=0.25)
        if msg is not None:
            last_msg_t = time.time()
            update_state_from_msg(state, msg, statustext_fh=statustext_fh)

        now = time.time()
        if telemetry_writer is not None and telemetry_start_t is not None and now - last_sample_t >= sample_interval:
            telemetry_writer.writerow(telemetry_row(state, telemetry_start_t))
            last_sample_t = now


def gps_has_3d_fix(state):
    fix_type = safe_int(state.get("gps_fix_type"), 0)
    return fix_type >= 3


def wait_for_gps_fix(mav, proc, state, args, statustext_fh, telemetry_writer, telemetry_start_t):
    print("[INFO] Waiting for GPS 3D fix before arming")
    ok = pump_mavlink_until(
        mav,
        proc,
        state,
        gps_has_3d_fix,
        args.prearm_timeout,
        statustext_fh,
        telemetry_writer,
        telemetry_start_t,
        args.sample_interval,
    )
    if ok:
        sats = state.get("gps_satellites_visible")
        print("[INFO] GPS fix ready: fix_type=%s satellites=%s" % (state.get("gps_fix_type"), sats))
    return ok


def wait_for_armed(mav, proc, state, args, statustext_fh, telemetry_writer, telemetry_start_t, timeout_sec=None):
    if timeout_sec is None:
        timeout_sec = args.command_interval
    return pump_mavlink_until(
        mav,
        proc,
        state,
        lambda item: item.get("armed") == 1,
        timeout_sec,
        statustext_fh,
        telemetry_writer,
        telemetry_start_t,
        args.sample_interval,
    )


def wait_for_mode(mav, proc, state, mode_name, timeout_sec, statustext_fh, telemetry_writer, telemetry_start_t, sample_interval):
    desired = str(mode_name).upper()
    return pump_mavlink_until(
        mav,
        proc,
        state,
        lambda item: str(item.get("mode") or "").upper() == desired,
        timeout_sec,
        statustext_fh,
        telemetry_writer,
        telemetry_start_t,
        sample_interval,
    )


def prepare_takeoff(mav, proc, args, mission_console_path, command_log_path, statustext_fh=None, telemetry_writer=None, telemetry_start_t=None):
    state = {}
    start_t = telemetry_start_t or time.time()
    send_console_command(proc, "wp load %s" % mission_console_path, command_log_path)
    time.sleep(args.command_interval)
    send_console_command(proc, "mode guided", command_log_path)
    time.sleep(args.command_interval)

    if not wait_for_gps_fix(mav, proc, state, args, statustext_fh, telemetry_writer, telemetry_start_t):
        return state, {"status": "prearm_timeout", "success": False, "duration_s": time.time() - start_t}

    arm_deadline = time.time() + args.arm_timeout
    while time.time() < arm_deadline:
        send_console_command(proc, "arm throttle", command_log_path)
        remaining = max(0.0, arm_deadline - time.time())
        wait_window = min(max(args.command_interval, 0.5), remaining)
        if wait_for_armed(mav, proc, state, args, statustext_fh, telemetry_writer, telemetry_start_t, wait_window):
            print("[INFO] Vehicle armed")
            send_console_command(proc, "takeoff %.1f" % args.takeoff_alt, command_log_path)
            return state, None

    return state, {"status": "arm_timeout", "success": False, "duration_s": time.time() - start_t}


def monitor_flight(mav, proc, args, telemetry_path, statustext_path, command_log_path, meta):
    state = {}
    mission_name = meta["mission"]
    mission_points = parse_mission_points(mission_name)
    post_takeoff_params = post_takeoff_param_list_for_run(args, meta)
    post_auto_params = post_auto_param_list_for_run(args, meta)
    result = {
        "status": "unknown",
        "success": False,
        "start_wall_time": time.time(),
        "auto_sent_wall_time": None,
        "finish_wall_time": None,
        "time_to_auto_s": None,
        "time_to_land_s": None,
        "timeout_sec": args.timeout_sec,
        "strict_success_requires_auto": True,
        "strict_success_requires_low_alt": True,
        "strict_success_requires_near_target": True,
        "success_radius_m": args.success_radius_m,
        "post_takeoff_params": dict(post_takeoff_params),
        "post_takeoff_params_applied": False,
        "post_auto_params": dict(post_auto_params),
        "post_auto_params_applied": False,
        "attack_deferred_until_after_takeoff": defer_attack_until_after_takeoff(args),
    }

    start_t = time.time()
    last_sample_t = 0.0
    last_msg_t = time.time()
    takeoff_ok_since = None
    land_ok_since = None
    auto_sent = False
    auto_confirmed = False

    with open(telemetry_path, "w", newline="") as telem_f, open(statustext_path, "a") as statustext_f:
        writer = csv.DictWriter(telem_f, fieldnames=TELEMETRY_COLUMNS)
        writer.writeheader()

        while True:
            now = time.time()
            if proc is not None and proc.poll() is not None and now - start_t > 5:
                result["status"] = "sim_exited"
                break
            if now - last_msg_t > 20:
                result["status"] = "mavlink_quiet"
                break

            msg = mav.recv_match(type=MAVLINK_TYPES, blocking=True, timeout=0.25)
            if msg is not None:
                last_msg_t = time.time()
                update_state_from_msg(state, msg, statustext_fh=statustext_f)

            now = time.time()
            if now - last_sample_t >= args.sample_interval:
                writer.writerow(telemetry_row(state, start_t))
                telem_f.flush()
                last_sample_t = now

            rel_alt = safe_float(state.get("relative_alt_m"))
            if not auto_sent:
                if now - start_t > args.takeoff_timeout:
                    result["status"] = "takeoff_timeout"
                    break
                if rel_alt is not None and rel_alt >= args.takeoff_alt:
                    if takeoff_ok_since is None:
                        takeoff_ok_since = now
                    elif now - takeoff_ok_since >= args.takeoff_sustain:
                        print("[INFO] Takeoff altitude sustained; switching to AUTO")
                        if post_takeoff_params:
                            print("[INFO] Applying post-takeoff params: %s" % ", ".join("%s=%s" % item for item in post_takeoff_params))
                            failed_params = []
                            for name, value in post_takeoff_params:
                                if not set_param(mav, name, value):
                                    failed_params.append(name)
                            if failed_params:
                                result["status"] = "post_takeoff_param_failed"
                                result["post_takeoff_param_failed"] = ",".join(failed_params)
                                break
                            result["post_takeoff_params_applied"] = True
                        send_console_command(proc, "mode auto", command_log_path)
                        auto_ok = wait_for_mode(
                            mav,
                            proc,
                            state,
                            "AUTO",
                            args.auto_mode_timeout,
                            statustext_f,
                            writer,
                            start_t,
                            args.sample_interval,
                        )
                        now = time.time()
                        auto_sent = True
                        if not auto_ok:
                            result["status"] = "auto_mode_timeout"
                            break
                        auto_confirmed = True
                        if post_auto_params:
                            print("[INFO] Applying post-AUTO params: %s" % ", ".join("%s=%s" % item for item in post_auto_params))
                            failed_params = []
                            for name, value in post_auto_params:
                                if not set_param(mav, name, value):
                                    failed_params.append(name)
                            if failed_params:
                                result["status"] = "post_auto_param_failed"
                                result["post_auto_param_failed"] = ",".join(failed_params)
                                break
                            result["post_auto_params_applied"] = True
                        result["auto_confirmed"] = True
                        result["auto_sent_wall_time"] = now
                        result["auto_confirmed_wall_time"] = now
                        result["time_to_auto_s"] = now - start_t
                else:
                    takeoff_ok_since = None
                continue

            auto_elapsed = now - result["auto_sent_wall_time"]
            result["auto_confirmed"] = bool(auto_confirmed)
            if auto_elapsed > args.timeout_sec:
                result["status"] = "timeout_auto"
                break

            armed = state.get("armed")
            if armed == 0 and auto_elapsed > 5:
                finish_with_strict_success(result, "landed_disarmed", state, mission_points, args)
                break
            if is_landed_state_on_ground(state) and auto_elapsed > 5:
                finish_with_strict_success(result, "landed_state_ground", state, mission_points, args)
                break
            if rel_alt is not None and rel_alt <= args.land_alt and speed_for_landing(state) <= args.land_speed:
                if land_ok_since is None:
                    land_ok_since = now
                elif now - land_ok_since >= args.land_sustain:
                    finish_with_strict_success(result, "landed_low_alt_speed", state, mission_points, args)
                    break
            else:
                land_ok_since = None

    finish_t = time.time()
    result["finish_wall_time"] = finish_t
    if result["auto_sent_wall_time"] is not None:
        result["time_to_land_s"] = finish_t - result["auto_sent_wall_time"]
    result["duration_s"] = finish_t - start_t
    return result


def arducopter_state_dirs(aircraft=None, extra_dirs=None):
    if extra_dirs:
        return list(extra_dirs)
    dirs = [ARDUCOPTER_DIR]
    if aircraft:
        dirs.append(os.path.join(ARDUCOPTER_DIR, aircraft))
    return dirs


def list_bin_logs(aircraft=None, extra_dirs=None):
    out = set()
    for state_dir in arducopter_state_dirs(aircraft, extra_dirs=extra_dirs):
        logs_dir = os.path.join(state_dir, "logs")
        if os.path.isdir(logs_dir):
            out.update(os.path.abspath(p) for p in glob.glob(os.path.join(logs_dir, "*.BIN")))
        out.update(os.path.abspath(p) for p in glob.glob(os.path.join(state_dir, "**", "*.BIN"), recursive=True))
    return out


def latest_log_from_lastlog(aircraft=None, extra_dirs=None):
    for state_dir in arducopter_state_dirs(aircraft, extra_dirs=extra_dirs):
        lastlogs = [os.path.join(state_dir, "logs", "LASTLOG.TXT")]
        lastlogs.extend(glob.glob(os.path.join(state_dir, "**", "LASTLOG.TXT"), recursive=True))
        for lastlog in lastlogs:
            logs_dir = os.path.dirname(lastlog)
            if not os.path.isfile(lastlog):
                continue
            try:
                with open(lastlog, "r") as f:
                    text = f.read().strip()
                number = int(text)
            except Exception:
                continue
            candidates = [
                os.path.join(logs_dir, "%08d.BIN" % number),
                os.path.join(logs_dir, "%08d.bin" % number),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    return os.path.abspath(path)
    return None


def copy_flight_log(before_logs, run_dir, aircraft=None, extra_dirs=None):
    after_logs = list_bin_logs(aircraft, extra_dirs=extra_dirs)
    candidates = sorted(list(after_logs - before_logs), key=lambda p: os.path.getmtime(p), reverse=True)
    lastlog = latest_log_from_lastlog(aircraft, extra_dirs=extra_dirs)
    if lastlog is not None and os.path.abspath(lastlog) in after_logs and os.path.abspath(lastlog) not in before_logs:
        candidates.insert(0, os.path.abspath(lastlog))
    if not candidates:
        return None
    src = candidates[0]
    dst = os.path.join(run_dir, "flight.BIN")
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def copy_param_snapshot(run_dir, aircraft=None, extra_dirs=None):
    for state_dir in arducopter_state_dirs(aircraft, extra_dirs=extra_dirs):
        candidates = [
            os.path.join(state_dir, "mav.parm"),
        ]
        if aircraft:
            candidates.append(os.path.join(state_dir, aircraft, "mav.parm"))
        candidates.extend(sorted(
            glob.glob(os.path.join(state_dir, "**", "mav.parm"), recursive=True),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        ))
        seen = set()
        for src in candidates:
            src = os.path.abspath(src)
            if src in seen or not os.path.isfile(src):
                continue
            seen.add(src)
            dst = os.path.join(run_dir, "mav.parm")
            try:
                if os.path.abspath(src) == os.path.abspath(dst):
                    return src
                shutil.copy2(src, dst)
                return dst
            except Exception:
                continue
    return None


def waf_configured():
    cache_path = os.path.join(SIMULATOR_DIR, "build", "c4che", "_cache.py")
    return os.path.isfile(cache_path)


def run_build(args):
    if args.skip_build:
        print("[INFO] Skipping waf build")
        return
    if not waf_configured():
        print("[RUN] ./waf configure --board sitl")
        subprocess.check_call(["./waf", "configure", "--board", "sitl"], cwd=SIMULATOR_DIR)
    print("[RUN] ./waf copter")
    subprocess.check_call(["./waf", "copter"], cwd=SIMULATOR_DIR)


def connect_mavlink(addr, heartbeat_timeout):
    if mavutil is None:
        raise RuntimeError("pymavlink is required for live SITL experiments")
    mav = mavutil.mavlink_connection(addr)
    mav.wait_heartbeat(timeout=heartbeat_timeout)
    return mav


def run_one(args, meta):
    run_dir = meta["run_dir"]
    ensure_dir(run_dir)
    root_meta = dict(meta)
    root_meta["run_dir"] = run_dir
    root_meta["last_started_at"] = datetime.now().isoformat()

    attempt_id, attempt_dir = make_attempt_dir(run_dir)
    attempt_meta = dict(root_meta)
    attempt_meta["attempt_id"] = attempt_id
    attempt_meta["attempt_dir"] = attempt_dir
    attempt_meta["started_at"] = datetime.now().isoformat()
    root_meta["current_attempt_id"] = attempt_id
    root_meta["current_attempt_dir"] = attempt_dir
    write_json(os.path.join(run_dir, "run_meta.json"), root_meta)
    write_json(os.path.join(attempt_dir, "run_meta.json"), attempt_meta)

    stdout_path = os.path.join(attempt_dir, "stdout.log")
    telemetry_path = os.path.join(attempt_dir, "telemetry.csv")
    statustext_path = os.path.join(attempt_dir, "statustext.log")
    command_log_path = os.path.join(attempt_dir, "commands.log")
    defer_attack = defer_attack_until_after_takeoff(args)
    state_dir = os.path.join(attempt_dir, "sitl_state") if use_isolated_sitl_state(args) else None
    launch_param_path = os.path.join(attempt_dir, "launch_params.parm")
    prepare_sitl_state_dir(state_dir)
    write_param_file(launch_param_path, launch_param_list(meta, defer_attack=defer_attack))
    state_dirs = [state_dir] if state_dir else None
    before_logs = list_bin_logs(args.aircraft, extra_dirs=state_dirs)
    proc = None
    stdout_fh = None
    mav = None
    result = None

    print("[RUN] %s %s" % (meta["run_id"], attempt_id))
    try:
        proc, stdout_fh, sim_cmd = start_sim(
            args,
            attempt_dir,
            stdout_path,
            state_dir=state_dir,
            launch_param_path=launch_param_path,
        )
        attempt_meta["sim_command"] = sim_cmd
        attempt_meta["sitl_state_dir"] = state_dir
        attempt_meta["launch_param_path"] = launch_param_path
        attempt_meta["launch_params"] = dict(launch_param_list(meta, defer_attack=defer_attack))
        root_meta["sim_command"] = sim_cmd
        root_meta["sitl_state_dir"] = state_dir
        root_meta["launch_param_path"] = launch_param_path
        root_meta["launch_params"] = dict(launch_param_list(meta, defer_attack=defer_attack))
        write_json(os.path.join(run_dir, "run_meta.json"), root_meta)
        write_json(os.path.join(attempt_dir, "run_meta.json"), attempt_meta)
        time.sleep(args.start_settle)

        mav = connect_mavlink(args.mavlink, args.heartbeat_timeout)
        print("[INFO] Heartbeat received")
        time.sleep(args.post_heartbeat_settle)

        if defer_attack:
            print("[INFO] Deferring attack scenarios until after takeoff: launch SIM_GPS_ATK_SCN=0 SIM_IMU_ATK_SCN=0")
        set_run_params(mav, meta, defer_attack=defer_attack)

        mission_console_path = mission_path_for_console(meta["mission"])
        result = None
        with open(statustext_path, "w") as preflight_statustext:
            _, result = prepare_takeoff(
                mav,
                proc,
                args,
                mission_console_path,
                command_log_path,
                statustext_fh=preflight_statustext,
            )
        if result is None:
            result = monitor_flight(mav, proc, args, telemetry_path, statustext_path, command_log_path, meta)
    except KeyboardInterrupt:
        result = {
            "status": "interrupted",
            "success": False,
            "attempt_complete": False,
            "interrupted_at": datetime.now().isoformat(),
        }
        raise
    except Exception as exc:
        result = {
            "status": "exception",
            "success": False,
            "attempt_complete": True,
            "exception": repr(exc),
            "duration_s": None,
        }
        eprint("[ERROR] %s failed: %s" % (meta["run_id"], exc))
    finally:
        try:
            stop_sim(proc, stdout_fh)
        finally:
            if args.force_cleanup:
                force_cleanup_leftovers()
        if mav is not None:
            try:
                mav.close()
            except Exception:
                pass
        if result is not None and result.get("status") == "interrupted":
            bin_path = copy_flight_log(before_logs, attempt_dir, args.aircraft, extra_dirs=state_dirs)
            parm_path = copy_param_snapshot(attempt_dir, args.aircraft, extra_dirs=state_dirs)
            result["bin_path"] = bin_path
            result["param_snapshot_path"] = parm_path
            result["finished_at"] = datetime.now().isoformat()
            result["attempt_id"] = attempt_id
            result["attempt_dir"] = attempt_dir
            write_json(os.path.join(attempt_dir, "run_result.json"), result)

    bin_path = copy_flight_log(before_logs, attempt_dir, args.aircraft, extra_dirs=state_dirs)
    parm_path = copy_param_snapshot(attempt_dir, args.aircraft, extra_dirs=state_dirs)
    result["bin_path"] = bin_path
    result["param_snapshot_path"] = parm_path
    result["finished_at"] = datetime.now().isoformat()
    result["attempt_complete"] = True
    result["attempt_id"] = attempt_id
    result["attempt_dir"] = attempt_dir
    write_json(os.path.join(attempt_dir, "run_result.json"), result)

    analysis = analyze_run(attempt_dir)
    analysis["attempt_complete"] = True
    analysis["attempt_id"] = attempt_id
    analysis["attempt_dir"] = attempt_dir
    analysis["artifact_dir"] = attempt_dir
    write_json(os.path.join(attempt_dir, "analysis.json"), analysis)
    write_json(os.path.join(run_dir, "run_result.json"), result)
    write_json(os.path.join(run_dir, "analysis.json"), analysis)
    return analysis


def rms(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / float(len(values)))


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return sum(values) / float(len(values))


def std(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) * (v - m) for v in values) / float(len(values) - 1))


def max_abs(values):
    values = [abs(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return max(values)


def project_latlon(lat, lon, ref_lat, ref_lon):
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    x = (lon - ref_lon) * meters_per_deg_lon
    y = (lat - ref_lat) * meters_per_deg_lat
    return x, y


def line_metrics_for_point(lat, lon, mission_points):
    if mission_points is None or lat is None or lon is None:
        return None, None
    home = mission_points["home"]
    target = mission_points["target"]
    px, py = project_latlon(lat, lon, home["lat"], home["lon"])
    tx, ty = project_latlon(target["lat"], target["lon"], home["lat"], home["lon"])
    line_len = math.sqrt(tx * tx + ty * ty)
    if line_len <= 1e-6:
        return None, None
    cross_track = abs(tx * py - ty * px) / line_len
    final_dist = math.sqrt((px - tx) * (px - tx) + (py - ty) * (py - ty))
    return cross_track, final_dist


def read_telemetry_metrics(path, mission_name):
    metrics = {}
    if not os.path.isfile(path):
        return metrics

    mission_points = parse_mission_points(mission_name)
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    metrics["telemetry_rows"] = len(rows)
    if not rows:
        return metrics

    rel_alt = []
    groundspeed = []
    xtrack = []
    final_dists = []
    modes = []
    landed_states = []
    vx = []
    vy = []
    vz = []
    times = []
    rollspeed = []
    pitchspeed = []
    yawspeed = []

    for row in rows:
        t = safe_float(row.get("run_time_s"))
        if t is not None:
            times.append(t)
        rel_alt.append(safe_float(row.get("relative_alt_m")))
        mode = (row.get("mode") or "").strip()
        if mode:
            modes.append(mode)
        landed_state = safe_int(row.get("landed_state"))
        if landed_state is not None:
            landed_states.append(landed_state)
        groundspeed.append(safe_float(row.get("groundspeed_mps")))
        vx.append(safe_float(row.get("vx_mps")))
        vy.append(safe_float(row.get("vy_mps")))
        vz.append(safe_float(row.get("vz_mps")))
        rollspeed.append(safe_float(row.get("rollspeed_rad_s")))
        pitchspeed.append(safe_float(row.get("pitchspeed_rad_s")))
        yawspeed.append(safe_float(row.get("yawspeed_rad_s")))
        lat = safe_float(row.get("lat"))
        lon = safe_float(row.get("lon"))
        ct, fd = line_metrics_for_point(lat, lon, mission_points)
        if ct is not None:
            xtrack.append(ct)
        if fd is not None:
            final_dists.append(fd)

    metrics["max_relative_alt_m"] = max([v for v in rel_alt if v is not None], default=None)
    nonempty_rel_alt = [v for v in rel_alt if v is not None]
    metrics["final_relative_alt_m"] = nonempty_rel_alt[-1] if nonempty_rel_alt else None
    metrics["auto_mode_seen"] = any(str(mode).upper() == "AUTO" for mode in modes)
    metrics["final_mode"] = modes[-1] if modes else None
    metrics["final_landed_state"] = landed_states[-1] if landed_states else None
    metrics["mean_groundspeed_mps"] = mean(groundspeed)
    metrics["max_groundspeed_mps"] = max([v for v in groundspeed if v is not None], default=None)
    metrics["mean_xtrack_error_m"] = mean(xtrack)
    metrics["max_xtrack_error_m"] = max(xtrack) if xtrack else None
    metrics["final_dist_to_target_m"] = final_dists[-1] if final_dists else None
    metrics["min_final_dist_to_target_m"] = min(final_dists) if final_dists else None
    metrics["attitude_rate_rms_rad_s"] = rms(
        [
            math.sqrt((rs or 0.0) ** 2 + (ps or 0.0) ** 2 + (ys or 0.0) ** 2)
            for rs, ps, ys in zip(rollspeed, pitchspeed, yawspeed)
            if rs is not None or ps is not None or ys is not None
        ]
    )

    accel = []
    attitude_rate_delta = []
    for i in range(1, min(len(times), len(vx), len(vy), len(vz))):
        dt = times[i] - times[i - 1]
        if dt <= 1e-6:
            continue
        vals = [vx[i], vy[i], vz[i], vx[i - 1], vy[i - 1], vz[i - 1]]
        if any(v is None for v in vals):
            continue
        ax = (vx[i] - vx[i - 1]) / dt
        ay = (vy[i] - vy[i - 1]) / dt
        az = (vz[i] - vz[i - 1]) / dt
        accel.append(math.sqrt(ax * ax + ay * ay + az * az))
        rates = [rollspeed[i], pitchspeed[i], yawspeed[i], rollspeed[i - 1], pitchspeed[i - 1], yawspeed[i - 1]]
        if not any(v is None for v in rates):
            dr = (rollspeed[i] - rollspeed[i - 1]) / dt
            dp = (pitchspeed[i] - pitchspeed[i - 1]) / dt
            dy = (yawspeed[i] - yawspeed[i - 1]) / dt
            attitude_rate_delta.append(math.sqrt(dr * dr + dp * dp + dy * dy))
    jerk = []
    for i in range(1, len(accel)):
        if i >= len(times):
            break
        dt = times[i] - times[i - 1]
        if dt <= 1e-6:
            continue
        jerk.append((accel[i] - accel[i - 1]) / dt)
    metrics["accel_rms_mps2"] = rms(accel)
    metrics["jerk_rms_mps3"] = rms(jerk)
    metrics["attitude_jerk_rms_rad_s2"] = rms(attitude_rate_delta)
    return metrics


def summarize_pid_piper_csvs(run_dir):
    metrics = {}
    csv_paths = sorted(glob.glob(os.path.join(run_dir, "attack_*.csv")))
    fixed = os.path.join(run_dir, "Data_Piper_Training_Wide.csv")
    if os.path.isfile(fixed):
        csv_paths.append(fixed)
    if not csv_paths:
        metrics["pid_piper_csv_rows"] = 0
        return metrics

    residual_abs = []
    alpha = []
    fused_minus_selected = []
    axis_values = defaultdict(lambda: {"residual_abs": [], "alpha": []})
    rows = 0
    for path in csv_paths:
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows += 1
                    axis = (row.get("angle_type") or "").strip().lower()
                    res = safe_float(row.get("residual"))
                    a = safe_float(row.get("alpha"))
                    y_fused = safe_float(row.get("y_fused"))
                    y_selected = safe_float(row.get("y_selected"))
                    if res is not None:
                        residual_abs.append(abs(res))
                        if axis:
                            axis_values[axis]["residual_abs"].append(abs(res))
                    if a is not None:
                        alpha.append(a)
                        if axis:
                            axis_values[axis]["alpha"].append(a)
                    if y_fused is not None and y_selected is not None:
                        fused_minus_selected.append(y_fused - y_selected)
        except Exception:
            continue

    metrics["pid_piper_csv_rows"] = rows
    metrics["pid_piper_csv_files"] = len(csv_paths)
    metrics["residual_abs_mean"] = mean(residual_abs)
    metrics["residual_abs_max"] = max(residual_abs) if residual_abs else None
    metrics["alpha_mean"] = mean(alpha)
    metrics["alpha_std"] = std(alpha)
    metrics["alpha_clip_low_fraction"] = mean([1.0 if v <= 0.02 else 0.0 for v in alpha])
    metrics["alpha_clip_high_fraction"] = mean([1.0 if v >= 0.98 else 0.0 for v in alpha])
    metrics["fused_minus_selected_rms"] = rms(fused_minus_selected)
    for axis, values in axis_values.items():
        prefix = "axis_%s_" % axis
        metrics[prefix + "residual_abs_mean"] = mean(values["residual_abs"])
        metrics[prefix + "alpha_mean"] = mean(values["alpha"])
        metrics[prefix + "alpha_std"] = std(values["alpha"])
    return metrics


def analyze_bin_log(bin_path):
    metrics = {"bin_parse_ok": False}
    if not bin_path or not os.path.isfile(bin_path):
        return metrics
    if mavutil is None:
        metrics["bin_parse_error"] = "pymavlink not available"
        return metrics
    try:
        log = mavutil.mavlink_connection(bin_path)
        counts = Counter()
        first_t = None
        last_t = None
        att_roll = []
        att_pitch = []
        att_yaw = []
        while True:
            msg = log.recv_match(blocking=False)
            if msg is None:
                break
            typ = msg.get_type()
            counts[typ] += 1
            d = msg.to_dict()
            t = d.get("TimeUS")
            if t is not None:
                t = float(t) / 1.0e6
            else:
                t = d.get("TimeMS")
                if t is not None:
                    t = float(t) / 1000.0
            if t is not None:
                if first_t is None:
                    first_t = t
                last_t = t
            if typ == "ATT":
                for field, bucket in (("Roll", att_roll), ("Pitch", att_pitch), ("Yaw", att_yaw)):
                    value = safe_float(d.get(field))
                    if value is not None:
                        bucket.append(math.radians(value))
        metrics["bin_parse_ok"] = True
        metrics["bin_message_count"] = sum(counts.values())
        metrics["bin_duration_s"] = (last_t - first_t) if first_t is not None and last_t is not None else None
        metrics["bin_top_message_types"] = ",".join("%s:%d" % (k, v) for k, v in counts.most_common(12))
        metrics["bin_att_roll_rms_rad"] = rms(att_roll)
        metrics["bin_att_pitch_rms_rad"] = rms(att_pitch)
        metrics["bin_att_yaw_rms_rad"] = rms(att_yaw)
    except Exception as exc:
        metrics["bin_parse_error"] = repr(exc)
    return metrics


def apply_strict_success_from_analysis(analysis, result):
    radius = safe_float(result.get("success_radius_m"), DEFAULT_SUCCESS_RADIUS_M)
    land_alt = safe_float(result.get("success_land_alt_m"), DEFAULT_LAND_ALT_M)
    auto_confirmed = bool(result.get("auto_confirmed"))
    if not auto_confirmed:
        auto_confirmed = bool(analysis.get("auto_mode_seen"))
    if not auto_confirmed:
        auto_confirmed = result.get("auto_sent_wall_time") is not None or result.get("time_to_auto_s") is not None
    final_alt = safe_float(result.get("success_relative_alt_m"))
    if final_alt is None:
        final_alt = safe_float(analysis.get("final_relative_alt_m"))
    final_dist = safe_float(result.get("success_final_dist_to_target_m"))
    if final_dist is None:
        final_dist = safe_float(analysis.get("final_dist_to_target_m"))
    low_alt = final_alt is not None and final_alt <= land_alt
    near_target = final_dist is not None and final_dist <= radius
    strict_success = auto_confirmed and low_alt and near_target
    analysis["auto_confirmed"] = auto_confirmed
    analysis["success_relative_alt_m"] = final_alt
    analysis["success_final_dist_to_target_m"] = final_dist
    analysis["success_land_alt_m"] = land_alt
    analysis["success_radius_m"] = radius
    analysis["success_low_alt"] = low_alt
    analysis["success_near_target"] = near_target
    analysis["success"] = bool(strict_success)
    status = analysis.get("status") or result.get("status")
    if status:
        while status.endswith(OUTSIDE_SUCCESS_GATE_SUFFIX):
            status = status[: -len(OUTSIDE_SUCCESS_GATE_SUFFIX)]
        if strict_success:
            analysis["status"] = status
        elif status.startswith("landed_"):
            analysis["status"] = "%s%s" % (status, OUTSIDE_SUCCESS_GATE_SUFFIX)


def refresh_strict_success_for_run(run_dir, analysis):
    artifact_dir = analysis.get("artifact_dir") or analysis.get("attempt_dir")
    if not artifact_dir or not os.path.isdir(artifact_dir):
        artifact_dir = find_latest_completed_attempt_dir(run_dir) or run_dir
    result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.isfile(result_path):
        result_path = os.path.join(artifact_dir, "run_result.json")
    result = {}
    if os.path.isfile(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)
    telemetry_metrics = read_telemetry_metrics(os.path.join(artifact_dir, "telemetry.csv"), analysis.get("mission") or "")
    analysis.update(telemetry_metrics)
    apply_strict_success_from_analysis(analysis, result)
    return analysis


def flatten_for_csv(payload):
    out = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, sort_keys=True)
        else:
            out[key] = value
    return out


def analyze_run(run_dir):
    meta_path = os.path.join(run_dir, "run_meta.json")
    result_path = os.path.join(run_dir, "run_result.json")
    meta = {}
    result = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    if os.path.isfile(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)

    analysis = {}
    for key in ["run_index", "run_id", "mission", "mode", "gps_scn", "imu_scn", "gps_seed", "imu_seed"]:
        analysis[key] = meta.get(key)
    analysis["status"] = result.get("status")
    analysis["success"] = bool(result.get("success", False))
    analysis["duration_s"] = result.get("duration_s")
    analysis["time_to_auto_s"] = result.get("time_to_auto_s")
    analysis["time_to_land_s"] = result.get("time_to_land_s")
    for key in [
        "auto_confirmed",
        "success_relative_alt_m",
        "success_final_dist_to_target_m",
        "success_groundspeed_mps",
        "success_land_alt_m",
        "success_radius_m",
        "success_low_alt",
        "success_near_target",
        "post_takeoff_params_applied",
        "post_takeoff_param_failed",
        "post_auto_params_applied",
        "post_auto_param_failed",
    ]:
        if key in result:
            analysis[key] = result.get(key)
    if result.get("exception"):
        analysis["exception"] = result.get("exception")

    telemetry_metrics = read_telemetry_metrics(os.path.join(run_dir, "telemetry.csv"), analysis.get("mission") or "")
    analysis.update(telemetry_metrics)
    apply_strict_success_from_analysis(analysis, result)
    analysis.update(summarize_pid_piper_csvs(run_dir))
    analysis.update(analyze_bin_log(os.path.join(run_dir, "flight.BIN")))
    return analysis


def key_for_group(row, keys):
    return tuple(row.get(k) for k in keys)


def aggregate_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[key_for_group(row, keys)].append(row)
    out = []
    numeric_metric_names = [
        "success",
        "duration_s",
        "time_to_auto_s",
        "time_to_land_s",
        "success_final_dist_to_target_m",
        "success_relative_alt_m",
        "mean_xtrack_error_m",
        "max_xtrack_error_m",
        "final_dist_to_target_m",
        "min_final_dist_to_target_m",
        "mean_groundspeed_mps",
        "max_groundspeed_mps",
        "accel_rms_mps2",
        "jerk_rms_mps3",
        "attitude_rate_rms_rad_s",
        "attitude_jerk_rms_rad_s2",
        "residual_abs_mean",
        "alpha_mean",
        "alpha_std",
        "alpha_clip_low_fraction",
        "alpha_clip_high_fraction",
    ]
    for key_values, items in sorted(groups.items()):
        row = dict(zip(keys, key_values))
        row["runs"] = len(items)
        for name in numeric_metric_names:
            values = []
            for item in items:
                if name == "success":
                    values.append(1.0 if item.get("success") else 0.0)
                else:
                    val = safe_float(item.get(name))
                    if val is not None:
                        values.append(val)
            if not values:
                continue
            suffix = "rate" if name == "success" else "mean"
            row["%s_%s" % (name, suffix)] = mean(values)
            if name != "success":
                row["%s_std" % name] = std(values)
        out.append(row)
    return out


def compare_mode_rows(rows):
    group_keys = ["mission", "gps_scn", "imu_scn"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_for_group(row, group_keys)].append(row)

    metrics = [
        "success_rate",
        "duration_s_mean",
        "time_to_land_s_mean",
        "success_final_dist_to_target_m_mean",
        "success_relative_alt_m_mean",
        "mean_xtrack_error_m_mean",
        "max_xtrack_error_m_mean",
        "final_dist_to_target_m_mean",
        "jerk_rms_mps3_mean",
        "attitude_jerk_rms_rad_s2_mean",
        "residual_abs_mean_mean",
    ]
    comparisons = []
    for key_values, items in sorted(grouped.items()):
        by_mode = {}
        for item in items:
            by_mode[int(item.get("mode"))] = item
        row = dict(zip(group_keys, key_values))
        row["modes_present"] = ",".join(str(m) for m in sorted(by_mode))
        for metric in metrics:
            for mode in sorted(by_mode):
                row["mode%d_%s" % (mode, metric)] = safe_float(by_mode[mode].get(metric))
            if COMPARISON_TARGET in by_mode:
                target_value = safe_float(by_mode[COMPARISON_TARGET].get(metric))
                for baseline in COMPARISON_BASELINES:
                    if baseline not in by_mode:
                        continue
                    baseline_value = safe_float(by_mode[baseline].get(metric))
                    if baseline_value is not None and target_value is not None:
                        row[
                            "delta_mode%d_minus_mode%d_%s"
                            % (COMPARISON_TARGET, baseline, metric)
                        ] = target_value - baseline_value
            if 0 in by_mode and 2 in by_mode:
                v0 = safe_float(by_mode[0].get(metric))
                v2 = safe_float(by_mode[2].get(metric))
                if v0 is not None and v2 is not None:
                    row["delta_mode2_minus_mode0_%s" % metric] = v2 - v0
        comparisons.append(row)
    return comparisons


def fieldnames_for_rows(rows):
    keys = []
    seen = set()
    preferred = [
        "run_index",
        "run_id",
        "mission",
        "mode",
        "gps_scn",
        "imu_scn",
        "gps_seed",
        "imu_seed",
        "status",
        "success",
        "auto_confirmed",
        "success_low_alt",
        "success_near_target",
        "success_final_dist_to_target_m",
        "success_relative_alt_m",
        "post_takeoff_params_applied",
        "post_auto_params_applied",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in sorted(row.keys()):
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def write_analysis_report(path, run_rows, by_condition, comparisons):
    total = len(run_rows)
    success_by_mode = defaultdict(list)
    for row in run_rows:
        success_by_mode[row.get("mode")].append(1.0 if row.get("success") else 0.0)

    lines = []
    lines.append("# PID-Piper Mode Experiment Report")
    lines.append("")
    lines.append("- Generated: %s" % datetime.now().isoformat())
    lines.append("- Runs analyzed: %d" % total)
    for mode in sorted(success_by_mode, key=lambda x: str(x)):
        try:
            mode_i = int(mode)
        except Exception:
            mode_i = mode
        label = MODE_LABELS.get(mode_i, "mode_%s" % mode)
        lines.append("- Mode %s (%s) success rate: %.3f" % (mode, label, mean(success_by_mode[mode]) or 0.0))
    lines.append("")
    for baseline in COMPARISON_BASELINES:
        delta_key = "delta_mode%d_minus_mode%d_success_rate" % (COMPARISON_TARGET, baseline)
        target_label = MODE_LABELS.get(COMPARISON_TARGET, "mode_%d" % COMPARISON_TARGET)
        baseline_label = MODE_LABELS.get(baseline, "mode_%d" % baseline)
        scored = []
        for row in comparisons:
            delta = safe_float(row.get(delta_key))
            if delta is not None:
                scored.append((delta, row))
        if not scored:
            continue
        lines.append("## Largest %s Success Improvements vs %s" % (target_label, baseline_label))
        for delta, row in sorted(scored, key=lambda x: x[0], reverse=True)[:10]:
            lines.append(
                "- %s gps=%s imu=%s: success delta %.3f"
                % (row.get("mission"), row.get("gps_scn"), row.get("imu_scn"), delta)
            )
        lines.append("")
        lines.append("## Largest %s Success Regressions vs %s" % (target_label, baseline_label))
        for delta, row in sorted(scored, key=lambda x: x[0])[:10]:
            lines.append(
                "- %s gps=%s imu=%s: success delta %.3f"
                % (row.get("mission"), row.get("gps_scn"), row.get("imu_scn"), delta)
            )
        lines.append("")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Success means AUTO was confirmed, the vehicle landed within the AUTO timeout, altitude was below the landing threshold, and final distance to the mission target was within the configured success radius.")
    lines.append("- Path error metrics are computed against the mission home-to-target line.")
    lines.append("- PID-Piper metrics are read from attack_*.csv in the completed attempt directory generated by PID_PIPER_OUTPUT_DIR.")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def aggregate_analysis(output_dir, force_reanalyze=False, refresh_success=False):
    run_rows = []
    meta_paths = sorted(glob.glob(os.path.join(output_dir, "*", "run_meta.json")))
    total = len(meta_paths)
    if total:
        mode_desc = "force-reanalyze" if force_reanalyze else ("refresh-success" if refresh_success else "cached")
        progress("[INFO] Analysis mode=%s runs=%d" % (mode_desc, total))
    for index, meta_path in enumerate(meta_paths, 1):
        run_dir = os.path.dirname(meta_path)
        run_name = os.path.basename(run_dir)
        analysis_path = os.path.join(run_dir, "analysis.json")
        if os.path.isfile(analysis_path) and not force_reanalyze:
            with open(analysis_path, "r") as f:
                analysis = json.load(f)
            if refresh_success:
                progress("[ANALYZE] %04d/%04d refresh-success %s" % (index, total, run_name))
                analysis = refresh_strict_success_for_run(run_dir, analysis)
                write_json(analysis_path, analysis)
            elif index == 1 or index == total or (index % 50) == 0:
                progress("[ANALYZE] %04d/%04d cached %s" % (index, total, run_name))
        else:
            attempt_dir = find_latest_completed_attempt_dir(run_dir)
            if attempt_dir is None:
                print("[WARN] Skipping incomplete run without completed attempt: %s" % run_dir)
                continue
            progress("[ANALYZE] %04d/%04d force-reanalyze %s" % (index, total, run_name))
            analysis = analyze_run(attempt_dir)
            analysis["attempt_complete"] = True
            analysis["attempt_id"] = os.path.basename(attempt_dir)
            analysis["attempt_dir"] = attempt_dir
            analysis["artifact_dir"] = attempt_dir
            write_json(analysis_path, analysis)
        run_rows.append(flatten_for_csv(analysis))

    if not run_rows:
        print("[WARN] No runs found under %s" % output_dir)
        return []

    write_csv(os.path.join(output_dir, "runs.csv"), run_rows, fieldnames_for_rows(run_rows))
    by_condition = aggregate_rows(run_rows, ["mission", "gps_scn", "imu_scn", "mode"])
    write_csv(os.path.join(output_dir, "summary_by_condition.csv"), by_condition, fieldnames_for_rows(by_condition))
    comparisons = compare_mode_rows(by_condition)
    write_csv(os.path.join(output_dir, "comparison_by_mode.csv"), comparisons, fieldnames_for_rows(comparisons) if comparisons else [])
    write_analysis_report(os.path.join(output_dir, "analysis_report.md"), run_rows, by_condition, comparisons)
    print("[OK] Wrote aggregate analysis to %s" % output_dir)
    return run_rows


def should_skip_run(args, meta):
    if not args.resume:
        return False
    analysis_path = os.path.join(meta["run_dir"], "analysis.json")
    if not os.path.isfile(analysis_path):
        return False
    try:
        with open(analysis_path, "r") as f:
            payload = json.load(f)
        if payload.get("status") in RETRYABLE_LAUNCH_STATUSES:
            return False
        return bool(payload.get("status")) and bool(payload.get("attempt_complete", False))
    except Exception:
        return False


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="0,2,4", help="Comma-separated ATC_PIPER_MODE values.")
    parser.add_argument("--missions", default="mission-1.txt,mission-2.txt", help="Comma-separated mission files under Tools/autotest/mission.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated deterministic attack seeds.")
    parser.add_argument(
        "--attack-grid",
        default="layered",
        choices=["none", "base", "layered", "stress", "extended"],
        help="Attack grid: none=(0,0), base=0..3 grid, layered=base plus selected 4/5 cases, stress=selected 6/7 stress cases, extended=0..7 grid.",
    )
    parser.add_argument("--attack-combos", default="", help="Explicit GPS:IMU combos, e.g. 0:0,4:0,0:5.")
    parser.add_argument("--output-dir", default="", help="Experiment output directory.")
    parser.add_argument(
        "--append-missing",
        action="store_true",
        help=(
            "Append only conditions not already present in output-dir by inspecting existing run_meta.json files. "
            "Useful for adding PID baseline runs to an existing mode2/mode4 output directory."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip runs that already have a completed root analysis.json; interrupted attempts are rerun in a new attempt_* directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the manifest and print planned run count.")
    parser.add_argument("--analyze-only", default="", help="Analyze an existing output directory without launching SITL.")
    parser.add_argument("--force-reanalyze", action="store_true", help="When used with --analyze-only, recompute each run's analysis.json instead of reusing cached analysis files.")
    parser.add_argument("--refresh-success", action="store_true", help="When used with --analyze-only, reuse cached metrics but recompute strict success/status from telemetry.")
    parser.add_argument("--skip-build", action="store_true", help="Do not run ./waf copter before experiments.")
    parser.add_argument("--sim-rebuild", action="store_true", help="Allow sim_vehicle to rebuild instead of passing --no-rebuild.")
    parser.add_argument("--sim-vehicle-py", default=os.environ.get("SIMVEHICLE_PY", "python2.7"), help="Python executable used to run sim_vehicle.py.")
    parser.add_argument("--mavlink", default=DEFAULT_MAVLINK_ADDR, help="MAVLink connection string.")
    parser.add_argument("--aircraft", default="pid_piper_mode_experiment", help="sim_vehicle aircraft name.")
    parser.add_argument("--show-map", action="store_true", help="Pass --map to sim_vehicle. Disabled by default to avoid leftover map windows.")
    parser.add_argument("--show-console", action="store_true", help="Pass --console to sim_vehicle. Disabled by default.")
    parser.add_argument("--force-cleanup", action="store_true", help="After each run, pkill known simulator helper processes on POSIX.")
    parser.add_argument("--allow-windows-live", action="store_true", help="Allow live SITL launch from Windows. Normally run live experiments inside WSL/Linux.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the number of planned runs for smoke testing.")
    parser.add_argument("--heartbeat-timeout", type=float, default=30.0)
    parser.add_argument("--start-settle", type=float, default=2.0)
    parser.add_argument("--post-heartbeat-settle", type=float, default=10.0)
    parser.add_argument("--command-interval", type=float, default=2.0)
    parser.add_argument("--prearm-timeout", type=float, default=60.0, help="Maximum seconds to wait for GPS/prearm readiness before arming.")
    parser.add_argument("--arm-timeout", type=float, default=60.0, help="Maximum seconds to retry arming before treating the launch as incomplete.")
    parser.add_argument("--takeoff-timeout", type=float, default=120.0)
    parser.add_argument("--auto-mode-timeout", type=float, default=15.0, help="Maximum seconds to wait for HEARTBEAT-confirmed AUTO mode after sending mode auto.")
    parser.add_argument("--timeout-sec", type=float, default=300.0, help="AUTO-phase timeout in seconds.")
    parser.add_argument("--takeoff-alt", type=float, default=50.0)
    parser.add_argument("--takeoff-sustain", type=float, default=2.0)
    parser.add_argument("--land-alt", type=float, default=3.0)
    parser.add_argument("--land-speed", type=float, default=1.0)
    parser.add_argument("--land-sustain", type=float, default=5.0)
    parser.add_argument("--success-radius-m", type=float, default=10.0, help="Run succeeds only if final distance to the mission target is within this radius after AUTO and landing.")
    parser.add_argument(
        "--disable-failsafes-after-takeoff",
        action="store_true",
        help="After takeoff altitude is sustained but before AUTO, set FS_EKF_THRESH=0 and FS_CRASH_CHECK=0 for no-failsafe stress testing.",
    )
    parser.add_argument(
        "--keep-failsafes-in-stress",
        action="store_true",
        help="Do not auto-enable --disable-failsafes-after-takeoff for --attack-grid stress.",
    )
    parser.add_argument(
        "--allow-stress-mode-subset",
        action="store_true",
        help="For --attack-grid stress, keep the user-provided --modes instead of forcing inclusion of modes 0,2,4.",
    )
    parser.add_argument(
        "--defer-attack-until-after-takeoff",
        action="store_true",
        help="Launch with SIM_GPS_ATK_SCN=0 and SIM_IMU_ATK_SCN=0, then apply the run's attack scenarios after takeoff altitude is sustained and before AUTO.",
    )
    parser.add_argument(
        "--keep-attack-during-launch",
        action="store_true",
        help="For --attack-grid stress, keep the planned attack scenarios active during prearm/arming/takeoff instead of deferring them.",
    )
    parser.add_argument(
        "--isolated-sitl-state",
        action="store_true",
        help="Use a per-attempt SITL state directory with a fresh EEPROM and launch parameter overlay.",
    )
    parser.add_argument(
        "--shared-sitl-state",
        action="store_true",
        help="For --attack-grid stress, reuse the normal ArduCopter SITL state directory instead of isolating per attempt.",
    )
    parser.add_argument("--post-takeoff-param", default="", help="Comma-separated NAME=VALUE params to set after takeoff altitude is sustained and before AUTO.")
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--extra-sim-arg", action="append", default=[], help="Extra argument passed to sim_vehicle.py; repeat as needed.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    configure_stress_defaults(args)

    if args.analyze_only:
        aggregate_analysis(os.path.abspath(args.analyze_only), force_reanalyze=args.force_reanalyze, refresh_success=args.refresh_success)
        return 0

    if not args.output_dir:
        args.output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, "mode_experiment_%s" % now_stamp())
    args.output_dir = os.path.abspath(args.output_dir)
    ensure_dir(args.output_dir)

    if args.append_missing:
        manifest, full_manifest, completed_count, existing_count = build_append_missing_manifest(args)
        print("[INFO] Append-missing scan: planned_conditions=%d existing_conditions=%d completed_conditions=%d missing_or_incomplete_conditions=%d" % (
            len(full_manifest),
            existing_count,
            completed_count,
            len(manifest),
        ))
    else:
        manifest = build_manifest(args)
        full_manifest = manifest
    manifest_path = os.path.join(args.output_dir, "experiment_manifest.csv")
    write_csv(
        manifest_path,
        full_manifest,
        ["run_index", "run_id", "mission", "mode", "gps_scn", "imu_scn", "gps_seed", "imu_seed", "run_dir"],
    )
    print("[OK] Wrote manifest: %s" % manifest_path)
    if args.append_missing:
        append_manifest_path = os.path.join(args.output_dir, "experiment_manifest_append_missing.csv")
        write_csv(
            append_manifest_path,
            manifest,
            ["run_index", "run_id", "mission", "mode", "gps_scn", "imu_scn", "gps_seed", "imu_seed", "run_dir"],
        )
        print("[OK] Wrote append-missing manifest: %s" % append_manifest_path)
    print("[INFO] Planned runs: %d" % len(manifest))

    if args.dry_run:
        for row in manifest[:10]:
            print("[PLAN] {run_id}: mission={mission} mode={mode} gps={gps_scn} imu={imu_scn} seed={gps_seed}".format(**row))
        if len(manifest) > 10:
            print("[PLAN] ... %d more" % (len(manifest) - 10))
        return 0

    if os.name == "nt" and not args.allow_windows_live:
        eprint("[ERROR] Live SITL experiments should be launched inside WSL/Linux for this workspace.")
        eprint("        Example: wsl.exe -d Ubuntu-18.04 bash -lc \"cd /home/jiran/work/pid-piper && python3 simulator/Tools/autotest/run_pid_piper_mode_experiments.py\"")
        eprint("        Use --dry-run or --analyze-only from Windows, or pass --allow-windows-live if you have a Windows-native SITL setup.")
        return 2

    run_build(args)
    if args.force_cleanup:
        force_cleanup_leftovers()

    for meta in manifest:
        if should_skip_run(args, meta):
            print("[SKIP] %s" % meta["run_id"])
            continue
        try:
            analysis = run_one(args, meta)
            print("[DONE] %s status=%s success=%s" % (meta["run_id"], analysis.get("status"), analysis.get("success")))
        except KeyboardInterrupt:
            print("[INFO] Interrupted; aggregating completed runs before exit")
            break

    aggregate_analysis(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
