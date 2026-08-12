#!/usr/bin/env python3
# tools/run_bench.py

import asyncio
import os
import signal
import time
import shutil
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml


# ============================ USER CONFIG =====================================

# Assumption: already ran 
#   python scripts/kp_planner_benchmark.py --bench isaac
# and have problems/*.yaml ready.

PROBLEMS_DIR = Path("problems")

# Cases (groups) we will evaluate
CASE_SPECS = [
    # full (single file each)
    #{"name": "full_3x3",  "glob": "full_3x3.yaml"},
    {"name": "full_5x5",  "glob": "full_5x5.yaml"},

    # sub (5 files each)
    #{"name": "sub_3x3_k5",  "glob": "sub_3x3_k5_*.yaml"},
    #{"name": "sub_5x5_k16", "glob": "sub_5x5_k16_*.yaml"},
]

# How many times to run each problem yaml
N_RUNS_PER_PROBLEM = 10

# Isaac Sim ► scene publisher (never exits; we watch the log)
# You MUST set these paths for your machine:
ISAACSIM_PATH              = "/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release"
ISAAC_ROS2_BRIDGE_LIB_DIR  = f"{ISAACSIM_PATH}/exts/isaacsim.ros2.bridge/humble/lib/"
ISAAC_MOVEIT_WS_SETUP      = "/home/luhao/Workspace/planning/isaacsim-ros-ws/build_ws/humble/isaac_sim_moveit_ws/install/setup.bash"

# MoveIt bringup used with Isaac Sim (never exits; we watch the log)
MOVEIT_WS_SETUP            = "/home/luhao/Workspace/manipulation/overlay_ws/install/setup.bash"
LOCAL_SETUP_BASH           = str(Path("install") / "local_setup.bash")

# Executor launch (fed with problem.yaml)
EXECUTOR_LAUNCH_CMDLINE    = 'ros2 launch blocksworld_executor smash_demo.launch.py problem:="{problem}"'

# Timeouts / retries
ISAAC_SCENE_READY_TIMEOUT_SEC   = 180
MOVEIT_READY_TIMEOUT_SEC        = 180
SLEEP_AFTER_MOVEIT_READY_SEC    = 0.3

EXEC_TIMEOUT_SEC                = 600   # 10 minutes
EXEC_MAX_ATTEMPTS               = 3
EXEC_RETRY_BACKOFF_SEC          = 2.0

# Output layout
RUNS_DIR_NAME = "runs"

# =============================================================================


# Workspace & output layout
WS = Path(os.getcwd())
RUNS_DIR = WS / RUNS_DIR_NAME
RUNS_DIR.mkdir(exist_ok=True)

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def env_base():
    return os.environ.copy()


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


# ---------------------------- PROCESS HELPERS ---------------------------------

async def spawn_exec(cmd_list, cwd=None, env=None, logfile=None, new_session=True):
    lf = open(logfile, "wb") if logfile else asyncio.subprocess.DEVNULL
    p = await asyncio.create_subprocess_exec(
        *cmd_list,
        cwd=cwd, env=env,
        stdout=lf, stderr=lf,
        start_new_session=new_session,
    )
    p._lf = lf if logfile else None
    return p


async def spawn_shell(cmdline, cwd=None, env=None, logfile=None, new_session=True):
    lf = open(logfile, "wb") if logfile else asyncio.subprocess.DEVNULL
    p = await asyncio.create_subprocess_exec(
        "/bin/bash", "-lc", cmdline,
        cwd=cwd, env=env,
        stdout=lf, stderr=lf,
        start_new_session=new_session,
    )
    p._lf = lf if logfile else None
    return p


async def terminate(proc, name, timeout=8):
    """SIGINT -> wait -> SIGTERM -> wait -> SIGKILL."""
    if proc is None or proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout)
        return
    except asyncio.TimeoutError:
        pass

    try:
        os.killpg(pgid, signal.SIGTERM)
        await asyncio.wait_for(proc.wait(), 3)
        return
    except asyncio.TimeoutError:
        pass

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def nowstamp():
    return time.strftime("%Y%m%d-%H%M%S")


# ------------------------------ LOG WATCHERS ----------------------------------

async def wait_log_regex(log_path: Path, pattern: str, timeout: float):
    t0 = time.time()
    cre = re.compile(pattern)
    while time.time() - t0 < timeout:
        if log_path.exists():
            try:
                data = log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                data = ""
            data = strip_ansi(data)
            m = cre.search(data)
            if m:
                return m
        await asyncio.sleep(0.25)
    return None


async def wait_log_all_substrings(log_path: Path, substrings, timeout: float):
    t0 = time.time()
    seen = set()
    subs = list(substrings)
    while time.time() - t0 < timeout:
        if log_path.exists():
            try:
                data = log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                data = ""
            data = strip_ansi(data)
            for s in subs:
                if s in data:
                    seen.add(s)
            if len(seen) == len(subs):
                return True
        await asyncio.sleep(0.25)
    return False


# ------------------------------ YAML PARSERS ----------------------------------

def parse_problem_actions(problem_path: Path) -> Dict[str, Any]:
    """
    Extract:
      - num_actions: number of entries under 'actions'
      - num_smashes: count of entries with type == 'Smash' (case-insensitive)
    """
    data = yaml.safe_load(problem_path.read_text(encoding="utf-8", errors="ignore")) or {}
    actions = data.get("actions") or []
    num_actions = 0
    num_smashes = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        num_actions += 1
        t = str(a.get("type", "")).strip().lower()
        if t == "smash":
            num_smashes += 1
    return {
        "num_actions": num_actions,
        "num_smashes": num_smashes,
    }


def parse_case_seed_from_filename(p: Path) -> Optional[int]:
    """
    For files like sub_3x3_k5_greedy_4.yaml -> return 4
    Else None.
    """
    m = re.search(r"_([0-9]+)\.ya?ml$", p.name)
    return int(m.group(1)) if m else None


# ------------------------------ EXECUTOR PARSERS ------------------------------

def parse_executor_success_metrics(log_path: Path) -> Optional[Dict[str, Any]]:
    """
    Success is defined by presence of BOTH:
      1) All Tasks Execution Successfully! (knock retry:K)
      2) task and motion planning time:T s,execution time:E s

    Returns dict with:
      - knock_retry
      - tamp_time_sec
      - execution_time_sec
      - total_time_sec = tamp + execution
    or None if missing.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None

    text = strip_ansi(text)

    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    # success line
    m1 = re.findall(
        rf"All\s+Tasks\s+Execution\s+Successfully!\s*\(\s*knock\s+retry\s*:\s*(\d+)\s*\)",
        text,
        flags=re.IGNORECASE
    )
    # timing line
    m2 = re.findall(
        rf"task\s+and\s+motion\s+planning\s+time\s*:\s*({num})\s*s\s*,\s*execution\s+time\s*:\s*({num})\s*s",
        text,
        flags=re.IGNORECASE
    )

    if not m1 or not m2:
        return None

    knock_retry = int(m1[-1])
    tamp = float(m2[-1][0])
    exec_t = float(m2[-1][1])

    return {
        "knock_retry": knock_retry,
        "tamp_time_sec": tamp,
        "execution_time_sec": exec_t,
        "total_time_sec": tamp + exec_t,
    }


# ------------------------------ AGG HELPERS -----------------------------------

def _avg_present(runs, key):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def _sum_present(runs, key):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    return sum(vals) if vals else None


def _count_true(runs, key):
    return sum(1 for r in runs if r.get(key) is True)


# ------------------------------ ISAAC PIPELINE --------------------------------

async def run_one_problem_once(
    problem_path: Path,
    run_dir: Path,
    env: Dict[str, str],
    label: str,
) -> Dict[str, Any]:
    """
    Run Isaac scene -> MoveIt bringup -> Executor (with retries).
    Return a dict for this single run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # copy problem file into run dir for record
    local_problem = run_dir / problem_path.name
    shutil.copy2(problem_path, local_problem)

    prob_stats = parse_problem_actions(problem_path)
    seed = parse_case_seed_from_filename(problem_path)

    # ---------------- (A) executor attempts loop ----------------
    attempts_used = 0
    exec_run_retries = 0
    success = False
    fail_reason = None
    last_exec_log = None

    # metrics extracted from log on success
    log_metrics = None
    exec_wall_time_sec = None

    for attempt in range(1, EXEC_MAX_ATTEMPTS + 1):
        attempts_used = attempt

        # ---------------- 1) ISAAC SCENE PUBLISHER
        isaac_scene_log = run_dir / f"isaac_scene_attempt{attempt}.log"
        isaac_scene_cmd = rf'''
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:{ISAAC_ROS2_BRIDGE_LIB_DIR}"
export ISAACSIM_PATH="{ISAACSIM_PATH}"
export ISAACSIM_PYTHON_EXE="${{ISAACSIM_PATH}}/python.sh"
source "{ISAAC_MOVEIT_WS_SETUP}"
"$ISAACSIM_PYTHON_EXE" src/isaac_moveit/scripts/scene_isaacsim.py --yaml "{local_problem}" --headless
'''
        isaac_scene = await spawn_shell(isaac_scene_cmd, env=env, logfile=str(isaac_scene_log))

        m = await wait_log_regex(
            isaac_scene_log,
            r"\[scene\]\s+Published\s+(\d+)\s+objects\s+to\s+/planning_scene",
            timeout=ISAAC_SCENE_READY_TIMEOUT_SEC,
        )
        if not m:
            await terminate(isaac_scene, "isaac_scene")
            fail_reason = "scene_ready_timeout"
            print(f"[{label}] attempt {attempt}: scene READY TIMEOUT")
            exec_run_retries += 1
            await asyncio.sleep(EXEC_RETRY_BACKOFF_SEC)
            continue

        # ---------------- 2) ISAAC MOVEIT BRINGUP
        isaac_moveit_log = run_dir / f"isaac_moveit_attempt{attempt}.log"
        isaac_moveit_cmd = rf'''
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source "{MOVEIT_WS_SETUP}"
source "{LOCAL_SETUP_BASH}"
ros2 launch isaac_moveit isaac_moveit.launch.py use_rviz:=false
'''
        isaac_moveit = await spawn_shell(isaac_moveit_cmd, env=env, logfile=str(isaac_moveit_log))

        ok = await wait_log_all_substrings(
            isaac_moveit_log,
            [
                "Configured and activated panda_arm_controller",
                "Configured and activated joint_state_broadcaster",
                "Configured and activated panda_hand_controller",
            ],
            timeout=MOVEIT_READY_TIMEOUT_SEC,
        )
        if not ok:
            await terminate(isaac_moveit, "isaac_moveit")
            await terminate(isaac_scene, "isaac_scene")
            fail_reason = "moveit_ready_timeout"
            print(f"[{label}] attempt {attempt}: moveit READY TIMEOUT")
            exec_run_retries += 1
            await asyncio.sleep(EXEC_RETRY_BACKOFF_SEC)
            continue

        await asyncio.sleep(SLEEP_AFTER_MOVEIT_READY_SEC)

        # ---------------- 3) EXECUTOR
        last_exec_log = run_dir / f"executor_attempt{attempt}.log"
        executor_cmd = rf'''
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source "{MOVEIT_WS_SETUP}"
source "{LOCAL_SETUP_BASH}"
{EXECUTOR_LAUNCH_CMDLINE.format(problem=str(local_problem))}
'''
        t0 = time.perf_counter()
        exec_proc = await spawn_shell(executor_cmd, env=env, logfile=str(last_exec_log))

        timed_out = False
        try:
            await asyncio.wait_for(exec_proc.wait(), timeout=EXEC_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            timed_out = True
            await terminate(exec_proc, "executor")

        t1 = time.perf_counter()
        exec_wall_time_sec = t1 - t0

        # regardless of return code, check log for required success lines
        log_metrics = parse_executor_success_metrics(last_exec_log)

        # cleanup long-runners
        await terminate(isaac_moveit, "isaac_moveit")
        await terminate(isaac_scene, "isaac_scene")

        if log_metrics is not None:
            success = True
            fail_reason = None
            break

        # else retry if we can
        fail_reason = "exec_timeout" if timed_out else "exec_missing_success_lines"
        print(f"[{label}] attempt {attempt}: executor FAIL ({fail_reason}); wall={exec_wall_time_sec:.2f}s")

        if attempt < EXEC_MAX_ATTEMPTS:
            exec_run_retries += 1
            await asyncio.sleep(EXEC_RETRY_BACKOFF_SEC)

    stat = {
        "problem": str(problem_path),
        "problem_copy": str(local_problem),
        "seed": seed,
        **prob_stats,

        "success": success,
        "fail_reason": fail_reason,
        "exec_attempts_used": attempts_used,
        "exec_run_retries": max(0, attempts_used - 1),  # how many reruns were needed for executor stage
        "exec_wall_time_sec": round(exec_wall_time_sec, 6) if exec_wall_time_sec is not None else None,
    }

    if log_metrics:
        stat.update(log_metrics)

    # persist per-run summary
    (run_dir / "summary.json").write_text(json.dumps(stat, indent=2))
    return stat


# ------------------------------ SCENARIO RUNNER -------------------------------

def resolve_case_files(case_spec: Dict[str, Any]) -> List[Path]:
    files = sorted(PROBLEMS_DIR.glob(case_spec["glob"]))
    return files


async def run_one_case(case_spec: Dict[str, Any], batch_dir: Path) -> Dict[str, Any]:
    env = env_base()

    case_name = case_spec["name"]
    case_dir = batch_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    files = resolve_case_files(case_spec)
    if not files:
        raise RuntimeError(f"Case {case_name}: no files matched glob '{case_spec['glob']}' under {PROBLEMS_DIR}")

    # sanity expectation: sub cases should have 5 files
    is_sub = ("sub_" in case_name)
    if is_sub and len(files) != 5:
        print(f"[WARN] {case_name}: expected 5 files but got {len(files)}")

    print(f"\n=== CASE: {case_name} ===")
    print("  files:")
    for p in files:
        print("   -", p)

    runs: List[Dict[str, Any]] = []
    successes = 0

    # run each problem file N_RUNS_PER_PROBLEM times
    for pidx, problem_path in enumerate(files):
        for ridx in range(N_RUNS_PER_PROBLEM):
            run_tag = f"{problem_path.stem}__run{ridx:02d}"
            run_dir = case_dir / run_tag
            label = f"{case_name}|{problem_path.name}|{ridx+1}/{N_RUNS_PER_PROBLEM}"

            print(f"[{label}] start")
            st = await run_one_problem_once(problem_path, run_dir, env, label)
            runs.append(st)
            if st.get("success"):
                successes += 1
            print(f"[{label}] done: success={st.get('success')} "
                  f"smashes={st.get('num_smashes')} actions={st.get('num_actions')} "
                  f"knock_retry={st.get('knock_retry')} "
                  f"tamp={st.get('tamp_time_sec')} exec={st.get('execution_time_sec')} total={st.get('total_time_sec')} "
                  f"attempts={st.get('exec_attempts_used')}")

    n_runs = len(runs)
    success_rate = successes / max(1, n_runs)

    # averages over successful runs (for times/knock_retry); counts over all runs
    runs_succ = [r for r in runs if r.get("success") is True]

    summary = {
        "case": case_name,
        "glob": case_spec["glob"],
        "n_problem_files": len(files),
        "n_runs_per_problem": N_RUNS_PER_PROBLEM,
        "n_runs_total": n_runs,
        "successes": successes,
        "success_rate": success_rate,

        # Problem-level constants (still averaged over runs, but should be stable)
        "avg_num_smashes": _avg_present(runs, "num_smashes"),
        "avg_num_actions": _avg_present(runs, "num_actions"),

        # Runtime metrics (use only successful runs, otherwise missing)
        "avg_tamp_time_sec": _avg_present(runs_succ, "tamp_time_sec"),
        "avg_execution_time_sec": _avg_present(runs_succ, "execution_time_sec"),
        "avg_total_time_sec": _avg_present(runs_succ, "total_time_sec"),
        "avg_knock_retry": _avg_present(runs_succ, "knock_retry"),

        # Wall time we measured around executor stage (successful runs only)
        "avg_exec_wall_time_sec": _avg_present(runs_succ, "exec_wall_time_sec"),
        "total_exec_wall_time_sec": _sum_present(runs_succ, "exec_wall_time_sec"),

        # Rerun bookkeeping
        "total_exec_run_retries": _sum_present(runs, "exec_run_retries"),
        "avg_exec_run_retries": _avg_present(runs, "exec_run_retries"),

        "runs": runs,
    }

    (case_dir / "case_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[CASE {case_name}] success {successes}/{n_runs} = {success_rate:.2%}")
    return summary


# ------------------------------ OUTPUT TABLE ----------------------------------

def format_float(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def print_summary_table(overall: List[Dict[str, Any]]) -> None:
    # simple fixed-width table
    headers = [
        "case",
        "runs",
        "succ%",
        "smash",
        "actions",
        "tamp(s)",
        "exec(s)",
        "total(s)",
        "knRetry",
        "runRetries",
    ]
    rows = []
    for s in overall:
        rows.append([
            s["case"],
            str(s["n_runs_total"]),
            f"{100.0*s['success_rate']:.1f}",
            format_float(s.get("avg_num_smashes"), 1),
            format_float(s.get("avg_num_actions"), 1),
            format_float(s.get("avg_tamp_time_sec"), 3),
            format_float(s.get("avg_execution_time_sec"), 3),
            format_float(s.get("avg_total_time_sec"), 3),
            format_float(s.get("avg_knock_retry"), 2),
            str(int(s.get("total_exec_run_retries") or 0)),
        ])

    # compute column widths
    colw = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            colw[i] = max(colw[i], len(cell))

    def fmt_row(r):
        return " | ".join(cell.ljust(colw[i]) for i, cell in enumerate(r))

    sep = "-+-".join("-"*w for w in colw)
    print("\n=== OVERALL SUMMARY ===")
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()


# ---------------------------------- MAIN --------------------------------------

async def main():
    batch_stamp = nowstamp()
    batch_dir = RUNS_DIR / batch_stamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    # check problems dir
    if not PROBLEMS_DIR.exists():
        raise RuntimeError(f"Missing {PROBLEMS_DIR}. Run problem generation first.")

    # run all cases
    all_summaries: List[Dict[str, Any]] = []
    for case in CASE_SPECS:
        s = await run_one_case(case, batch_dir=batch_dir)
        all_summaries.append(s)

    overall = {
        "batch_stamp": batch_stamp,
        "batch_dir": str(batch_dir),
        "n_cases": len(all_summaries),
        "cases": all_summaries,
    }
    (batch_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2))

    print_summary_table(all_summaries)
    print(f"[DONE] wrote overall_summary.json to {batch_dir}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

