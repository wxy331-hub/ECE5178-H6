"""Every race the maze allows, in simulation: each reachable (start plate, heading, goal plate).

    .venv/Scripts/python.exe project/validate.py            # all 2,208 races
    .venv/Scripts/python.exe project/validate.py --quick    # every 8th start/goal pair, all 4 headings

A race passes when the run says 'arrived' and the simulator's true position
has the whole ball on the goal plate.  Judged on the simulator's truth, not on
the estimate the controller stopped on.  Failures are listed so each can be
read off a route.
"""

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import race  # noqa: E402
from mission import make_mission, plan_cells, plate_name, count_turns, reachable_from  # noqa: E402
from sphero_env.envs.custom_maze_full import cell_to_world  # noqa: E402


def all_races():
    cells = [(c, r) for c in range(5) for r in range(5)]
    for start, goal in itertools.product(cells, cells):
        if start != goal and goal in reachable_from(start):
            for heading in (0, 90, 180, 270):
                yield start, heading, goal


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    races = list(all_races())
    if args.quick:
        pairs = sorted({(s, g) for s, _, g in races})[::8]
        races = [(s, h, g) for s, g in pairs for h in (0, 90, 180, 270)]
    env = race.R.make_sim_env(render=False)
    outcomes, failures, times_by_turns = Counter(), [], {}
    started = time.time()
    try:
        for start, heading, goal in races:
            mission = make_mission(tuple(cell_to_world(start)), heading, tuple(cell_to_world(goal)))
            result = race.run_race(mission, env, verbose=False)
            ok = result["outcome"] == "arrived" and result["true_on_goal"]
            outcomes["pass" if ok else result["outcome"] + ("" if result["outcome"] != "arrived" else "-but-off-goal")] += 1
            cells = plan_cells(start, heading, goal)
            if ok:
                times_by_turns.setdefault((len(cells) - 1, count_turns(cells, heading)), []).append(result["elapsed_s"])
            else:
                failures.append(f"{plate_name(start)} h{heading} -> {plate_name(goal)}: {result['outcome']}, "
                                f"true end ({result['true_xy'][0]:+.3f}, {result['true_xy'][1]:+.3f})")
    finally:
        env.close()
    print(f"{len(races)} races in {time.time() - started:.0f} s: {dict(outcomes)}")
    for line in failures[:40]:
        print("  FAIL", line)
    longest = max(times_by_turns, key=lambda k: (k[0], k[1]))
    print(f"simulated time: median {np.median([t for v in times_by_turns.values() for t in v]):.1f} s; "
          f"longest route class {longest[0]} plates/{longest[1]} turns: median {np.median(times_by_turns[longest]):.1f} s")


if __name__ == "__main__":
    main()
