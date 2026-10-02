"""Train the Lab 4 policy: behaviour cloning of the teacher, then two DAgger rounds.

Run from the repository root:

    .venv/Scripts/python.exe labs/lab4/train.py            # full run, writes weight.pth
    .venv/Scripts/python.exe labs/lab4/train.py --quick    # small run to check the pipeline

Behaviour cloning fits the teacher's action at the states it was shown.  A
cloned policy then drifts into states the teacher never visited, so DAgger
runs the policy itself, asks the teacher what it should have done at every
state the policy reached, and trains again on the lot.
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import dataset as D
import runtime as R
from expert import expert_policy
from Policy import MAX_TURN_RATE, Policy

LAB_DIR = Path(__file__).resolve().parent
WEIGHTS = LAB_DIR / "weight.pth"
HARDWARE_LOG_WEIGHT = 10


def as_controller(policy):
    """The network as the run loop calls it: numpy state in, numpy action out."""
    def act(state):
        with torch.no_grad():
            return policy(torch.tensor(state, dtype=torch.float32)).numpy().astype(np.float64)
    return act


def _tensors(states, actions):
    return (torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions[:, 0] > 0.0, dtype=torch.float32),
            torch.tensor(actions[:, 1] / MAX_TURN_RATE, dtype=torch.float32))


def fit(policy, train_data, val_data, epochs, seed, lr=1e-3, batch=256, patience=10):
    """Fit the drive/stop decision and the turn rate, keeping the best validation epoch."""
    torch.manual_seed(seed)
    x, drive, rate = _tensors(*train_data)
    vx, vdrive, vrate = _tensors(*val_data)
    # Headings sampled anywhere make stopping-to-turn the common label; weight
    # the classes so that neither decision becomes the default.
    pos_weight = (1.0 - drive).sum() / drive.sum().clamp(min=1.0)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth = torch.nn.SmoothL1Loss()

    def loss_of(inputs, target_drive, target_rate):
        raw = policy.raw_outputs(inputs)
        # Both terms are dimensionless: a logit, and a rate over its maximum.
        return bce(raw[:, 0], target_drive) + 2.0 * smooth(torch.tanh(raw[:, 1]), target_rate)

    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, stale, epoch = float("inf"), None, 0, 0
    for epoch in range(epochs):
        policy.train()
        shuffled = torch.randperm(len(x))
        for i in range(0, len(shuffled), batch):
            index = shuffled[i:i + batch]
            optimiser.zero_grad()
            loss = loss_of(x[index], drive[index], rate[index])
            loss.backward()
            optimiser.step()
        policy.eval()
        with torch.no_grad():
            val_loss = float(loss_of(vx, vdrive, vrate))
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    policy.load_state_dict(best_state)
    policy.eval()
    with torch.no_grad():
        raw = policy.raw_outputs(vx)
        accuracy = float(((raw[:, 0] > 0.0).float() == vdrive).float().mean())
        rate_error = float((MAX_TURN_RATE * (torch.tanh(raw[:, 1]) - vrate)).abs().median())
    print(f"  fit: {len(x)} train / {len(vx)} fixed val, {epoch + 1} epochs, val loss {best:.4f}, "
          f"drive/stop agreement {accuracy:.3f}, median turn-rate error {rate_error:.3f} rad/s")
    return policy


def _stack(*pairs):
    return (np.vstack([s for s, _ in pairs]), np.vstack([a for _, a in pairs]))


def evaluate(controller, scenarios):
    results = []
    for scenario in scenarios:
        out = D.run_scenario(controller, scenario)
        m = R.metrics(out["records"])
        results.append({
            "passed": out["stop_reason"] == "arrived" and R.passes(m), "metrics": m,
            "steps": out["steps"], "stop": out["stop_reason"],
            "interventions": out["interventions"], "decisions": out["decisions"],
            "reasons": out["reasons"],
        })
    return results


def report(name, results):
    passed = [r for r in results if r["passed"]]
    reasons = Counter()
    for r in results:
        reasons.update(r["reasons"])
    changed = sum(r["interventions"] for r in results) / max(1, sum(r["decisions"] for r in results))
    worst = {k: max(r["metrics"][k] for r in results) for k in R.LIMITS}
    median_steps = float(np.median([r["steps"] for r in passed])) if passed else float("nan")
    print(f"  {name:7}: {len(passed)}/{len(results)} pass, median {median_steps:.0f} steps, "
          f"safety changed {changed:.1%} of actions {dict(reasons)}; worst "
          + ", ".join(f"{k} {v:.3f}" for k, v in worst.items()))
    failures = Counter(r["stop"] for r in results if not r["passed"])
    if failures:
        print(f"           failures: {dict(failures)}")
    return len(passed) / len(results)


def agreement_on_hardware_positions(policy, rng):
    """How often the policy decides as the teacher would where the real robot went."""
    states, actions = D.label(D.hardware_states(D.VALIDATION_RUNS, per_point=4, rng=rng))
    with torch.no_grad():
        out = policy(torch.tensor(states, dtype=torch.float32)).numpy()
    drive = float(((out[:, 0] > 0) == (actions[:, 0] > 0)).mean())
    rate = float(np.median(np.abs(out[:, 1] - actions[:, 1])))
    print(f"  data7/8 positions ({len(states)} states): drive/stop agreement {drive:.3f}, "
          f"median turn-rate error {rate:.3f} rad/s")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Lab 4 behaviour-cloning policy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small run to check the pipeline")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    size = 0.1 if args.quick else 1.0
    started = time.time()

    print("Labelling the dataset ...")
    sources = {
        "synthetic": D.synthetic_states(int(20000 * size), rng),
        "teacher runs": D.expert_run_states(max(3, int(60 * size)), rng),
        "hardware positions": D.hardware_states(D.TRAIN_RUNS, per_point=8, rng=rng),
        # Whatever lab4.py has recorded on the robot since, under logs/.
        "hardware logs": D.logged_states(),
    }
    pairs = []
    for name, raw in sources.items():
        if len(raw) == 0:
            print(f"  {name}: none yet")
            continue
        kept, labels = D.label(raw)
        print(f"  {name}: {len(kept)} of {len(raw)} states answerable, "
              f"{(labels[:, 0] > 0).mean():.0%} labelled drive")
        if name == "hardware logs":
            # A few hundred robot states against ~40k simulated ones would be
            # averaged away; they are the states the real robot actually
            # reaches, so they count HARDWARE_LOG_WEIGHT times.
            kept, labels = (np.repeat(a, HARDWARE_LOG_WEIGHT, axis=0) for a in (kept, labels))
            print(f"    counted {HARDWARE_LOG_WEIGHT} times each")
        pairs.append((kept, labels))
    train_data = _stack(*pairs)
    # Fixed before any training and never trained on: a synthetic draw of its
    # own, whole teacher runs of its own, and the two held-out hardware runs.
    # DAgger adds to the training set only.
    val_rng = np.random.default_rng(args.seed + 500)
    val_data = _stack(
        D.label(D.synthetic_states(int(2000 * size), val_rng)),
        D.label(D.expert_run_states(max(1, int(6 * size)), val_rng)),
        D.label(D.hardware_states(D.VALIDATION_RUNS, per_point=4, rng=val_rng)),
    )
    print(f"  {len(val_data[0])} validation states held out; {time.time() - started:.0f} s")

    policy = Policy()
    print("Behaviour cloning ...")
    fit(policy, train_data, val_data, epochs=20 if args.quick else 100, seed=args.seed)

    for round_ in (1, 2):
        print(f"DAgger round {round_} ...")
        scenarios = [D.random_scenario(rng) for _ in range(3 if args.quick else 20)]
        visited = D.label(D.visited_states(as_controller(policy), scenarios))
        print(f"  {len(visited[0])} states the policy reached, relabelled by the teacher")
        train_data = _stack(train_data, visited)
        fit(policy, train_data, val_data, epochs=10 if args.quick else 40, seed=args.seed + round_)
        print(f"  {time.time() - started:.0f} s")

    torch.save(policy.state_dict(), WEIGHTS)
    print(f"Saved {WEIGHTS}")

    print("Closed loop on unseen scenarios, policy against the teacher ...")
    held_out = np.random.default_rng(args.seed + 1000)
    scenarios = [D.random_scenario(held_out) for _ in range(10 if args.quick else 100)]
    report("policy", evaluate(as_controller(policy), scenarios))
    report("teacher", evaluate(expert_policy, scenarios))
    agreement_on_hardware_positions(policy, np.random.default_rng(args.seed + 2000))
    print(f"Done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
