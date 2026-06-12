"""Create a compact smoke subset from real IBM fraud trajectories."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path


def create_subset(
    source: Path,
    destination: Path,
    user_limit: int = 30,
    history_length: int = 25,
) -> None:
    """Capture prior history plus the first fraud target for selected users."""
    selected: list[list[dict[str, str]]] = []
    current_user: str | None = None
    history: deque[dict[str, str]] = deque(maxlen=history_length)
    selected_current_user = False

    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Source CSV has no header")
        for row in reader:
            user = row["User"]
            if current_user is None or user != current_user:
                if len(selected) >= user_limit:
                    break
                current_user = user
                history.clear()
                selected_current_user = False
            if (
                not selected_current_user
                and row["Is Fraud?"].strip().lower() == "yes"
                and len(history) == history_length
            ):
                selected.append([dict(item) for item in history] + [dict(row)])
                selected_current_user = True
            history.append(dict(row))

    if len(selected) < 3:
        raise RuntimeError(f"Only found {len(selected)} eligible fraud users")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trajectory in selected:
            writer.writerows(trajectory)
    print(f"Wrote {len(selected)} users and {sum(map(len, selected))} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/credit_card_transactions-ibm_v2.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/real_ibm_smoke.csv"),
    )
    parser.add_argument("--users", type=int, default=30)
    args = parser.parse_args()
    create_subset(args.source, args.output, args.users)


if __name__ == "__main__":
    main()
