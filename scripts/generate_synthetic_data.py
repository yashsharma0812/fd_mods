"""Generate a small IBM-shaped dataset for end-to-end smoke testing."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(path: Path, users: int = 30, transactions_per_user: int = 26) -> None:
    """Write deterministic transaction histories with learnable precursor patterns."""
    rng = random.Random(42)
    fields = [
        "User",
        "Card",
        "Year",
        "Month",
        "Day",
        "Time",
        "Amount",
        "Use Chip",
        "Merchant Name",
        "Merchant City",
        "Merchant State",
        "Zip",
        "MCC",
        "Errors?",
        "Is Fraud?",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for user_index in range(users):
            timestamp = datetime(2024, 1, 1, 8, 0) + timedelta(days=user_index)
            fraud_targets = {12, 22}
            for transaction_index in range(transactions_per_user):
                is_precursor = any(
                    target - 5 <= transaction_index < target
                    for target in fraud_targets
                )
                if is_precursor:
                    amount = 180 + transaction_index * 8 + rng.uniform(-5, 5)
                    use_chip = "Online Transaction"
                    merchant = f"NOVEL_{user_index}_{transaction_index}"
                    city = "Online"
                    state = "UNKNOWN"
                    mcc = 5734
                    minutes = 5
                else:
                    amount = 15 + rng.uniform(0, 35)
                    use_chip = "Chip Transaction"
                    merchant = f"LOCAL_{user_index % 5}"
                    city = ["Austin", "Boston", "Denver"][user_index % 3]
                    state = ["TX", "MA", "CO"][user_index % 3]
                    mcc = 5411
                    minutes = 90
                timestamp += timedelta(minutes=minutes)
                writer.writerow(
                    {
                        "User": user_index,
                        "Card": user_index % 2,
                        "Year": timestamp.year,
                        "Month": timestamp.month,
                        "Day": timestamp.day,
                        "Time": timestamp.strftime("%H:%M"),
                        "Amount": f"${amount:,.2f}",
                        "Use Chip": use_chip,
                        "Merchant Name": merchant,
                        "Merchant City": city,
                        "Merchant State": state,
                        "Zip": 10000 + user_index,
                        "MCC": mcc,
                        "Errors?": "UNKNOWN",
                        "Is Fraud?": "Yes" if transaction_index in fraud_targets else "No",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/raw/synthetic_ibm_transactions.csv",
        type=Path,
    )
    args = parser.parse_args()
    generate(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
