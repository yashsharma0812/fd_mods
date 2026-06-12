import pandas as pd

from src.splits import split_transactions


def test_user_level_splits_are_disjoint():
    frame = pd.DataFrame(
        {
            "User": [f"U{number}" for number in range(20) for _ in range(3)],
            "value": range(60),
        }
    )
    splits = split_transactions(frame, "User", 0.7, 0.15, 0.15, 42, True)

    train = set(splits.train["User"])
    validation = set(splits.validation["User"])
    test = set(splits.test["User"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(frame["User"])
