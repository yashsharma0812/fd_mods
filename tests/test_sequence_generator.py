import numpy as np
import pandas as pd

from src.sequence_generator import generate_sequences


def test_target_transaction_is_not_in_input_window():
    frame = pd.DataFrame(
        {
            "User": ["A"] * 6,
            "Is Fraud?": [0, 0, 0, 0, 0, 1],
        }
    )
    # The feature value identifies the source transaction exactly.
    transformed = np.arange(6, dtype=np.float32).reshape(-1, 1)

    result = generate_sequences(
        frame,
        transformed,
        sequence_length=5,
        user_column="User",
        target_column="Is Fraud?",
    )

    assert result.features.shape == (1, 5, 1)
    np.testing.assert_array_equal(result.features[0, :, 0], [0, 1, 2, 3, 4])
    assert result.labels.tolist() == [1.0]
    assert 5 not in result.features[0, :, 0]


def test_windows_never_cross_users():
    frame = pd.DataFrame(
        {
            "User": ["A", "A", "A", "B", "B", "B"],
            "Is Fraud?": [0, 0, 1, 0, 0, 1],
        }
    )
    transformed = np.array([[10], [11], [12], [20], [21], [22]], dtype=np.float32)

    result = generate_sequences(
        frame,
        transformed,
        sequence_length=2,
        user_column="User",
        target_column="Is Fraud?",
    )

    np.testing.assert_array_equal(result.features[:, :, 0], [[10, 11], [20, 21]])
    assert result.target_users.tolist() == ["A", "B"]
