from pathlib import Path

import pytest

from src.config import DATASET_PATH
from src.data_validation import DataValidationError, load_dataset, validate_dataset_dataframe
from src.generate_dataset import generate_dataset


def test_generate_dataset_creates_10000_rows():
    path = Path("data/test_student_dropout.csv")
    df = generate_dataset(path, rows=1000)
    assert len(df) == 1000
    assert df["student_id"].is_unique
    assert not df.isnull().any().any()
    assert df["dropout_next_semester"].isin([0, 1]).all()
    path.unlink(missing_ok=True)


def test_default_dataset_is_valid():
    df = load_dataset(DATASET_PATH)
    assert len(df) == 10_000
    assert df["student_id"].is_unique
    assert df["dropout_next_semester"].isin([0, 1]).all()
    assert not df.isnull().any().any()
    assert df["previous_gpa"].between(0, 10).all()
    assert df["current_gpa"].between(0, 10).all()


def test_gpa_validation_uses_ten_point_scale():
    row = load_dataset(DATASET_PATH).head(1).copy()
    row["previous_gpa"] = 10.0
    row["current_gpa"] = 9.5
    validate_dataset_dataframe(row)

    row["current_gpa"] = 10.1
    with pytest.raises(DataValidationError, match="current_gpa"):
        validate_dataset_dataframe(row)
