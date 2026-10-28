from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_PATH, DATA_DIR

RANDOM_SEED = 2026


def generate_dataset(path: str | Path = DATASET_PATH, rows: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    departments = ["CSE", "ECE", "ME", "EEE", "Civil", "IT", "Biotech", "MBA", "Law", "Arts"]
    income_bands = ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]

    records: list[dict] = []
    for idx in range(rows):
        age = int(rng.integers(18, 26))
        year = int(rng.integers(1, 6))
        department = departments[int(rng.integers(len(departments)))]
        attendance = int(rng.integers(45, 100))
        internal_marks = int(rng.integers(35, 96))
        assignment_completion = int(rng.integers(30, 99))
        previous_gpa = round(float(rng.uniform(5.0, 10.0)), 2)
        current_gpa = round(
            float(max(1.25, min(10.0, previous_gpa - rng.uniform(0.0, 3.0) + rng.uniform(0.0, 1.75)))),
            2,
        )
        failed_subjects = int(rng.integers(0, 4))
        study_hours = int(rng.integers(2, 32))
        commute = int(rng.integers(15, 120))
        family_income_band = income_bands[int(rng.integers(len(income_bands)))]
        first_generation = int(rng.integers(0, 2))
        scholarship = int(rng.integers(0, 2))
        fee_delay = int(rng.integers(0, 90))
        engagement = int(rng.integers(25, 99))
        counselling = int(rng.integers(0, 8))

        risk_logit = (
            -2.3
            + (100 - attendance) * 0.055
            + max(0, 8.75 - current_gpa) * 0.56
            + failed_subjects * 0.9
            + fee_delay * 0.015
            + (1 - engagement / 100) * 3.8
            + (1 if family_income_band in {"Low", "Lower-Middle"} else 0) * 0.85
            + (1 if first_generation else 0) * 0.7
            + (1 if commute > 60 else 0) * 0.6
            - study_hours * 0.05
            - rng.uniform(0.2, 1.0)
        )
        dropout_prob = 1 / (1 + math.exp(-risk_logit))
        dropout = int(rng.random() < dropout_prob)

        records.append(
            {
                "student_id": f"STU{idx + 1:06d}",
                "age": age,
                "year_of_study": year,
                "department": department,
                "attendance_pct": attendance,
                "internal_marks": internal_marks,
                "assignment_completion_pct": assignment_completion,
                "previous_gpa": round(previous_gpa, 2),
                "current_gpa": round(current_gpa, 2),
                "failed_subjects": failed_subjects,
                "study_hours_weekly": study_hours,
                "commute_minutes": commute,
                "family_income_band": family_income_band,
                "first_generation_student": first_generation,
                "scholarship": scholarship,
                "fee_payment_delay_days": fee_delay,
                "engagement_score": engagement,
                "counselling_sessions": counselling,
                "dropout_next_semester": dropout,
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("student_id").reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    generate_dataset()
    print("Synthetic dataset created successfully.")
