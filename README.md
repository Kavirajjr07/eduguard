# EduGuard AI

EduGuard AI is a synthetic student dropout risk prediction and counselling support system designed for the SIH 2026 institutional demo. It uses a Streamlit dashboard, SQLite persistence, scikit-learn model pipelines, and explainable supportive guidance.

## Quick start

1. python -m pip install -r requirements.txt
2. python -m src.generate_dataset
3. python -m src.train_model
4. python -m src.seed_database
5. python -m streamlit run app.py

## Demo accounts

- admin / admin123
- faculty / faculty123
- counsellor / counsellor123
- student / student123

## Model note

This project uses synthetic data only. Metrics reflect the dataset generated for demonstration and should not be treated as real institutional evidence.
