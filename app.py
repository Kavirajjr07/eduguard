from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from src.auth import login_user, logout_user, require_permission, require_role
from src.config import DATASET_PATH, FEATURE_COLUMNS, MODEL_PATH
from src.counselling import build_recommendation, next_followup_date, risk_level_from_probability
from src.data_validation import load_dataset, validate_csv_upload
from src.database import CounsellingCase, Intervention, SessionLocal, User, create_db_and_tables
from src.predict import bulk_predict, load_metadata, load_model, predict_student
from src.seed_database import seed_demo_users
from src.ui_components import build_risk_chart, ensure_model_available, render_demo_notice, render_metric_card

st.set_page_config(page_title="EduGuard AI", page_icon="🎓", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


def inject_creative_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #e0f2fe 0%, #f8fafc 45%, #ecfeff 100%);
                color: #0f172a;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.96));
                color: #e2e8f0;
            }
            .stMetric > div {
                background: rgba(255,255,255,0.78);
                border: 1px solid rgba(148,163,184,0.35);
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            }
            .stButton > button {
                border-radius: 999px;
                padding: 0.7rem 1.2rem;
                background: linear-gradient(135deg, #2563eb, #14b8a6);
                color: white;
                border: none;
                font-weight: 600;
            }
            .hero-panel {
                background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(20,184,166,0.14));
                border: 1px solid rgba(37,99,235,0.15);
                border-radius: 30px;
                padding: 2rem;
                box-shadow: 0 20px 50px rgba(37,99,235,0.12);
            }
            .login-panel {
                background: rgba(255,255,255,0.88);
                border: 1px solid rgba(148,163,184,0.35);
                border-radius: 28px;
                padding: 2rem;
                box-shadow: 0 22px 48px rgba(15, 23, 42, 0.12);
            }
            .chip {
                display: inline-block;
                background: rgba(14, 165, 233, 0.1);
                color: #0f172a;
                border-radius: 999px;
                padding: 0.4rem 0.8rem;
                font-size: 0.8rem;
                font-weight: 600;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .big-title {
                font-size: 3.2rem;
                line-height: 1.05;
                font-weight: 800;
                letter-spacing: -0.05em;
            }
            .subtitle {
                font-size: 1.05rem;
                color: #334155;
                line-height: 1.7;
            }
            .accent-text { color: #1d4ed8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_creative_css()


def show_login_screen() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <span class="chip">Smart India Hackathon 2026</span>
            <div class="big-title">EduGuard AI</div>
            <div class="subtitle">Predict Early. Support Personally. Prevent Dropouts.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            <div class="hero-panel" style="margin-top: 1.25rem; min-height: 330px;">
                <h3>Why institutions choose EduGuard</h3>
                <p class="subtitle">Early-warning analytics, personalized counselling guidance, and privacy-conscious support workflows designed for academic teams.</p>
                <div class="chip">AI model monitoring</div>
                <div class="chip">Counselling insights</div>
                <div class="chip">Risk heatmaps</div>
                <div class="chip">Support planning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="login-panel" style="margin-top: 1.25rem;">', unsafe_allow_html=True)
        st.markdown("### Secure access")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if login_user(username, password):
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.markdown("### Demo accounts")
        st.code("admin / admin123\nfaculty / faculty123\ncounsellor / counsellor123\nstudent / student123")
        st.caption("If demo passwords were overridden with environment variables, use those configured values. Demo credentials are for local testing only.")
        st.markdown("</div>", unsafe_allow_html=True)


def sidebar_navigation() -> str:
    user = st.session_state.get("user", {})
    role = user.get("role", "STUDENT")
    options = [
        "Executive Dashboard",
        "Student Analytics",
        "Risk Prediction",
        "Bulk Prediction",
        "Adaptive Counselling",
        "Counselling Follow-up",
        "Model Performance",
        "Institutional Insights",
        "Settings",
    ]
    if role == "STUDENT":
        visible_options = ["Risk Prediction", "Settings"]
    elif role == "FACULTY":
        visible_options = ["Executive Dashboard", "Student Analytics", "Risk Prediction", "Bulk Prediction", "Model Performance", "Institutional Insights", "Settings"]
    elif role == "COUNSELLOR":
        visible_options = ["Adaptive Counselling", "Counselling Follow-up", "Risk Prediction", "Settings"]
    else:
        visible_options = options

    return st.sidebar.radio("Navigation", visible_options, index=0)


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str, modified_time: float):
    return load_model(model_path)


def get_model():
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model is missing: {model_path}. Run `python -m src.train_model`.")
    return cached_model(str(model_path), model_path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def dataset_snapshot(dataset_path: str, dataset_modified_time: float, model_path: str, model_modified_time: float) -> pd.DataFrame:
    dataset = load_dataset(dataset_path)
    model = cached_model(model_path, model_modified_time)
    prediction_df = dataset.copy()
    prediction_df["probability"] = model.predict_proba(prediction_df[FEATURE_COLUMNS])[:, 1]
    prediction_df["risk_level"] = prediction_df["probability"].apply(risk_level_from_probability)
    return prediction_df


def get_dataset_snapshot() -> pd.DataFrame:
    dataset_path = Path(DATASET_PATH)
    model_path = Path(MODEL_PATH)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Student dataset is missing: {dataset_path}. Run `python -m src.generate_dataset`.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model is missing: {model_path}. Run `python -m src.train_model`.")
    return dataset_snapshot(
        str(dataset_path),
        dataset_path.stat().st_mtime,
        str(model_path),
        model_path.stat().st_mtime,
    )


def render_dashboard() -> None:
    st.markdown("<h1 style='margin-bottom:0.4rem;'>Executive Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Institutional overview across student support and warning signals.")
    render_demo_notice()
    if not ensure_model_available():
        return
    df = get_dataset_snapshot()
    total_students = len(df)
    risk_counts = df["risk_level"].value_counts().to_dict()
    avg_risk = df["probability"].mean() * 100
    support_review = len(df[df["risk_level"].isin(["Medium", "High"])])
    active_cases = 14
    completed_interventions = 21

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total students", total_students)
    with col2:
        render_metric_card("Low risk", risk_counts.get("Low", 0))
    with col3:
        render_metric_card("Medium risk", risk_counts.get("Medium", 0))
    with col4:
        render_metric_card("High risk", risk_counts.get("High", 0))
    with col5:
        render_metric_card("Avg predicted risk", f"{avg_risk:.1f}%")

    st.markdown("### Risk distribution")
    fig = build_risk_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Department overview")
    dept_summary = df.groupby("department").agg(avg_risk=("probability", "mean"), students=("student_id", "count")).reset_index()
    st.dataframe(dept_summary.sort_values("avg_risk", ascending=False), use_container_width=True)


def render_student_analytics() -> None:
    st.title("Student Analytics")
    if not ensure_model_available():
        return
    df = get_dataset_snapshot()
    department = st.selectbox("Department", ["All"] + sorted(df["department"].unique().tolist()))
    year = st.selectbox("Year", ["All"] + sorted(df["year_of_study"].unique().tolist()))
    attendance_threshold = st.slider("Attendance cutoff (%)", 0, 100, 75)
    gpa_threshold = st.slider("Current GPA cutoff (/10)", 0.0, 10.0, 6.25, step=0.05)
    risk_filter = st.selectbox("Risk category", ["All", "Low", "Medium", "High"])

    filtered = df.copy()
    if department != "All":
        filtered = filtered[filtered["department"] == department]
    if year != "All":
        filtered = filtered[filtered["year_of_study"] == int(year)]
    filtered = filtered[filtered["attendance_pct"] >= attendance_threshold]
    filtered = filtered[filtered["current_gpa"] >= gpa_threshold]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    st.dataframe(filtered[["student_id", "department", "year_of_study", "attendance_pct", "current_gpa", "risk_level", "probability"]].sort_values("probability", ascending=False), use_container_width=True)


def render_individual_prediction() -> None:
    st.title("Individual Risk Prediction")
    st.caption("Enter academic and engagement indicators to estimate dropout risk.")
    if not ensure_model_available():
        return
    model = get_model()
    metadata = load_metadata()

    with st.form("student_prediction_form"):
        student_id = st.text_input("Student ID", placeholder="STU000123")
        age = st.number_input("Age", min_value=15, max_value=40, value=20)
        year_of_study = st.number_input("Year of study", min_value=1, max_value=6, value=2)
        department = st.selectbox("Department", ["CSE", "ECE", "ME", "EEE", "Civil", "IT", "Biotech", "MBA", "Law", "Arts"])
        attendance_pct = st.slider("Attendance (%)", 0, 100, 78)
        internal_marks = st.slider("Internal marks (%)", 0, 100, 70)
        assignment_completion_pct = st.slider("Assignment completion (%)", 0, 100, 80)
        previous_gpa = st.slider("Previous GPA (/10)", 0.0, 10.0, 8.0, step=0.05)
        current_gpa = st.slider("Current GPA (/10)", 0.0, 10.0, 7.25, step=0.05)
        failed_subjects = st.number_input("Failed subjects", min_value=0, max_value=8, value=1)
        study_hours_weekly = st.number_input("Study hours per week", min_value=1, max_value=50, value=12)
        commute_minutes = st.number_input("Commute minutes", min_value=0, max_value=180, value=35)
        family_income_band = st.selectbox("Family income band", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
        first_generation_student = st.selectbox("First generation student", [0, 1])
        scholarship = st.selectbox("Scholarship", [0, 1])
        fee_payment_delay_days = st.number_input("Fee payment delay days", min_value=0, max_value=180, value=5)
        engagement_score = st.slider("Engagement score", 0, 100, 68)
        counselling_sessions = st.number_input("Counselling sessions", min_value=0, max_value=20, value=1)
        submitted = st.form_submit_button("Predict risk")

    if submitted:
        student = {
            "student_id": student_id or "STU000123",
            "age": int(age),
            "year_of_study": int(year_of_study),
            "department": department,
            "attendance_pct": int(attendance_pct),
            "internal_marks": int(internal_marks),
            "assignment_completion_pct": int(assignment_completion_pct),
            "previous_gpa": float(previous_gpa),
            "current_gpa": float(current_gpa),
            "failed_subjects": int(failed_subjects),
            "study_hours_weekly": int(study_hours_weekly),
            "commute_minutes": int(commute_minutes),
            "family_income_band": family_income_band,
            "first_generation_student": int(first_generation_student),
            "scholarship": int(scholarship),
            "fee_payment_delay_days": int(fee_payment_delay_days),
            "engagement_score": int(engagement_score),
            "counselling_sessions": int(counselling_sessions),
        }
        result = predict_student(student, model=model, metadata=metadata)
        probability_pct = result["probability"] * 100
        st.metric("Predicted dropout probability", f"{probability_pct:.1f}%")
        st.metric("Risk level", result["risk_level"])
        st.markdown("### Explanation")
        st.write(result["explanation"])
        st.markdown("### Recommended support")
        st.write(f"- {result['intervention']['title']}: {result['intervention']['action']}")


def render_bulk_prediction() -> None:
    st.title("Bulk Prediction")
    st.caption("Upload a CSV containing the required student features for batch scoring.")
    uploaded = st.file_uploader("Choose a CSV", type=["csv"])
    if uploaded is None:
        return
    try:
        df = validate_csv_upload(uploaded)
    except Exception as exc:
        st.error(str(exc))
        return
    model = get_model()
    metadata = load_metadata()
    result = bulk_predict(df, model=model, metadata=metadata)
    st.dataframe(result[["student_id", "probability", "risk_level", "probability_pct"]] if "probability_pct" in result.columns else result[["student_id", "probability", "risk_level"]], use_container_width=True)
    csv = result.to_csv(index=False)
    st.download_button("Download predictions", data=csv, file_name="bulk_predictions.csv", mime="text/csv")


def render_counselling() -> None:
    st.title("Adaptive Counselling")
    if not ensure_model_available():
        return
    df = get_dataset_snapshot().head(20)
    for _, row in df.iterrows():
        recs = build_recommendation(row.to_dict())
        with st.expander(f"{row['student_id']} - {row['risk_level']} risk"):
            st.write(f"Probability: {row['probability'] * 100:.1f}%")
            for rec in recs:
                st.markdown(f"- **{rec['title']}**: {rec['action']} (Priority: {rec['priority']})")


def render_follow_up() -> None:
    st.title("Counselling Follow-up")
    st.table(pd.DataFrame([
        {"Case": "C-101", "Student": "STU000012", "Status": "In Progress", "Follow-up": "2026-11-10", "Outcome": "Attendance improved"},
        {"Case": "C-102", "Student": "STU000245", "Status": "Follow-up Required", "Follow-up": "2026-11-11", "Outcome": "Pending"},
    ]))


def render_model_performance() -> None:
    st.title("Model Performance")
    metadata = load_metadata()
    if not metadata or "metrics" not in metadata or "test" not in metadata["metrics"]:
        st.error("Model evaluation metadata is missing or incomplete. Run `python -m src.train_model` to regenerate the model and metrics.")
        return
    test_metrics = metadata["metrics"]["test"]
    st.caption("Evaluation on a held-out synthetic test set. These results are not evidence of real-world effectiveness.")
    metric_columns = st.columns(5)
    for column, (label, key, format_string) in zip(
        metric_columns,
        [
            ("ROC-AUC", "roc_auc", ".3f"),
            ("PR-AUC", "pr_auc", ".3f"),
            ("Precision", "precision", ".3f"),
            ("Recall", "recall", ".3f"),
            ("Brier score", "brier_score", ".3f"),
        ],
    ):
        value = test_metrics.get(key)
        column.metric(label, format(value, format_string) if isinstance(value, (int, float)) else "Unavailable")

    st.markdown(f"**Model:** `{metadata.get('selected_model', 'Unknown')}` &nbsp; · &nbsp; **Version:** `{metadata.get('model_version', 'Unknown')}`")
    comparison = metadata["metrics"].get("comparison", {})
    if comparison:
        st.markdown("#### Model comparison")
        comparison_rows = [
            {"Model": name.replace("_", " ").title(), **{
                key: model_metrics.get(key)
                for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score")
            }}
            for name, model_metrics in comparison.items()
        ]
        st.dataframe(pd.DataFrame(comparison_rows).set_index("Model"), use_container_width=True)

    st.markdown("#### Held-out test confusion matrix")
    matrix = test_metrics.get("confusion_matrix")
    if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        matrix_frame = pd.DataFrame(matrix, index=["Actual continue", "Actual dropout"], columns=["Predicted continue", "Predicted dropout"])
        st.dataframe(matrix_frame, use_container_width=True)
    else:
        st.info("Confusion matrix is not available in the saved metadata.")

    if not ensure_model_available():
        return
    try:
        data = load_dataset(DATASET_PATH)
        x_train, x_test, y_train, y_test = train_test_split(
            data[FEATURE_COLUMNS],
            data["dropout_next_semester"],
            test_size=0.20,
            stratify=data["dropout_next_semester"],
            random_state=42,
        )
        # Training used a validation split from the training partition; the test partition
        # therefore remains the same deterministic held-out set.
        del x_train, y_train
        probabilities = get_model().predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="uniform")
        threshold = float(metadata.get("selected_threshold", 0.5))
        predicted_labels = (probabilities >= threshold).astype(int)
        current_matrix = confusion_matrix(y_test, predicted_labels, labels=[0, 1])

        roc_fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line={"dash": "dash"}))
        roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=330)

        pr_fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", name="Precision–recall"))
        pr_fig.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision", height=330)

        calibration_fig = go.Figure(go.Scatter(x=predicted, y=observed, mode="lines+markers", name="Model"))
        calibration_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line={"dash": "dash"}))
        calibration_fig.update_layout(title="Calibration", xaxis_title="Mean predicted probability", yaxis_title="Observed dropout fraction", height=330)

        chart_col1, chart_col2 = st.columns(2)
        chart_col1.plotly_chart(roc_fig, use_container_width=True)
        chart_col2.plotly_chart(pr_fig, use_container_width=True)
        st.plotly_chart(calibration_fig, use_container_width=True)
        st.caption(f"Confusion matrix and curves use the saved threshold ({threshold:.3f}) and the held-out synthetic test split.")
        if not (current_matrix == matrix).all():
            st.warning("The current model's test predictions do not match the saved confusion matrix. Retrain the model to align model and metadata.")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Could not render model evaluation plots: {exc}")


def render_institutional_insights() -> None:
    st.title("Institutional Insights")
    df = get_dataset_snapshot()
    st.dataframe(df.groupby("department").agg(avg_risk=("probability", "mean"), high_risk=("risk_level", lambda s: (s == "High").sum())).reset_index(), use_container_width=True)


def render_settings() -> None:
    st.title("Settings and About")
    st.markdown("## Project goals")
    st.write("EduGuard AI predicts dropout risk using synthetic institutional data and recommends supportive interventions.")
    st.markdown("## Ethical safeguards")
    st.write("The model must not trigger punitive academic action. Staff review and human judgment remain essential.")
    st.markdown("## Demo credentials")
    st.code("admin / admin123\nfaculty / faculty123\ncounsellor / counsellor123\nstudent / student123")


def main() -> None:
    if not st.session_state.get("authenticated"):
        show_login_screen()
        return

    user = st.session_state.get("user", {})
    st.sidebar.title(f"Welcome, {user.get('full_name', user.get('username', 'User'))}")
    st.sidebar.write(f"Role: {user.get('role', 'STUDENT')}")
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()

    selected = sidebar_navigation()
    if selected == "Executive Dashboard":
        render_dashboard()
    elif selected == "Student Analytics":
        render_student_analytics()
    elif selected == "Risk Prediction":
        render_individual_prediction()
    elif selected == "Bulk Prediction":
        render_bulk_prediction()
    elif selected == "Adaptive Counselling":
        render_counselling()
    elif selected == "Counselling Follow-up":
        render_follow_up()
    elif selected == "Model Performance":
        render_model_performance()
    elif selected == "Institutional Insights":
        render_institutional_insights()
    else:
        render_settings()


if __name__ == "__main__":
    create_db_and_tables()
    seed_demo_users()
    main()
