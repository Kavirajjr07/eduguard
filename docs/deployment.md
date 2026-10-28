# Deployment guidance

This application is designed for Streamlit Community Cloud or a lightweight self-hosted Streamlit deployment.

- Add Python dependency file: requirements.txt
- Keep secrets in environment variables or .env files excluded from source control
- Ensure the model has been trained before deployment
- Run database seeding once during setup
