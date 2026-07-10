# 🌩️ Nimbus AutoML Frontend

A premium, interactive **Streamlit frontend** dashboard for the **Nimbus** multi-agent AutoML pipeline. 

This UI allows users to upload a custom CSV dataset, select a target variable, monitor real-time execution of the agentic graph, visualize leaderboard scores, and download both the narrative markdown report and the self-contained `model.pkl` deployment bundle.

## 🚀 Features

- **Upload & Preview**: Upload custom CSV datasets and preview structure, columns, and dimensions.
- **Instant Testing**: One-click synthetic churn dataset generation for zero-setup smoke testing.
- **Continuous Execution Progress (Real-Time)**: Streams the LangGraph nodes (`graph.stream()`) live to show current agent outputs (Profiler concerns, Data Prep plans, Feature Selector picks, Model Leaderboard validation, and Supervisor retries).
- **Interactive Report Hub**: Renders the complete markdown report directly in the app.
- **Frictionless Downloads**: 1-click download of the `model.pkl` (fitted via `joblib`) and the generated executive summary `nimbus_report.md`.
- **Flexible configuration**: Select LLM providers (Gemini, Groq, Ollama) and customize models, retries, and keys dynamically.

## 📁 Repository Structure

```
nimbus-frontend/
├── .streamlit/
│   └── config.toml        # Theme configuration (premium dark theme)
├── automl_agents/         # Copy of the nimbus backend package
│   ├── graph/
│   ├── nodes/
│   ├── tools/
│   ├── schemas.py
│   └── ...
├── app.py                 # Core Streamlit application
├── requirements.txt       # Frontend & backend Python package dependencies
├── .env.example           # Environment template file
├── .gitignore             # Standard Python/Streamlit exclusions
└── README.md              # Project documentation
```
