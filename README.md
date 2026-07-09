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

---

## 🛠️ Getting Started

### Prerequisites

1. Python 3.12+
2. A Gemini or Groq API Key (optional if using Ollama).

### Installation & Run

1. Clone or copy this repository to your computer.
2. Initialize and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate    # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and add your API Keys:
   ```bash
   cp .env.example .env
   # Open .env and add: GOOGLE_API_KEY or GROQ_API_KEY
   ```
5. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### 💡 Quick Integration with Nimbus Workspace

If you are running the frontend on the same machine as your `nimbus` development workspace, the frontend will automatically search for and load your configuration from your backend `.env` file at `C:\Users\dwive\OneDrive\Desktop\nimbus\.env`!

---

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

---

## ☁️ Deployment

To deploy this frontend online (e.g., to Streamlit Community Cloud):
1. Create a public repository on GitHub and push this code.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and link your repository.
3. Add your secrets (`GOOGLE_API_KEY` or `GROQ_API_KEY`) in the Streamlit Cloud dashboard.
4. Deploy!
