# KSB Coursework Marker

AI-powered coursework assessment tool that evaluates student work against Knowledge, Skills, and Behaviours (KSB) criteria with Pass/Merit/Referral grading.

## Features

- **Multi-module support:** MLCC, AIDI, DPS
- **KSB-based evaluation** with structured rubrics
- **RAG retrieval** for evidence-based assessment
- **Local LLM** via Ollama (Gemma 3 4B)
- **Export** assessments as Markdown

## Setup
`ash
# Create virtual environment
python -m venv coursework-env
coursework-env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull gemma3:4b

# Run
streamlit run ui/ksb_app.py
`

## Modules

| Module | KSBs | Focus |
|--------|------|-------|
| MLCC | 11 | Cloud ML, performance benchmarking |
| AIDI | 19 | Business value, ethics, stakeholders |
| DPS | 19 | EDA, hypothesis testing, visualisation |
