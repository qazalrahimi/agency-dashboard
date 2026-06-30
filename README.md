# 📊 Agency Operations Dashboard

An interactive dashboard analyzing advertising agency project operations — built with Python.

## Live Demo
[Add your live link here after deployment]

## Features
- 📊 Project status overview (completed, in-progress, cancelled, on-hold)
- 📈 Delay and revision analysis by unit and category
- 🔍 Deep dive into cancellation/hold reasons and client approval rates
- 👤 Individual team member performance and workload
- 🏢 Brand performance ranking with Key Account Score
- 🤖 Ask AI questions about the data in natural language

## Built With
- Python + Streamlit
- Pandas — data analysis
- Plotly — interactive charts
- Groq API (LLaMA 3.3) — AI-powered insights

## Project Structure
- `app.py` — Main Streamlit dashboard
- `agency_dashboard_analysis.ipynb` — Step-by-step Jupyter notebook exploring the data
- `agency_projects.csv` — Sample dataset
- `requirements.txt` — Python dependencies

## Installation

git clone https://github.com/qazalrahimi/agency-dashboard.git
cd agency-dashboard
pip install -r requirements.txt

Create a `.env` file:
GROQ_API_KEY=your_groq_api_key

Run:
streamlit run app.py
