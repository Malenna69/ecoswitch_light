# EcoSwitch — Web Deploy Pack

Ready for Streamlit Cloud, Hugging Face Spaces, or Render.
- **Main file**: `app.py`
- **Core**: `modules/__init__.py` (copied from your `core.py`)

## Deploy on Streamlit Cloud
1) Push these files to a GitHub repo (root).
2) On share.streamlit.io pick the repo and set **Main file** = `app.py`.
3) Deploy. If Python=3.13 causes issues with `pvlib`, either switch runtime to 3.11 or rely on the conditional requirement below.

