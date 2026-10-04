from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Shift-Aware NIDS", layout="wide")
st.title("Shift-Aware Selective NIDS proof-of-concept")

results_path = Path(__file__).resolve().parent / "reports" / "poc" / "lodo_poc_results.csv"
if results_path.exists():
    df = pd.read_csv(results_path)
    st.success(f"Loaded {len(df)} fold results from {results_path}")
    st.dataframe(df.round(3), use_container_width=True)
else:
    st.warning("No results yet. Run the proof-of-concept script first: python src/run_poc.py")

with st.sidebar:
    st.header("Status")
    st.write("Current implementation includes:")
    st.markdown(
        "- dataset discovery and verification\n"
        "- leakage-safe preprocessing\n"
        "- LightGBM and MLP baselines\n"
        "- LODO proof-of-concept runner\n"
        "- CSV/JSON result export"
    )
