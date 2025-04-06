import streamlit as st
import pandas as pd
import io
from backend import CostimAIzeOrchestrator

st.set_page_config(page_title="CostimAIze", layout="wide")

html_header = """
<div style='text-align: center; margin-bottom: -20px;'>
    <img src="logo.png" width="160"/>
    <h1 style='color:#2C3E50;'>CostimAIze</h1>
    <h4 style='color:#7F8C8D;'>Smart Cost Estimation & Price Analysis System</h4>
</div>
"""
st.markdown(html_header, unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def get_orchestrator():
    return CostimAIzeOrchestrator()

orch = get_orchestrator()

if "service_selected" not in st.session_state:
    st.session_state["service_selected"] = None

service = st.selectbox("Select Service", ["Project Cost Estimation", "Bid Price Analysis", "Upload Historical Prices"])

if st.session_state["service_selected"] != service:
    st.session_state.clear()
    st.session_state["service_selected"] = service

if service == "Project Cost Estimation":
    st.subheader("Project Cost Estimation")
    sow_file = st.file_uploader("Upload Scope of Work (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
    context = {}
    if sow_file:
        st.success("Scope of Work uploaded successfully!")
        context["project_type"] = st.text_input("Project Type")
        context["location"] = st.text_input("Location")
        context["duration"] = st.number_input("Duration (days)", min_value=1)
        context["contract_type"] = st.selectbox("Contract Type", ["Lump Sum Turnkey (LSTK)", "Unit Rate", "LSTK & Unit Rate"])
        context["notes"] = st.text_area("Additional Notes")
        if st.button("Estimate Cost"):
            result = orch.run_estimation(sow_file, context)
            st.write("Estimated Cost Report:")
            st.json(result)

elif service == "Bid Price Analysis":
    st.subheader("Bid Price Analysis")
    sow_file = st.file_uploader("Upload Scope of Work", type=["txt", "pdf", "docx"])
    bid_data = st.file_uploader("Upload Contractor Bid (CSV)", type=["csv"])
    if sow_file and bid_data:
        st.success("Files uploaded successfully!")
        contractor_df = pd.read_csv(bid_data)
        if st.button("Analyze Bid"):
            result = orch.analyze_bid(sow_file, contractor_df.to_dict(orient='records'))
            st.write("Bid Analysis:")
            st.json(result)

elif service == "Upload Historical Prices":
    st.subheader("Upload Historical Prices")
    hist_file = st.file_uploader("Upload Historical Data (CSV, XLSX)", type=["csv", "xlsx"])
    metadata = st.text_area("Metadata (optional)")
    if hist_file and st.button("Archive Data"):
        result = orch.archive_historical_prices(hist_file, metadata)
        st.write("Archive Result:")
        st.write(result)