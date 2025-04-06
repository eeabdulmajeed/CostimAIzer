import streamlit as st
from backend.cost_estimator import CostEstimator

st.title("CostimAIze - Smart Pricing Engineer")

estimator = CostEstimator()

task_description = st.text_area("Enter task description:", "Build a small house with 3 rooms in a high-inflation area")
if st.button("Estimate Cost"):
    result = estimator.analyze_and_estimate(task_description)
    st.write("Analysis Details:", result)

actual_bid = st.number_input("Enter actual bid (USD):", min_value=0.0)
if actual_bid > 0 and st.button("Analyze Bid"):
    analysis = estimator.compare_with_bid(task_description, actual_bid)
    st.write("Bid Analysis:", analysis)

user_input = st.text_input("Add comments or adjustments:")
if user_input and st.button("Update Estimate"):
    update = estimator.update_with_user_input(task_description, user_input)
    st.write("Update:", update)