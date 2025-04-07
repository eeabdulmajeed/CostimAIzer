import streamlit as st
import openai
import json
import os
from typing import Dict, Optional
import requests

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper function to fetch market data
def fetch_helper_data() -> Dict:
    try:
        response = requests.get("https://api.example.com/market-data")
        data = response.json()
        return {
            "inflation_rate": data.get("inflation", 1.03),
            "material_cost": data.get("materials", 500),
            "labor_rate": data.get("labor", 250)
        }
    except Exception:
        return {"inflation_rate": 1.03, "material_cost": 500, "labor_rate": 250}

# Cost estimation class
class CostEstimator:
    def analyze_and_estimate(self, task_description: str) -> Dict:
        helper_data = fetch_helper_data()
        prompt = f"""
        As a cautious pricing engineer, analyze the following text to extract main tasks, detect any contradictions, and estimate the cost based on deep understanding. Use the helper data: inflation_rate={helper_data['inflation_rate']}, material_cost={helper_data['material_cost']}, labor_rate={helper_data['labor_rate']}. Return the result in JSON format with fields:
        - tasks: list of tasks
        - contradictions: list of contradictions if any
        - total_cost: total cost (USD)
        - reasoning: explanation of the estimation logic
        Text: {task_description}
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        try:
            return json.loads(response.choices[0].text.strip())
        except json.JSONDecodeError:
            return {"error": "Invalid response format from OpenAI"}

    def compare_with_bid(self, task_description: str, actual_bid: float) -> Dict:
        prompt = f"""
        As a cautious pricing engineer, compare the actual bid with the previous estimate based on the text: {task_description}. Actual bid: {actual_bid} USD. Return the result in JSON format with fields:
        - estimated_cost: estimated cost
        - actual_bid: actual bid
        - deviation_percent: deviation percentage
        - recommendation: recommendation for adjustment if needed
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.5
        )
        try:
            return json.loads(response.choices[0].text.strip())
        except json.JSONDecodeError:
            return {"error": "Invalid response format from OpenAI"}

    def update_with_user_input(self, task_description: str, user_input: str) -> Dict:
        prompt = f"""
        As a cautious pricing engineer, update the previous estimate for the text: {task_description} based on the comments: {user_input}. Return the result in JSON format with fields:
        - updated_cost: updated cost
        - reasoning: explanation of the adjustment
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.6
        )
        try:
            return json.loads(response.choices[0].text.strip())
        except json.JSONDecodeError:
            return {"error": "Invalid response format from OpenAI"}

# Streamlit app
st.title("CostimAIze - Smart Pricing Engineer")
st.image("assets/logo.png", use_column_width=True)

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
