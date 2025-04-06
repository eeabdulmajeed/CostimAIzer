import openai
import json
import os
from typing import Dict, Optional
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")

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