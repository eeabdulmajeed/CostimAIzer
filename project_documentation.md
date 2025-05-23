# CostimAIze Project Documentation

## Project Overview
- **Project Name**: CostimAIze
- **Project ID**: CostimAIze-Project-2025
- **Objective**: Develop an AI-powered application that acts as a cautious and intelligent pricing engineer for project cost estimation, bid analysis, and historical price archiving.
- **Platform**: Streamlit Cloud
- **AI Model**: OpenAI (gpt-3.5-turbo)
- **Start Date**: April 2025

## Core Requirements
### General Requirements
- The application must act as an intelligent and cautious pricing engineer, capable of learning and evolving.
- All AI functions must use real, advanced intelligence (OpenAI) and not local or mock logic.
- All logic layers (including cautious pricing) must rely on OpenAI for decision-making.
- The application provides three main services:
  1. **Cost Estimation**: Generate a cost estimation report for a project based on a scope of work.
  2. **Bid Analysis**: Analyze bid prices based on a pricing table.
  3. **Historical Price Archiving**: Archive historical prices for 90 days.

### Cost Estimation Requirements
- No pre-programmed guidance for learning or development, except for the cautious pricing logic.
- Consider all direct and indirect costs, including global market conditions (inflation, material prices, interest rates, etc.).
- The AI determines influencing factors and their impact percentages independently.
- Estimation is based on executing the scope of work as a single, integrated project.
- The AI must continue working even if helper data is unavailable, with the option to apply reserve percentages if deemed necessary.
- Helper data (historical prices, global market conditions, global news) is provided, but the AI decides whether to use it, how to use it, and its impact.

### Cautious Pricing Logic
- Retain results for 90 days without changes.
- Run cost estimation 100 times using Monte Carlo simulation before finalizing the result.
- Ensure the estimated price is logical and aligns with the scope of work, project conditions, and global market conditions.
- Avoid randomness or hallucination in pricing.
- The AI itself decides the final result and the reasoning behind it, without local rules.

### Interface Requirements
- **Dashboard**: Main page displaying attractive, auto-updated statistics about the application's activities.
- Display the three main services (Cost Estimation, Bid Analysis, Historical Price Archiving) for user selection.
- Navigate to service-specific pages upon selection.
- **Cost Estimation Interface**:
  1. Window to upload scope of work files (Word, Excel, PDF, multiple files allowed).
  2. Real-time analysis of the scope, displaying contradictions in separate windows.
  3. Windows for user responses to contradictions.
  4. Final cost estimation report window.

## Current File Structure
## Current Code
### app.py
```python
import streamlit as st
import openai
import json
import os
from typing import Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from docx import Document
import PyPDF2

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("costimaize.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_description TEXT,
        total_cost REAL,
        timestamp REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS bids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_description TEXT,
        actual_bid REAL,
        timestamp REAL
    )''')
    conn.commit()
    return conn

# Helper function to fetch market data using OpenAI
def fetch_helper_data() -> Dict:
    """Use OpenAI to fetch and analyze global market data."""
    prompt = """
    As a cautious pricing engineer, fetch and analyze current global market data relevant to project cost estimation. Provide the following information in JSON format:
    - inflation_rate: current global inflation rate
    - material_cost: average cost of construction materials (USD)
    - labor_rate: average labor rate (USD/hour)
    - global_news: a brief summary of global economic news
    If you cannot fetch the data, provide reasonable estimates based on your knowledge and explain your reasoning.
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a cautious pricing engineer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return {
            "inflation_rate": 1.03,
            "material_cost": 500,
            "labor_rate": 250,
            "global_news": "Unable to fetch global news."
        }

# Cost estimation class with cautious pricing logic using OpenAI
class CostEstimator:
    def __init__(self):
        # Initialize database
        self.conn = init_db()
        # Dictionary to store historical prices for 90 days
        if "price_history" not in st.session_state:
            st.session_state.price_history = {}
        self.price_history = st.session_state.price_history

    def read_file(self, uploaded_file) -> str:
        """Read content from uploaded files (Word, Excel, PDF)."""
        content = ""
        try:
            if uploaded_file.name.endswith(".docx"):
                doc = Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                content = df.to_string()
            elif uploaded_file.name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(uploaded_file)
                content = "\n".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            st.error(f"Failed to read file {uploaded_file.name}: {str(e)}")
        return content

    def validate_scope(self, task_description: str) -> Dict:
        """Analyze the scope of work for contradictions and extract main tasks using OpenAI."""
        prompt = f"""
        As a cautious pricing engineer, analyze the following scope of work to extract main tasks and detect any contradictions. Return the result in JSON format with fields:
        - tasks: list of tasks
        - contradictions: list of contradictions if any
        Text: {task_description}
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cautious pricing engineer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        try:
            return json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            return {"tasks": [], "contradictions": ["Failed to analyze scope of work"]}

    def estimate_cost_once(self, task_description: str, helper_data: Dict) -> Dict:
        """Run a single cost estimation using OpenAI."""
        prompt = f"""
        As a cautious pricing engineer, estimate the cost for the following text based on deep understanding. You may use the following helper data if you find it relevant: inflation_rate={helper_data['inflation_rate']}, material_cost={helper_data['material_cost']}, labor_rate={helper_data['labor_rate']}, global_news={helper_data['global_news']}. You decide whether to use this data, how to use it, and its impact on the cost. Include all direct and indirect costs, considering global market conditions, inflation, and project-specific factors. Return the result in JSON format with fields:
        - total_cost: total cost (USD)
        - reasoning: explanation of the estimation logic, including which helper data you used (if any) and why
        Text: {task_description}
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cautious pricing engineer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        try:
            return json.loads(response.choices[0].message.content.strip())
        except (json.JSONDecodeError, KeyError):
            return None

    def cautious_pricing(self, costs: list, historical_costs: list, task_description: str, helper_data: Dict) -> Dict:
        """Apply cautious pricing logic using OpenAI to ensure logical pricing."""
        if not costs:
            return {"error": "No valid cost estimates obtained from simulations"}

        # Prepare historical costs for OpenAI
        historical_costs_str = ", ".join([str(cost) for cost in historical_costs]) if historical_costs else "No historical costs available"

        prompt = f"""
        As a cautious pricing engineer, you have run 100 cost estimation simulations for the following task: {task_description}. The estimated costs are: {costs}. Historical costs for similar tasks are: {historical_costs_str}. Helper data available: inflation_rate={helper_data['inflation_rate']}, material_cost={helper_data['material_cost']}, labor_rate={helper_data['labor_rate']}, global_news={helper_data['global_news']}. Your task is to:
        1. Determine the final cost estimate, ensuring it is logical and avoids hallucination or randomness.
        2. Consider the scope of work, project conditions, global market conditions, and historical costs (if relevant).
        3. You decide how to use the helper data and historical costs, if at all.
        Return the result in JSON format with fields:
        - final_cost: the final cost estimate (USD)
        - reasoning: explanation of how you determined the final cost, including any adjustments for logical consistency
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cautious pricing engineer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        try:
            result = json.loads(response.choices[0].message.content.strip())
            return {
                "final_cost": result["final_cost"],
                "reasoning": result["reasoning"]
            }
        except (json.JSONDecodeError, KeyError):
            return {"error": "Failed to apply cautious pricing logic"}

    def analyze_and_estimate(self, task_description: str) -> Dict:
        """Estimate cost with cautious pricing logic and Monte Carlo simulation."""
        # Check if the task has a recent estimate (within 90 days)
        if task_description in self.price_history:
            entry = self.price_history[task_description]
            timestamp = entry["timestamp"]
            current_time = datetime.now().timestamp()
            if current_time - timestamp < 90 * 24 * 60 * 60:  # 90 days in seconds
                return {
                    "tasks": entry.get("tasks", []),
                    "contradictions": entry.get("contradictions", []),
                    "total_cost": entry["cost"],
                    "reasoning": "Retrieved from historical data (within 90 days)."
                }

        # Validate scope of work
        scope_analysis = self.validate_scope(task_description)
        contradictions = scope_analysis["contradictions"]
        tasks = scope_analysis["tasks"]

        # If contradictions exist, return them for user clarification
        if contradictions:
            return {
                "tasks": tasks,
                "contradictions": contradictions,
                "total_cost": None,
                "reasoning": "Contradictions detected, please resolve before estimating cost."
            }

        # Fetch market data
        helper_data = fetch_helper_data()

        # Monte Carlo simulation: Run 100 simulations
        costs = []
        reasonings = []
        for _ in range(100):
            result = self.estimate_cost_once(task_description, helper_data)
            if result is not None and "total_cost" in result:
                costs.append(result["total_cost"])
                reasonings.append(result["reasoning"])

        # Get historical costs for comparison
        historical_costs = [entry["cost"] for entry in self.price_history.values()]

        # Apply cautious pricing logic using OpenAI
        cautious_result = self.cautious_pricing(costs, historical_costs, task_description, helper_data)
        if "error" in cautious_result:
            return {"error": cautious_result["error"]}

        final_cost = cautious_result["final_cost"]
        reasoning = cautious_result["reasoning"]

        # Store the result in price history and database
        result = {
            "tasks": tasks,
            "contradictions": contradictions,
            "total_cost": final_cost,
            "reasoning": reasoning
        }
        self.price_history[task_description] = {
            "cost": final_cost,
            "tasks": tasks,
            "contradictions": contradictions,
            "timestamp": datetime.now().timestamp()
        }
        st.session_state.price_history = self.price_history

        # Save to database
        c = self.conn.cursor()
        c.execute("INSERT INTO projects (task_description, total_cost, timestamp) VALUES (?, ?, ?)",
                  (task_description, final_cost, datetime.now().timestamp()))
        self.conn.commit()

        return result

    def compare_with_bid(self, task_description: str, actual_bid: float) -> Dict:
        prompt = f"""
        As a cautious pricing engineer, compare the actual bid with the previous estimate based on the text: {task_description}. Actual bid: {actual_bid} USD. Return the result in JSON format with fields:
        - estimated_cost: estimated cost
        - actual_bid: actual bid
        - deviation_percent: deviation percentage
        - recommendation: recommendation for adjustment if needed
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cautious pricing engineer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        try:
            result = json.loads(response.choices[0].message.content.strip())
            # Save to database
            c = self.conn.cursor()
            c.execute("INSERT INTO bids (task_description, actual_bid, timestamp) VALUES (?, ?, ?)",
                      (task_description, actual_bid, datetime.now().timestamp()))
            self.conn.commit()
            return result
        except json.JSONDecodeError:
            return {"error": "Invalid response format from OpenAI"}

    def update_with_user_input(self, task_description: str, user_input: str) -> Dict:
        prompt = f"""
        As a cautious pricing engineer, update the previous estimate for the text: {task_description} based on the comments: {user_input}. Return the result in JSON format with fields:
        - updated_cost: updated cost
        - reasoning: explanation of the adjustment
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cautious pricing engineer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.6
        )
        try:
            return json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            return {"error": "Invalid response format from OpenAI"}

# Dashboard statistics from database
def get_dashboard_stats():
    conn = sqlite3.connect("costimaize.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM projects")
    total_projects = c.fetchone()[0] or 0
    c.execute("SELECT SUM(total_cost) FROM projects")
    total_cost_estimated = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM bids")
    bids_analyzed = c.fetchone()[0] or 0
    historical_prices_archived = len(st.session_state.get("price_history", {}))
    conn.close()
    return {
        "total_projects": total_projects,
        "total_cost_estimated": total_cost_estimated,
        "bids_analyzed": bids_analyzed,
        "historical_prices_archived": historical_prices_archived
    }

# Streamlit app with dashboard and service selection
st.title("CostimAIze - Smart Pricing Engineer")
st.image("assets/logo.png", use_column_width=True)

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    if st.button("Dashboard"):
        st.session_state.page = "dashboard"
    if st.button("Estimate Cost"):
        st.session_state.page = "estimate_cost"
    if st.button("Analyze Bids"):
        st.session_state.page = "analyze_bids"
    if st.button("Archive Historical Prices"):
        st.session_state.page = "archive_prices"

# Dashboard page
if st.session_state.page == "dashboard":
    st.header("Dashboard")
    stats = get_dashboard_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", stats["total_projects"])
    with col2:
        st.metric("Total Cost Estimated (USD)", stats["total_cost_estimated"])
    with col3:
        st.metric("Bids Analyzed", stats["bids_analyzed"])
    with col4:
        st.metric("Historical Prices Archived", stats["historical_prices_archived"])

    st.subheader("Our Services")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Estimate Cost", key="dashboard_estimate"):
            st.session_state.page = "estimate_cost"
    with col2:
        if st.button("Analyze Bids", key="dashboard_analyze"):
            st.session_state.page = "analyze_bids"
    with col3:
        if st.button("Archive Historical Prices", key="dashboard_archive"):
            st.session_state.page = "archive_prices"

# Estimate Cost page
elif st.session_state.page == "estimate_cost":
    st.header("Estimate Cost")
    
    # File upload for scope of work
    uploaded_files = st.file_uploader("Upload Scope of Work (Word, Excel, PDF)", type=["docx", "xlsx", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success("Files uploaded successfully!")

    # Analyze scope of work and show contradictions
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        st.subheader("Scope Analysis")
        estimator = CostEstimator()
        contradictions = []
        task_description = ""
        for uploaded_file in st.session_state.uploaded_files:
            content = estimator.read_file(uploaded_file)
            task_description += f"File: {uploaded_file.name}\n{content}\n"
            scope_analysis = estimator.validate_scope(content)
            contradictions.extend(scope_analysis["contradictions"])
            st.session_state.tasks = scope_analysis["tasks"]

        st.session_state.task_description = task_description

        # Display contradictions and allow user input
        if contradictions:
            st.subheader("Contradictions Found")
            for i, contradiction in enumerate(contradictions):
                st.write(f"Contradiction {i+1}: {contradiction}")
                user_response = st.text_input(f"Response to contradiction {i+1}:", key=f"contradiction_{i}")
                if user_response:
                    st.session_state[f"contradiction_response_{i}"] = user_response

        # Estimate cost after resolving contradictions
        if st.button("Proceed to Estimate Cost"):
            final_description = st.session_state.task_description
            if "contradiction_response_0" in st.session_state:
                final_description += f" with user responses: {st.session_state['contradiction_response_0']}"
            result = estimator.analyze_and_estimate(final_description)
            st.session_state.estimation_result = result
            st.session_state.page = "estimation_result"

# Estimation Result page
elif st.session_state.page == "estimation_result":
    st.header("Cost Estimation Report")
    if "estimation_result" in st.session_state:
        result = st.session_state.estimation_result
        if "error" in result:
            st.error(result["error"])
        else:
            st.write("Tasks:", result["tasks"])
            st.write("Contradictions:", result["contradictions"])
            st.write("Total Cost (USD):", result["total_cost"])
            st.write("Reasoning:", result["reasoning"])
    if st.button("Back to Dashboard"):
        st.session_state.page = "dashboard"

# Analyze Bids page
elif st.session_state.page == "analyze_bids":
    st.header("Analyze Bids")
    task_description = st.text_area("Enter task description for bid analysis:", "Build a small house with 3 rooms in a high-inflation area")
    actual_bid = st.number_input("Enter actual bid (USD):", min_value=0.0)
    if st.button("Analyze Bid"):
        estimator = CostEstimator()
        analysis = estimator.compare_with_bid(task_description, actual_bid)
        if "error" in analysis:
            st.error(analysis["error"])
        else:
            st.write("Estimated Cost (USD):", analysis["estimated_cost"])
            st.write("Actual Bid (USD):", analysis["actual_bid"])
            st.write("Deviation (%):", analysis["deviation_percent"])
            st.write("Recommendation:", analysis["recommendation"])
    if st.button("Back to Dashboard"):
        st.session_state.page = "dashboard"

# Archive Historical Prices page
elif st.session_state.page == "archive_prices":
    st.header("Archive Historical Prices")
    st.write("Historical prices are archived for 90 days.")
    estimator = CostEstimator()
    if estimator.price_history:
        for task, data in estimator.price_history.items():
            st.write(f"Task: {task}, Cost: {data['cost']} USD, Timestamp: {datetime.fromtimestamp(data['timestamp'])}")
    else:
        st.write("No historical prices available.")
    if st.button("Back to Dashboard"):
        st.session_state.page = "dashboard"
