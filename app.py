import streamlit as st
import openai
import json
import os
from typing import Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from docx import Document
import PyPDF2
import logging
import requests
import time

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxies = None

# Check if API key is set
if not openai.api_key:
    st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Configure logging
logging.basicConfig(filename='execution_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Test OpenAI API
try:
    test_response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "Hello, can you respond?"}],
        max_tokens=10
    )
    logger.info("OpenAI API test successful: %s", test_response.choices[0].message.content)
except Exception as e:
    logger.error("OpenAI API test failed: %s", str(e))

# Market data functions
def get_market_news() -> str:
    """Fetch latest market news using NewsAPI (simulated if no API key)."""
    api_key = os.getenv("NEWSAPI_KEY")
    if api_key:
        try:
            url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={api_key}"
            response = requests.get(url, timeout=10)
            news = response.json().get("articles", [])
            return "\n".join([article["title"] for article in news[:5]])
        except Exception as e:
            logger.error("Error fetching market news: %s", str(e))
    return "Simulated news: Global markets stable, slight material price increase."

def get_material_prices() -> Dict:
    """Fetch material prices using Alpha Vantage (simulated if no API key)."""
    api_key = os.getenv("ALPHAVANTAGE_KEY")
    if api_key:
        try:
            url = f"https://www.alphavantage.co/query?function=COMMODITY&symbol=STEEL&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            return {"steel": data.get("price", 700)}
        except Exception as e:
            logger.error("Error fetching material prices: %s", str(e))
    return {"steel": 700, "concrete": 300}

def get_inflation_rates() -> float:
    """Fetch inflation rates (simulated if no real source)."""
    return 1.06

def get_interest_rates() -> float:
    """Fetch interest rates (simulated if no real source)."""
    return 0.045

# Cost estimation class
class CostEstimator:
    def __init__(self):
        if "price_history" not in st.session_state:
            st.session_state.price_history = {}
        if "cost_estimates" not in st.session_state:
            st.session_state.cost_estimates = []
        self.price_history = st.session_state.price_history
        self.cost_estimates = st.session_state.cost_estimates

        if "projects" not in st.session_state:
            st.session_state.projects = []
        if "bids" not in st.session_state:
            st.session_state.bids = []

    def read_file(self, uploaded_file) -> str:
        """Read content from uploaded files."""
        logger.info("Reading file: %s", uploaded_file.name)
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
            logger.error("Error reading file %s: %s", uploaded_file.name, str(e))
        if not content.strip():
            st.warning(f"No content extracted from {uploaded_file.name}.")
            logger.warning("No content extracted from %s", uploaded_file.name)
        logger.info("File content for %s: %s", uploaded_file.name, content[:100])
        return content

    def validate_scope(self, task_description: str) -> Dict:
        """Analyze scope of work (ScopeGPT)."""
        if not task_description.strip():
            logger.warning("Empty task description received.")
            return {"tasks": [], "contradictions": ["No content provided"], "missing_details": []}

        logger.info("Analyzing scope: %s", task_description[:100])
        prompt = f"""
        You are ScopeGPT, a cautious pricing engineer. Analyze the following scope of work to extract ALL main tasks (direct and indirect) and detect contradictions or missing details. Direct tasks include material procurement, labor, installation, testing, commissioning. Indirect tasks include safety, security, shipping, financing. Contradictions are inconsistencies (e.g., conflicting timelines). Missing details are critical info not provided (e.g., location, timeline). **Return a valid JSON object starting with '{{' and ending with '}}'. If you encounter any error or cannot analyze the scope, return a default JSON: {{\"tasks\": [], \"contradictions\": [\"Scope analysis failed due to unexpected error\"], \"missing_details\": []}}. Do not include any additional text outside the JSON.**
        Return JSON with:
        - tasks: list of all tasks
        - contradictions: list of contradictions (empty if none)
        - missing_details: list of missing details (empty if none)
        Scope: {task_description}
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner("Analyzing scope..."):
                    response = openai.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1500,
                        temperature=0.5
                    )
                    response_content = response.choices[0].message.content.strip()
                    logger.info("ScopeGPT response (attempt %d): %s", attempt + 1, response_content)
                    if not response_content.strip() or response_content.isspace():
                        if attempt == max_retries - 1:
                            return {"tasks": [], "contradictions": ["Empty response from OpenAI"], "missing_details": []}
                        continue
                    if not response_content.startswith("{") or not response_content.endswith("}"):
                        if attempt == max_retries - 1:
                            return {"tasks": [], "contradictions": ["Invalid JSON format from OpenAI"], "missing_details": []}
                        continue
                    result = json.loads(response_content)
                    return result
            except json.JSONDecodeError as e:
                logger.error("JSON decode error (attempt %d): %s", attempt + 1, str(e))
                if attempt == max_retries - 1:
                    return {"tasks": [], "contradictions": ["Invalid JSON response from OpenAI"], "missing_details": []}
            except Exception as e:
                logger.error("Error in validate_scope (attempt %d): %s", attempt + 1, str(e))
                if attempt == max_retries - 1:
                    return {"tasks": [], "contradictions": [f"Analysis failed: {str(e)}"], "missing_details": []}
        return {"tasks": [], "contradictions": ["Analysis failed after multiple attempts"], "missing_details": []}

    def estimate_cost_once(self, task_description: str) -> Dict:
        """Estimate cost (MarketGPT) without helper data."""
        logger.info("Estimating cost for: %s", task_description[:100])
        prompt = f"""
        You are MarketGPT, a cautious pricing engineer. Estimate the cost for the scope: {task_description}. Identify ALL direct costs (materials, labor, installation, testing) and indirect costs (safety, shipping, financing). Provide unit costs and totals based on your analysis of the scope and optional market data (news: {get_market_news()}, prices: {get_material_prices()}, inflation: {get_inflation_rates()}, interest: {get_interest_rates()}). You decide if and how to use this data. Explain your reasoning. **Return a valid JSON object starting with '{{' and ending with '}}'. If you encounter any error or cannot estimate the cost, return a default JSON: {{\"total_cost\": null, \"cost_breakdown\": {{}}, \"reasoning\": \"Estimation failed due to insufficient data\"}}. Do not include any additional text outside the JSON.**
        Return JSON with:
        - total_cost: total in USD
        - cost_breakdown: direct and indirect costs with unit costs
        - reasoning: explanation
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner("Estimating cost..."):
                    response = openai.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.7
                    )
                    response_content = response.choices[0].message.content.strip()
                    logger.info("MarketGPT response (attempt %d): %s", attempt + 1, response_content)
                    if not response_content.strip() or response_content.isspace():
                        if attempt == max_retries - 1:
                            return {"total_cost": None, "cost_breakdown": {}, "reasoning": "Empty response from OpenAI"}
                        continue
                    if not response_content.startswith("{") or not response_content.endswith("}"):
                        if attempt == max_retries - 1:
                            return {"total_cost": None, "cost_breakdown": {}, "reasoning": "Invalid JSON format from OpenAI"}
                        continue
                    result = json.loads(response_content)
                    return result
            except json.JSONDecodeError as e:
                logger.error("Error in estimate_cost_once (attempt %d): %s", attempt + 1, str(e))
                if attempt == max_retries - 1:
                    return {"total_cost": None, "cost_breakdown": {}, "reasoning": "Invalid JSON response from OpenAI"}
            except Exception as e:
                logger.error("Error in estimate_cost_once (attempt %d): %s", attempt + 1, str(e))
                if attempt == max_retries - 1:
                    return {"total_cost": None, "cost_breakdown": {}, "reasoning": f"Estimation failed: {str(e)}"}
        return {"total_cost": None, "cost_breakdown": {}, "reasoning": "Estimation failed after multiple attempts"}

    def validate_cost(self, costs: list, historical_costs: list, task_description: str) -> Dict:
        """Validate costs (ValidatorGPT) without helper data."""
        if not costs:
            logger.warning("No valid cost estimates received.")
            return {"is_valid": False, "discrepancy": "No cost estimates provided", "clarification_request": ""}

        historical_costs_str = ", ".join([str(cost) for cost in historical_costs]) if historical_costs else "None"
        logger.info("Validating costs: %s", costs)
        prompt = f"""
        You are ValidatorGPT, a cautious pricing engineer. Validate the costs {costs} for the scope: {task_description}. Historical costs: {historical_costs_str}. Optional market data: news: {get_market_news()}, prices: {get_material_prices()}, inflation: {get_inflation_rates()}, interest: {get_interest_rates()}. Check logical consistency based on the scope and market conditions. Identify discrepancies and request clarification if needed. Return JSON with:
        - is_valid: boolean
        - discrepancy: description (or 'No discrepancy')
        - clarification_request: detailed request (or empty string)
        """
        try:
            with st.spinner("Validating costs..."):
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                result = json.loads(response.choices[0].message.content.strip())
                logger.info("ValidatorGPT result: %s", json.dumps(result))
                return result
        except Exception as e:
            logger.error("Error in validate_cost: %s", str(e))
            return {"is_valid": False, "discrepancy": f"Validation failed: {str(e)}", "clarification_request": ""}

    def request_clarification(self, task_description: str, clarification_request: str) -> Dict:
        """Request clarification (MarketGPT) without helper data."""
        logger.info("Requesting clarification for: %s", task_description[:100])
        prompt = f"""
        You are MarketGPT. Clarify the cost estimate for the scope: {task_description}. Clarification request: {clarification_request}. Optional market data: news: {get_market_news()}, prices: {get_material_prices()}, inflation: {get_inflation_rates()}, interest: {get_interest_rates()}. Provide a detailed breakdown and reasoning. Return JSON with:
        - total_cost: revised cost in USD
        - cost_breakdown: direct and indirect costs with unit costs
        - reasoning: explanation
        """
        try:
            with st.spinner("Requesting clarification..."):
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                result = json.loads(response.choices[0].message.content.strip())
                logger.info("MarketGPT clarification: %s", json.dumps(result))
                return result
        except Exception as e:
            logger.error("Error in request_clarification: %s", str(e))
            return {"total_cost": None, "cost_breakdown": {}, "reasoning": f"Clarification failed: {str(e)}"}

    def compare_with_bid(self, task_description: str, actual_bid: float) -> Dict:
        """Compare the estimated cost with the actual bid using OpenAI (BidGPT)."""
        prompt = f"""
        You are BidGPT, a cautious pricing engineer specialized in bid analysis. Compare the actual bid with the previous estimate based on the text: {task_description}. Actual bid: {actual_bid} USD. Return the result in JSON format with fields:
        - estimated_cost: estimated cost
        - actual_bid: actual bid
        - deviation_percent: deviation percentage
        - recommendation: recommendation for adjustment if needed
        """
        try:
            with st.spinner("Analyzing bid..."):
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are BidGPT, a cautious pricing engineer specialized in bid analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.5,
                    timeout=30
                )
                result = json.loads(response.choices[0].message.content.strip())
                return result
        except Exception as e:
            logger.error("Error in compare_with_bid: %s", str(e))
            return {"error": "Failed to analyze bid"}

    def update_with_user_input(self, task_description: str, user_input: str) -> Dict:
        """Update the previous estimate based on user input using OpenAI."""
        prompt = f"""
        You are a cautious pricing engineer. Update the previous estimate for the text: {task_description} based on the comments: {user_input}. Return the result in JSON format with fields:
        - updated_cost: updated cost
        - reasoning: explanation of the adjustment
        """
        try:
            with st.spinner("Updating estimate based on user input..."):
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a cautious pricing engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.6,
                    timeout=30
                )
                return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error("Error in update_with_user_input: %s", str(e))
            return {"error": "Failed to update estimate"}

    def coordinate_results(self, task_description: str, scope_result: Dict, market_result: Dict, validator_result: Dict, bid_result: Dict, dialogue_log: Dict) -> Dict:
        """Coordinate results (CoordinatorGPT)."""
        logger.info("Coordinating results for: %s", task_description[:100])
        prompt = f"""
        You are CoordinatorGPT, a cautious pricing engineer. Coordinate results for the scope: {task_description}:
        - ScopeGPT: {scope_result}
        - MarketGPT: {market_result}
        - ValidatorGPT: {validator_result}
        - BidGPT: {bid_result}
        - Dialogue Log: {dialogue_log}
        Determine the most logical final cost based on all data, resolving discrepancies with full freedom. Return JSON with:
        - final_cost: final cost in USD
        - cost_breakdown: direct and indirect costs with unit costs
        - reasoning: detailed explanation
        """
        try:
            with st.spinner("Coordinating results..."):
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                result = json.loads(response.choices[0].message.content.strip())
                logger.info("CoordinatorGPT result: %s", json.dumps(result))
                return result
        except Exception as e:
            logger.error("Error in coordinate_results: %s", str(e))
            return {"final_cost": None, "cost_breakdown": {}, "reasoning": f"Coordination failed: {str(e)}"}

    def analyze_and_estimate_multi_gpt(self, task_description: str) -> Dict:
        """Estimate cost using Multi-GPT Architecture."""
        logger.info("Starting Multi-GPT estimation for: %s", task_description[:100])
        if task_description in self.price_history:
            entry = self.price_history[task_description]
            if datetime.now().timestamp() - entry["timestamp"] < 90 * 24 * 60 * 60:
                logger.info("Retrieved from history: %s", task_description)
                return {
                    "tasks": entry.get("tasks", []),
                    "contradictions": entry.get("contradictions", []),
                    "missing_details": entry.get("missing_details", []),
                    "total_cost": entry["cost"],
                    "cost_breakdown": entry.get("cost_breakdown", {}),
                    "reasoning": "Retrieved from history (within 90 days)."
                }

        scope_result = self.validate_scope(task_description)
        contradictions = scope_result["contradictions"]
        tasks = scope_result["tasks"]
        missing_details = scope_result.get("missing_details", [])

        if contradictions:
            logger.warning("Contradictions found: %s", contradictions)
            return {
                "tasks": tasks,
                "contradictions": contradictions,
                "missing_details": missing_details,
                "total_cost": None,
                "cost_breakdown": {},
                "reasoning": "Contradictions detected, resolve before estimation."
            }

        if missing_details:
            logger.info("Missing details: %s", missing_details)

        costs = []
        reasonings = []
        cost_breakdowns = []
        for i in range(10):
            logger.info("Running simulation %d/10", i + 1)
            result = self.estimate_cost_once(task_description)
            if result["total_cost"] is not None:
                costs.append(result["total_cost"])
                reasonings.append(result["reasoning"])
                cost_breakdowns.append(result["cost_breakdown"])

        market_result = {
            "total_cost": sum(costs) / len(costs) if costs else 0,
            "cost_breakdown": cost_breakdowns[-1] if cost_breakdowns else {},
            "reasoning": reasonings[-1] if reasonings else "Estimated based on simulations."
        }

        historical_costs = [entry["cost"] for entry in self.price_history.values()]
        validator_result = self.validate_cost(costs, historical_costs, task_description)

        dialogue_log = {}
        if not validator_result["is_valid"]:
            clarification_request = validator_result["clarification_request"]
            if clarification_request:
                market_clarification = self.request_clarification(task_description, clarification_request)
                dialogue_log["ValidatorGPT_to_MarketGPT"] = [{"request": clarification_request, "response": market_clarification}]
                market_result = market_clarification
                validator_result["is_valid"] = True

        bid_result = {"estimated_cost": market_result["total_cost"], "actual_bid": None, "deviation_percent": None, "recommendation": "No bid provided."}
        coordinated_result = self.coordinate_results(task_description, scope_result, market_result, validator_result, bid_result, dialogue_log)

        result = {
            "tasks": tasks,
            "contradictions": contradictions,
            "missing_details": missing_details,
            "total_cost": coordinated_result["final_cost"],
            "cost_breakdown": coordinated_result["cost_breakdown"],
            "reasoning": coordinated_result["reasoning"]
        }

        # Save the cost estimate only if total_cost is not None
        if result["total_cost"] is not None:
            self.cost_estimates.append({
                "task_description": task_description,
                "total_cost": coordinated_result["final_cost"],
                "cost_breakdown": coordinated_result["cost_breakdown"],
                "tasks": tasks,
                "contradictions": contradictions,
                "missing_details": missing_details,
                "timestamp": datetime.now().timestamp()
            })
            st.session_state.cost_estimates = self.cost_estimates

        st.session_state.projects.append({
            "task_description": task_description,
            "total_cost": coordinated_result["final_cost"],
            "timestamp": datetime.now().timestamp()
        })
        logger.info("Estimation completed: %s", json.dumps(result))
        return result

# Dashboard stats
def get_dashboard_stats():
    stats = {
        "total_projects": len(st.session_state.get("projects", [])),
        "total_cost_estimated": sum(project["total_cost"] for project in st.session_state.get("projects", []) if project["total_cost"] is not None),
        "bids_analyzed": len(st.session_state.get("bids", [])),
        "cost_estimates_archived": len(st.session_state.get("cost_estimates", []))
    }
    logger.info("Dashboard stats: %s", stats)
    return stats

# Streamlit app
st.title("CostimAIze - Smart Pricing Engineer")
st.image("assets/logo.png", use_column_width=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

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
        st.metric("Cost Estimates Archived", stats["cost_estimates_archived"])

    st.subheader("Services")
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

elif st.session_state.page == "estimate_cost":
    st.header("Estimate Cost")
    st.subheader("Optional Information")
    project_location = st.text_input("Project Location:")
    project_timeline = st.text_input("Project Timeline:")
    project_type = st.selectbox("Project Type:", ["Capital", "Unit Price"])
    extra_info = st.text_area("Other Information:")

    uploaded_files = st.file_uploader("Upload Scope of Work", type=["docx", "xlsx", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success("Files uploaded successfully!")

    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        estimator = CostEstimator()
        task_description = ""
        for uploaded_file in st.session_state.uploaded_files:
            content = estimator.read_file(uploaded_file)
            task_description += f"File: {uploaded_file.name}\n{content}\n"
        if project_location:
            task_description += f"Location: {project_location}\n"
        if project_timeline:
            task_description += f"Timeline: {project_timeline}\n"
        if project_type:
            task_description += f"Type: {project_type}\n"
        if extra_info:
            task_description += f"Extra Info: {extra_info}\n"
        st.session_state.task_description = task_description

        if st.button("Proceed to Estimate Cost", key="proceed_button", disabled=st.session_state.is_processing):
            if not st.session_state.is_processing:
                st.session_state.is_processing = True
                time.sleep(0.1)
                st.info("Processing... Please wait.")
                try:
                    with st.spinner("Estimating cost..."):
                        result = estimator.analyze_and_estimate_multi_gpt(task_description)
                        st.session_state.estimation_result = result
                        st.session_state.page = "estimation_result"
                except Exception as e:
                    st.error(f"Error during estimation: {str(e)}")
                    logger.error("Error during estimation: %s", str(e))
                finally:
                    st.session_state.is_processing = False
            else:
                st.warning("Processing is already in progress. Please wait.")

elif st.session_state.page == "estimation_result":
    st.header("Cost Estimation Report")
    if "estimation_result" in st.session_state:
        result = st.session_state.estimation_result
        if result["contradictions"]:
            st.subheader("Contradictions")
            for i, contradiction in enumerate(result["contradictions"]):
                st.write(f"Contradiction {i+1}: {contradiction}")
                response = st.text_input(f"Response to contradiction {i+1}:", key=f"contra_{i}")
                if response:
                    st.session_state[f"contra_response_{i}"] = response
            if st.button("Re-estimate"):
                final_description = st.session_state.task_description
                for i, _ in enumerate(result["contradictions"]):
                    if f"contra_response_{i}" in st.session_state:
                        final_description += f"\nResponse {i+1}: {st.session_state[f'contra_response_{i}']}"
                estimator = CostEstimator()
                result = estimator.analyze_and_estimate_multi_gpt(final_description)
                st.session_state.estimation_result = result
                st.session_state.page = "estimation_result"
        else:
            st.subheader("Tasks")
            for i, task in enumerate(result["tasks"]):
                st.write(f"Task {i+1}: {task}")
            if result["missing_details"]:
                st.subheader("Missing Details")
                for i, detail in enumerate(result["missing_details"]):
                    st.write(f"Missing Detail {i+1}: {detail}")
            if result["total_cost"] is not None:
                st.subheader("Total Cost (USD)")
                st.write(result["total_cost"])
                st.subheader("Cost Breakdown")
                if "direct_costs" in result["cost_breakdown"]:
                    st.write("Direct Costs:")
                    for key, value in result["cost_breakdown"]["direct_costs"].items():
                        st.write(f"{key}: {value}")
                if "indirect_costs" in result["cost_breakdown"]:
                    st.write("Indirect Costs:")
                    for key, value in result["cost_breakdown"]["indirect_costs"].items():
                        st.write(f"{key}: {value}")
                st.subheader("Reasoning")
                st.write(result["reasoning"])
            if st.button("View Execution Log"):
                with open("execution_log.txt", "r") as f:
                    st.text(f.read())
        if st.button("Back to Dashboard"):
            st.session_state.page = "dashboard"

elif st.session_state.page == "analyze_bids":
    st.header("Analyze Bids")
    task_description = st.text_area("Task Description:")
    actual_bid = st.number_input("Actual Bid (USD):", min_value=0.0)
    if st.button("Analyze Bid"):
        estimator = CostEstimator()
        with st.spinner("Analyzing bid..."):
            analysis = estimator.compare_with_bid(task_description, actual_bid)
        if "error" in analysis:
            st.error(analysis["error"])
        else:
            st.write("Estimated Cost (USD):", analysis["estimated_cost"])
            st.write("Actual Bid (USD):", analysis["actual_bid"])
            st.write("Deviation (%):", analysis["deviation_percent"])
            st.write("Recommendation:", analysis["recommendation"])
            st.session_state.bids.append({
                "task_description": task_description,
                "actual_bid": actual_bid,
                "timestamp": datetime.now().timestamp()
            })
    if st.button("Back to Dashboard"):
        st.session_state.page = "dashboard"

elif st.session_state.page == "archive_prices":
    st.header("Archive Historical Prices")
    estimator = CostEstimator()
    if estimator.price_history:
        for task, data in estimator.price_history.items():
            st.write(f"Task: {task}, Cost: {data['cost']} USD, Timestamp: {datetime.fromtimestamp(data['timestamp'])}")
    else:
        st.write("No historical prices available.")
    if st.button("Back to Dashboard"):
        st.session_state.page = "dashboard"