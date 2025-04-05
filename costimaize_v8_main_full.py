
import time
import openai
import os
import csv
import hashlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class CostimAIzeOrchestrator:
    def __init__(self):
        self.historical_data = []
        self.context = {}
        self.historical_contexts = []
        self.cost_estimator = CautiousCostEstimator()
        self.previous_hash = None
        self.cached_result = None
openai.api_key = os.getenv("OPENAI_API_KEY")

    def extract_text(self, file):
        try:
            content = file.read()
            file_hash = hashlib.md5(content).hexdigest()
            self.context["file_hash"] = file_hash
            if file.type == "text/plain":
                return content.decode("utf-8")
            else:
                return "Content extracted successfully from non-text file..."
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def validate_sow(self, text):
        prompt = '''
        As an experienced pricing engineer, review this scope of work text and determine if it has enough detail for a cost estimate.
        Respond with 'Yes' if analyzable, 'No' with a reason if not.
        Text: "{text}"
        '''
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(text=text[:1000])}],
                max_tokens=50
            )
            return "Yes" in response.choices[0].message["content"]
        except Exception:
            return len(text.strip()) > 0

    def analyze_scope(self, text):
        prompt = '''
        As an experienced pricing engineer, analyze this scope of work and identify key elements relevant to cost estimation.
        Present your findings in a structure you deem appropriate based on your expertise.
        Text: "{text}"
        '''
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(text=text[:2000])}],
                max_tokens=500
            )
            return response.choices[0].message["content"]
        except Exception:
            return "Unable to analyze scope due to insufficient data or error"

    def detect_inquiries(self, text):
        prompt = '''
        As an experienced pricing engineer, identify any inconsistencies, gaps, or uncertainties in this scope of work.
        List your inquiries with a brief description and indicate if they are critical (True/False) based on your judgment.
        Text: "{text}"
        '''
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(text=text[:2000])}],
                max_tokens=500
            )
            return response.choices[0].message["content"].splitlines()
        except Exception:
            return ["Unable to detect inquiries due to error"]

    def analyze_market(self, scope_insights):
        prompt = '''
        As an experienced pricing engineer, assess the current global market conditions relevant to this project scope: {scope_insights}.
        Include factors you consider important (e.g., material prices, economic trends) and summarize in your preferred format.
        '''
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(scope_insights=scope_insights)}],
                max_tokens=300
            )
            return response.choices[0].message["content"]
        except Exception:
            return "No market data available due to error"

    def understand_historical_prices(self):
        if not self.historical_data:
            return "No historical data available for analysis"
        
        df = pd.DataFrame(self.historical_data)
        prompt = '''
        As an experienced pricing engineer, analyze this historical pricing data: {data}.
        Provide insights, trends, or observations you deem relevant.
        '''
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(data=df.to_dict(orient='records')[:50])}],
                max_tokens=300
            )
            return response.choices[0].message["content"]
        except Exception:
            return "Historical data present but not analyzed due to error"

    def run_estimation(self, file, context):
        text = self.extract_text(file)
        if not self.validate_sow(text):
            return "Scope of work is not analyzable"
        
        scope_insights = self.analyze_scope(text)
        market_data = self.analyze_market(scope_insights)
        historical_insights = self.understand_historical_prices()
        inquiries = self.detect_inquiries(text)

        self.set_context(**context, inquiry_responses=inquiries)
        features = self._extract_features(scope_insights)
        cost_estimate = self.cost_estimator.estimate(features) if features else "No estimate possible"
        
        return {
            "scope_insights": scope_insights,
            "market_data": market_data,
            "historical_insights": historical_insights,
            "inquiries": inquiries,
            "estimated_cost": cost_estimate
        }

    def _extract_features(self, scope_insights):
        try:
            return {"dynamic_feature": scope_insights} if scope_insights else None
        except Exception:
            return None

    def archive_historical_prices(self, file, metadata=None):
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file, engine="openpyxl")
            else:
                return "Unsupported file format."

            df = df.dropna()
            self.historical_data = df.to_dict(orient='list')
            if metadata:
                self.historical_contexts.append(metadata)
            self.cost_estimator.train(self.historical_data)
            return f"Data archived. Records: {len(df)}"
        except Exception as e:
            return f"Error processing historical data: {str(e)}"

    def set_context(self, **kwargs):
        self.context.update(kwargs)

    def analyze_bid(self, sow_file, contractor_data):
        text = self.extract_text(sow_file)
        if not self.validate_sow(text):
            return {"error": "Invalid Scope of Work"}
        
        scope_insights = self.analyze_scope(text)
        market_data = self.analyze_market(scope_insights)
        historical_insights = self.understand_historical_prices()

        prompt = '''
        As a cautious pricing engineer, analyze this bid: {bid_data}.
        Consider the scope insights: {scope_insights}, market data: {market_data}, and historical insights: {historical_insights}.
        Provide your analysis and verdict in your preferred format.
        '''
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(bid_data=contractor_data, scope_insights=scope_insights, market_data=market_data, historical_insights=historical_insights)}],
                max_tokens=500
            )
            return response.choices[0].message["content"]
        except Exception:
            return "Error in bid analysis"

    def generate_report(self, result):
        return "Report generation in progress based on: " + str(result)

class CautiousCostEstimator:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.caution_factor = 1.0
        self.trained = False
        self.feedback_history = []

    def train(self, historical_data):
        if not historical_data:
            return
        df = pd.DataFrame(historical_data)
        features = df.select_dtypes(include=[np.number]).columns
        if len(features) == 0:
            return
        X = df[features]
        y = X.pop(features[0]) if 'actual_cost' not in df else df['actual_cost']
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        errors = np.abs(predictions - y) if len(y) > 0 else np.array([0])
        self.caution_factor = 1 + (errors.mean() / y.mean() if y.mean() != 0 else 1.0)
        self.trained = True

    def estimate(self, features):
        if not self.trained or not features:
            return "Estimation not possible due to lack of data or training"
        try:
            X = pd.DataFrame([features])
            prediction = self.model.predict(X)[0] * self.caution_factor
            if self.feedback_history:
                adjustment = sum(f for f in self.feedback_history if f) / len(self.feedback_history) if self.feedback_history else 0
                prediction *= (1 + adjustment)
            return f"Estimated Cost: SAR {prediction:,.2f}"
        except Exception:
            return "Error in cost estimation"

    def update_feedback(self, feedback):
        self.feedback_history.append(feedback)
