import openai
import json
import os
from docx import Document
import PyPDF2
import pandas as pd
import logging

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def read_file(uploaded_file) -> str:
    """Read content from uploaded files."""
    content = ""
    try:
        if uploaded_file.endswith(".docx"):
            doc = Document(uploaded_file)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            content = df.to_string()
        elif uploaded_file.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            content = "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        logger.error("Error reading file %s: %s", uploaded_file, str(e))
    return content

def train_on_historical_data(scope_file: str, cost_file: str) -> Dict:
    """Train the AI on historical scope and cost data."""
    scope_content = read_file(scope_file)
    cost_content = read_file(cost_file)
    logger.info("Training on scope: %s, cost: %s", scope_content[:100], cost_content[:100])

    prompt = f"""
    You are a cautious pricing engineer. Analyze the historical scope of work and its associated cost to learn pricing patterns. Scope: {scope_content}. Cost: {cost_content}. Extract key factors (materials, labor, market conditions) and understand how they influenced the cost. Return JSON with:
    - learned_factors: dict of factors and their impact
    - reasoning: explanation of what you learned
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        result = json.loads(response.choices[0].message.content.strip())
        logger.info("Training result: %s", json.dumps(result))
        return result
    except Exception as e:
        logger.error("Error in training: %s", str(e))
        return {"learned_factors": {}, "reasoning": f"Training failed: {str(e)}"}

if __name__ == "__main__":
    scope_file = "historical_scope.docx"  # Example file
    cost_file = "historical_cost.xlsx"    # Example file
    result = train_on_historical_data(scope_file, cost_file)
    print(result)
