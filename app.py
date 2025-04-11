import streamlit as st
import openai
import json
import logging
import pdfplumber
from docx import Document
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    filename='execution_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# إعداد OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]  # تأكد من إضافة مفتاح API في إعدادات Streamlit

# دوال مساعدة لبيانات السوق (كما هي في الأصل)
def get_market_news():
    try:
        response = requests.get("https://newsapi.org/v2/everything?q=construction+market", 
                               params={"apiKey": st.secrets["NEWS_API_KEY"]})
        news = response.json().get("articles", [])
        return news[0]["description"] if news else "No news available"
    except Exception as e:
        logging.error(f"Error fetching market news: {str(e)}")
        return "Simulated news: Global markets stable, slight material price increase."

def get_material_prices():
    return {
        "steel": 700,
        "concrete": 300,
        "cement": 150,
        "labor_hourly_rate": 50,
        "equipment_rental_daily": 200,
        "timber": 400
    }

def get_market_data():
    return {
        "news": get_market_news(),
        "prices": get_material_prices(),
        "inflation": 1.06,
        "interest": 0.045
    }

# دالة ScopeGPT لتحليل نطاق العمل
def scope_gpt_analyze(file_content, project_type):
    try:
        prompt = f"""
        You are ScopeGPT. Analyze the following scope of work and extract tasks (direct and indirect),
        contradictions, and missing details. Return a JSON with the following structure:
        {{
            "tasks": ["task1", "task2", ...],
            "contradictions": ["contradiction1", ...],
            "missing_details": ["detail1", ...]
        }}
        Scope: {file_content}
        Project Type: {project_type}
        """
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        raw_response = response.choices[0].message.content
        logging.info(f"ScopeGPT raw response: {raw_response}")

        parsed_response = json.loads(raw_response)
        return parsed_response
    except Exception as e:
        logging.error(f"ScopeGPT failed: {str(e)}")
        return {
            "tasks": [],
            "contradictions": [],
            "missing_details": [f"Error in scope analysis: {str(e)}"]
        }

# دالة MarketGPT لتقدير التكلفة (مع تحسين لضمان JSON صالح)
def market_gpt_estimate(scope_data, market_data):
    try:
        prompt = f"""
        You are MarketGPT. Estimate the cost based on the scope and market data.
        Return a valid JSON with the following structure:
        {{
            "total_cost": <number or null>,
            "cost_breakdown": {{ "direct_costs": {{}}, "indirect_costs": {{}} }},
            "reasoning": "<explanation>"
        }}
        Ensure the JSON is syntactically correct. Escape any special characters in strings.
        Scope: {scope_data}
        Market Data: {market_data}
        """
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        raw_response = response.choices[0].message.content
        logging.info(f"MarketGPT raw response: {raw_response}")

        # تحليل JSON مع التحقق
        parsed_response = json.loads(raw_response)
        if "total_cost" not in parsed_response or "cost_breakdown" not in parsed_response:
            raise ValueError("Invalid MarketGPT response: missing required fields")

        return parsed_response
    except json.JSONDecodeError as e:
        logging.error(f"MarketGPT JSON parsing failed: {str(e)} - Raw response: {raw_response}")
        return {
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": f"JSON parsing failed: {str(e)}"
        }
    except Exception as e:
        logging.error(f"MarketGPT failed: {str(e)}")
        return {
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": f"Estimation failed: {str(e)}"
        }

# دالة ValidatorGPT للتحقق من التكاليف
def validate_cost(costs):
    try:
        # التأكد من أن جميع القيم أرقام
        validated_costs = [float(cost) for cost in costs]
        logging.info(f"Validated costs: {validated_costs}")

        # التحقق من المنطقية
        if any(cost < 0 for cost in validated_costs):
            raise ValueError("Costs cannot be negative")

        return validated_costs
    except Exception as e:
        logging.error(f"Error in validate_cost: {str(e)} - Costs: {costs}")
        raise

# دالة CoordinatorGPT لدمج النتائج (بدون Fallback)
def coordinate_results(simulations):
    valid_simulations = []
    valid_costs = []

    for sim in simulations:
        try:
            # تنظيف الاستجابة من الأحرف الخاصة
            if "reasoning" in sim:
                sim["reasoning"] = sim["reasoning"].replace('"', '\\"').replace('\n', ' ')
            if sim.get("total_cost") is not None:
                valid_simulations.append(sim)
                valid_costs.append(sim["total_cost"])
        except Exception as e:
            logging.error(f"Error processing simulation result: {str(e)} - Simulation: {sim}")
            continue

    if not valid_simulations:
        logging.error("No valid simulations found")
        return {
            "tasks": simulations[0].get("tasks", []),
            "contradictions": [],
            "missing_details": [],
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": "Failed to estimate cost: No valid simulations available"
        }

    # اختيار أفضل محاكاة (مثلاً: أعلى تكلفة للحذر)
    best_simulation = max(valid_simulations, key=lambda x: x["total_cost"])
    logging.info(f"Valid costs: {valid_costs}, Selected cost: {best_simulation['total_cost']}")

    return {
        "tasks": best_simulation.get("tasks", []),
        "contradictions": [],
        "missing_details": [],
        "total_cost": best_simulation["total_cost"],
        "cost_breakdown": best_simulation.get("cost_breakdown", {}),
        "reasoning": f"Cost selected from {len(valid_simulations)} valid simulations"
    }

# دالة التحليل والتقدير باستخدام Multi-GPT
def analyze_and_estimate_multi_gpt(file_content, project_type, market_data):
    # استخراج المهام
    scope_response = scope_gpt_analyze(file_content, project_type)
    tasks = scope_response.get("tasks", [])
    contradictions = scope_response.get("contradictions", [])
    missing_details = scope_response.get("missing_details", [])

    if contradictions:
        logging.warning(f"Contradictions found in scope: {contradictions}")
        return {
            "tasks": tasks,
            "contradictions": contradictions,
            "missing_details": missing_details,
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": "Estimation aborted due to contradictions in scope"
        }

    # تشغيل المحاكاة
    simulations = []
    for i in range(10):
        logging.info(f"Running simulation {i+1}/10")
        market_response = market_gpt_estimate(file_content, market_data)
        market_response["tasks"] = tasks  # إضافة المهام إلى كل محاكاة
        simulations.append(market_response)

    # التحقق من التكاليف
    costs = [sim["total_cost"] for sim in simulations if sim["total_cost"] is not None]
    if costs:
        try:
            validate_cost(costs)
        except Exception as e:
            logging.error(f"Cost validation failed: {str(e)}")
            return {
                "tasks": tasks,
                "contradictions": [],
                "missing_details": [],
                "total_cost": None,
                "cost_breakdown": {},
                "reasoning": f"Cost validation failed: {str(e)}"
            }

    # دمج النتائج
    final_result = coordinate_results(simulations)
    logging.info(f"Estimation completed: {final_result}")
    return final_result

# دالة لقراءة الملفات
def read_file(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
        return df.to_string()
    else:
        return file.read().decode('utf-8')

# الواجهة الرئيسية باستخدام Streamlit
def main():
    st.title("CostimAIze - Project Cost Estimation")

    # تهيئة session_state إذا لم يكن موجودًا
    if "analysis_completed" not in st.session_state:
        st.session_state.analysis_completed = False
    if "final_result" not in st.session_state:
        st.session_state.final_result = None

    # إدخال معلومات اختيارية
    st.subheader("Project Details (Optional)")
    project_location = st.text_input("Project Location:")
    project_timeline = st.text_input("Project Timeline:")
    project_type = st.selectbox("Project Type:", ["Capital", "Maintenance", "Other"])
    other_info = st.text_area("Other Information:")

    # رفع نطاق العمل
    st.subheader("Upload Scope of Work")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=['doc', 'docx', 'xlsx', 'pdf'],
        accept_multiple_files=True,
        help="Limit 200MB per file • DOC, XLSX, PDF files"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"{uploaded_file.name} {uploaded_file.size / 1024:.1f}KB")
            st.button("Remove", key=f"remove_{uploaded_file.name}")

        st.success("Files uploaded successfully!")

        # زر بدء التقدير
        if st.button("Proceed to Estimate Cost"):
            with st.spinner("Processing... Please wait."):
                # قراءة الملفات
                file_contents = []
                for uploaded_file in uploaded_files:
                    content = read_file(uploaded_file)
                    file_contents.append(f"File: {uploaded_file.name}\n{content}")

                # جمع المعلومات
                project_details = f"""
                PROJECT LOCATION: {project_location}
                PROJECT TIMELINE: {project_timeline}
                PROJECT TYPE: {project_type}
                OTHER INFORMATION: {other_info}
                """
                combined_content = project_details + "\n" + "\n".join(file_contents)

                # جلب بيانات السوق
                market_data = get_market_data()

                # تشغيل التحليل
                final_result = analyze_and_estimate_multi_gpt(combined_content, project_type, market_data)

                # تخزين النتائج في session_state
                st.session_state.final_result = final_result
                st.session_state.analysis_completed = True

    # زر "اظهر النتائج" (يظهر بعد انتهاء التحليل)
    if st.session_state.analysis_completed:
        if st.button("اظهر النتائج"):
            st.subheader("Estimation Results")
            final_result = st.session_state.final_result

            # عرض النتائج
            st.write("**Tasks Extracted:**")
            for task in final_result.get("tasks", []):
                st.write(f"- {task}")

            st.write("**Total Cost:**")
            total_cost = final_result.get("total_cost")
            if total_cost is not None:
                st.write(f"${total_cost:,.2f}")
            else:
                st.error("Cost estimation failed.")

            st.write("**Cost Breakdown:**")
            st.json(final_result.get("cost_breakdown", {}))

            st.write("**Reasoning:**")
            st.write(final_result.get("reasoning", "No reasoning provided."))

    # زر عرض سجل التنفيذ
    if st.button("View Execution Log"):
        try:
            with open("execution_log.txt", "r") as f:
                log_content = f.read()
            st.subheader("Execution Log")
            log_area = st.text_area("Log Content (Copyable)", log_content, height=300)
            # زر النسخ (باستخدام JavaScript)
            st.markdown(
                """
                <button onclick="navigator.clipboard.writeText(document.getElementById('log-content').value)">
                    Copy to Clipboard
                </button>
                <script>
                    document.getElementById('log-content').id = 'log-content';
                </script>
                """,
                unsafe_allow_html=True
            )
            # زر التحميل
            with open("execution_log.txt", "rb") as f:
                st.download_button(
                    label="Download Execution Log",
                    data=f,
                    file_name="execution_log.txt",
                    mime="text/plain"
                )
        except FileNotFoundError:
            st.error("Execution log not found.")

    # زر العودة إلى الـ Dashboard
    if st.button("Back to Dashboard"):
        st.session_state.analysis_completed = False
        st.session_state.final_result = None
        st.experimental_rerun()

if __name__ == "__main__":
    main()