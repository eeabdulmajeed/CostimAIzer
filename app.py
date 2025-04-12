import streamlit as st
import openai
import json
import logging
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    filename='execution_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# إنشاء ملف السجل مع رسالة افتراضية
try:
    with open("execution_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - INFO - التطبيق بدأ تشغيله\n")
except Exception as e:
    print(f"Error initializing log file: {str(e)}")

# إعداد OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# دالة لتحليل السوق ديناميكيًا (بدون Fallback)
def dynamic_market_analysis(context):
    try:
        prompt = f"""
        Analyze market conditions for the project: {context}.
        Return JSON with cost items (direct/indirect), ranges, and market impact.
        Do not use hardcoded values. If data is unavailable, return an error message.
        """
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        market_data = json.loads(response.choices[0].message.content)
        logging.info("تحليل السوق اكتمل")
        return market_data
    except Exception as e:
        logging.error(f"خطأ في تحليل السوق: {str(e)}")
        return {
            "status": "error",
            "message": "فشل تحليل السوق، يُرجى مراجعة يدوية",
            "items": {},
            "expected_range": {"min": 0, "max": float("inf")}
        }

# دالة لتحليل نطاق العمل (مع دعم OCR)
def scope_gpt_analyze(file_content, project_type):
    try:
        prompt = f"""
        You are ScopeGPT. Analyze the scope of work and extract tasks (direct and indirect),
        contradictions, and missing details. Return JSON:
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
        logging.info(f"ScopeGPT response: {raw_response}")
        return json.loads(raw_response)
    except Exception as e:
        logging.error(f"ScopeGPT failed: {str(e)}")
        return {
            "tasks": [],
            "contradictions": [],
            "missing_details": [f"Error in scope analysis: {str(e)}"]
        }

# دالة لتقدير التكلفة
def market_gpt_estimate(scope_data, market_data):
    try:
        prompt = f"""
        You are MarketGPT. Estimate the cost based on scope and market data.
        Return JSON:
        {{
            "total_cost": <number or null>,
            "cost_breakdown": {{ "direct_costs": {{}}, "indirect_costs": {{}} }},
            "reasoning": "<explanation>"
        }}
        Scope: {scope_data}
        Market Data: {market_data}
        """
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        raw_response = response.choices[0].message.content
        logging.info(f"MarketGPT response: {raw_response}")
        parsed_response = json.loads(raw_response)
        if "total_cost" not in parsed_response or "cost_breakdown" not in parsed_response:
            raise ValueError("Invalid MarketGPT response")
        return parsed_response
    except Exception as e:
        logging.error(f"MarketGPT failed: {str(e)}")
        return {
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": f"Estimation failed: {str(e)}"
        }

# دالة للتحقق من المنطقية (مع طبقة الاعتراض)
def validate_cost(costs, market_data):
    try:
        validated_costs = [float(cost) for cost in costs if cost is not None]
        if not validated_costs:
            raise ValueError("No valid costs provided")
        
        expected_range = market_data.get("expected_range", {"min": 0, "max": float("inf")})
        min_cost, max_cost = expected_range.get("min", 0), expected_range.get("max", float("inf"))
        for cost in validated_costs:
            if not (min_cost <= cost <= max_cost):
                raise ValueError(f"Cost {cost} outside expected range [{min_cost}, {max_cost}]")
            if cost < 0:
                raise ValueError("Costs cannot be negative")
        
        logging.info(f"Validated costs: {validated_costs}")
        return validated_costs
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
        raise

# دالة Monte Carlo
def monte_carlo_simulation(simulations, market_data):
    valid_costs = []
    valid_simulations = []
    
    for sim in simulations:
        try:
            if sim.get("total_cost") is not None:
                variation = np.random.normal(0, 0.05 * sim["total_cost"])
                adjusted_cost = max(0, sim["total_cost"] + variation)
                valid_costs.append(adjusted_cost)
                valid_simulations.append(sim)
        except Exception as e:
            logging.error(f"Simulation error: {str(e)}")
            continue
    
    if not valid_costs:
        logging.error("No valid Monte Carlo costs")
        return None, []
    
    avg_cost = np.mean(valid_costs)
    outliers = [c for c in valid_costs if abs(c - avg_cost) > 2 * np.std(valid_costs)]
    logging.info(f"Monte Carlo: avg={avg_cost}, outliers={len(outliers)}")
    return avg_cost, valid_simulations

# دالة لدمج النتائج
def coordinate_results(simulations, market_data):
    valid_simulations = [s for s in simulations if s.get("total_cost") is not None]
    if not valid_simulations:
        logging.error("No valid simulations")
        return {
            "tasks": [],
            "contradictions": [],
            "missing_details": [],
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": "No valid simulations available"
        }
    
    avg_cost, valid_simulations = monte_carlo_simulation(simulations, market_data)
    if avg_cost is None:
        return {
            "tasks": [],
            "contradictions": [],
            "missing_details": [],
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": "Monte Carlo simulation failed"
        }
    
    best_simulation = max(valid_simulations, key=lambda x: x["total_cost"])
    logging.info(f"Selected cost: {avg_cost}")
    
    total = sum(v for v in best_simulation["cost_breakdown"]["direct_costs"].values()) + \
            sum(v for v in best_simulation["cost_breakdown"]["indirect_costs"].values())
    breakdown = {
        "direct_costs": {k: {"cost": v, "percentage": v / total * 100}
                         for k, v in best_simulation["cost_breakdown"]["direct_costs"].items()},
        "indirect_costs": {k: {"cost": v, "percentage": v / total * 100}
                           for k, v in best_simulation["cost_breakdown"]["indirect_costs"].items()}
    }
    
    return {
        "tasks": best_simulation.get("tasks", []),
        "contradictions": [],
        "missing_details": [],
        "total_cost": avg_cost,
        "cost_breakdown": breakdown,
        "reasoning": f"Cost estimated from {len(valid_simulations)} simulations, market impact: {market_data.get('market_conditions', 'N/A')}"
    }

# دالة لقراءة الملفات (مع OCR)
def read_file(file):
    try:
        if file.name.endswith(('.pdf', '.png', '.jpg')):
            image = Image.open(file)
            text = pytesseract.image_to_string(image, lang="eng+ara")
            return text
        elif file.name.endswith('.docx'):
            from docx import Document
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
            return df.to_string()
        else:
            return file.read().decode('utf-8')
    except Exception as e:
        logging.error(f"File read error: {str(e)}")
        return ""

# دالة التحليل والتقدير
def analyze_and_estimate_multi_gpt(file_content, project_type, market_data):
    scope_response = scope_gpt_analyze(file_content, project_type)
    tasks = scope_response.get("tasks", [])
    contradictions = scope_response.get("contradictions", [])
    missing_details = scope_response.get("missing_details", [])
    
    if contradictions:
        logging.warning(f"Contradictions: {contradictions}")
        return {
            "tasks": tasks,
            "contradictions": contradictions,
            "missing_details": missing_details,
            "total_cost": None,
            "cost_breakdown": {},
            "reasoning": "تم إيقاف التقدير بسبب تناقضات"
        }
    
    simulations = []
    for i in range(10):
        logging.info(f"Simulation {i+1}/10")
        market_response = market_gpt_estimate(file_content, market_data)
        market_response["tasks"] = tasks
        simulations.append(market_response)
    
    costs = [sim["total_cost"] for sim in simulations if sim["total_cost"] is not None]
    if costs:
        try:
            validate_cost(costs, market_data)
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return {
                "tasks": tasks,
                "contradictions": [],
                "missing_details": [],
                "total_cost": None,
                "cost_breakdown": {},
                "reasoning": f"فشل التحقق: {str(e)}"
            }
    
    final_result = coordinate_results(simulations, market_data)
    logging.info(f"Final result: {final_result}")
    return final_result

# الواجهة الرئيسية
def main():
    st.title("CostimAIze - تقدير ذكي للتكاليف")
    
    if "analysis_completed" not in st.session_state:
        st.session_state.analysis_completed = False
    if "final_result" not in st.session_state:
        st.session_state.final_result = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    st.subheader("تفاصيل المشروع")
    project_location = st.text_input("موقع المشروع:")
    project_timeline = st.text_input("الجدول الزمني:")
    project_type = st.selectbox("نوع المشروع:", ["Capital", "Maintenance", "Other"])
    other_info = st.text_area("معلومات إضافية:")
    
    st.subheader("رفع نطاق العمل")
    st.info("إذا كنت تستخدم جوال، تأكد من إذن الوصول إلى الملفات. الحد الأقصى 200 ميجابايت لكل ملف.")
    uploaded_files = st.file_uploader(
        "ارفع الملفات",
        type=['docx', 'xlsx', 'pdf', 'png', 'jpg'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # تحديث قائمة الملفات المرفوعة
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success("تم رفع الملفات بنجاح!")
        for uploaded_file in st.session_state.uploaded_files:
            st.write(f"- {uploaded_file.name} ({uploaded_file.size / 1024:.1f}KB)")
    else:
        if st.session_state.uploaded_files:
            st.warning("تم إزالة الملفات المرفوعة. يرجى رفع ملفات جديدة.")
            st.session_state.uploaded_files = []
        else:
            st.warning("لم يتم رفع أي ملفات بعد.")

    # زر بدء التقدير
    if st.session_state.uploaded_files:
        if st.button("بدء التقدير", key="start_estimation"):
            with st.spinner("جارٍ التحليل..."):
                try:
                    file_contents = []
                    for uploaded_file in st.session_state.uploaded_files:
                        content = read_file(uploaded_file)
                        file_contents.append(f"File: {uploaded_file.name}\n{content}")
                    
                    project_details = f"""
                    LOCATION: {project_location}
                    TIMELINE: {project_timeline}
                    TYPE: {project_type}
                    OTHER: {other_info}
                    """
                    combined_content = project_details + "\n" + "\n".join(file_contents)
                    
                    market_data = dynamic_market_analysis(combined_content)
                    if market_data.get("status") == "error":
                        st.error(market_data["message"])
                        logging.error(market_data["message"])
                        return
                    
                    final_result = analyze_and_estimate_multi_gpt(combined_content, project_type, market_data)
                    st.session_state.final_result = final_result
                    st.session_state.analysis_completed = True
                except Exception as e:
                    st.error(f"فشل التحليل: {str(e)}")
                    logging.error(f"تحليل نطاق العمل فشل: {str(e)}")
    
    # زر إظهار النتائج
    if st.session_state.analysis_completed:
        if st.button("إظهار النتائج", key="show_results"):
            st.subheader("نتائج التقدير")
            final_result = st.session_state.final_result
            
            st.write("**المهام المستخرجة**:")
            for task in final_result.get("tasks", []):
                st.write(f"- {task}")
            
            st.write("**إجمالي التكلفة**:")
            total_cost = final_result.get("total_cost")
            if total_cost is not None:
                st.metric("التكلفة", f"${total_cost:,.2f}")
            else:
                st.error("فشل تقدير التكلفة")
            
            st.write("**تفاصيل التكلفة**:")
            df_direct = pd.DataFrame(final_result["cost_breakdown"].get("direct_costs", {})).T
            df_indirect = pd.DataFrame(final_result["cost_breakdown"].get("indirect_costs", {})).T
            if not df_direct.empty:
                st.write("التكاليف المباشرة:")
                st.dataframe(df_direct.style.format({"cost": "{:,.2f}", "percentage": "{:.1f}%"}))
            if not df_indirect.empty:
                st.write("التكاليف غير المباشرة:")
                st.dataframe(df_indirect.style.format({"cost": "{:,.2f}", "percentage": "{:.1f}%"}))
            
            st.write("**التفسير**:")
            st.write(final_result.get("reasoning", "غير متوفر"))
    
    # زر عرض سجل العمليات
    if st.button("عرض سجل العمليات", key="view_log"):
        try:
            with open("execution_log.txt", "r", encoding="utf-8") as f:
                log_content = f.read()
            if log_content.strip():
                st.text_area("سجل العمليات", log_content, height=300)
                st.download_button(
                    label="تحميل السجل",
                    data=log_content,
                    file_name="execution_log.txt",
                    mime="text/plain"
                )
            else:
                st.warning("السجل فارغ. لم يتم تسجيل أي عمليات بعد.")
        except FileNotFoundError:
            st.error("السجل غير متوفر. حاول مرة أخرى لاحقًا.")
    
    # زر العودة
    if st.button("العودة", key="reset"):
        st.session_state.analysis_completed = False
        st.session_state.final_result = None
        st.session_state.uploaded_files = []
        st.rerun()

if __name__ == "__main__":
    main()