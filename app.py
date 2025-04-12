import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pytesseract
from PIL import Image, ImageEnhance
import re
import requests
from pdf2image import convert_from_bytes
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict

# تحميل بيانات NLTK تلقائيًا عند بدء التطبيق
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# إعدادات OCR (مسار Tesseract على خادم Linux)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# دالة لتحسين جودة الصور قبل OCR
def preprocess_image(image):
    image = image.convert('L')  # تحويل إلى تدرج الرمادي
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # زيادة التباين
    return image

# دالة لاستخراج النصوص من الصور (OCR)
def extract_text_from_image(image):
    try:
        image = preprocess_image(image)
        text = pytesseract.image_to_string(image, lang='ara+eng')
        return text
    except Exception as e:
        return f"خطأ في استخراج النص: {str(e)}"

# دالة لاستخراج النصوص من PDF
def extract_text_from_pdf(uploaded_file):
    try:
        # تحويل PDF إلى صور
        images = convert_from_bytes(uploaded_file.read(), size=(2000, 2000))
        text = ""
        for i, image in enumerate(images):
            st.write(f"معالجة الصفحة {i + 1} من {len(images)}...")
            page_text = extract_text_from_image(image)
            text += page_text + "\n"
        return text
    except Exception as e:
        return f"خطأ في قراءة الملف: {str(e)}"

# دالة لتحليل النص واستخراج المهام باستخدام NLP
def extract_tasks(text):
    sentences = text.split('\n')
    tasks = []
    for sentence in sentences:
        if len(sentence.strip()) > 10:  # تجاهل الجمل القصيرة جدًا
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            # البحث عن جمل تحتوي على أفعال وأسماء (مهام محتملة)
            if any(tag.startswith('VB') for word, tag in tagged) and any(tag.startswith('NN') for word, tag in tagged):
                tasks.append(sentence.strip())
    return tasks

# دالة لاستخراج الأسعار من النص
def extract_prices(text):
    prices = re.findall(r'\d+\.?\d*', text)
    return [float(price) for price in prices]

# دالة لتحليل نطاق العمل (ScopeGPT)
def analyze_scope(text):
    tasks = extract_tasks(text)
    direct_costs = defaultdict(float)
    indirect_costs = defaultdict(float)
    missing_details = []

    # تحديد البنود المباشرة وغير المباشرة بناءً على السياق
    for task in tasks:
        task_lower = task.lower()
        if "design" in task_lower:
            direct_costs["design"] += 500000  # تكلفة افتراضية
        if "transformer" in task_lower:
            direct_costs["transformers"] += 4000000
        if "gis" in task_lower:
            direct_costs["gis_system"] += 2500000
        if "scada" in task_lower or "sas" in task_lower:
            direct_costs["scada_sas"] += 1200000
        if "fire" in task_lower:
            direct_costs["fire_protection"] += 500000
        if "cybersecurity" in task_lower:
            direct_costs["cybersecurity"] += 300000
        if "civil" in task_lower:
            direct_costs["civil_works"] += 1500000
        if "labor" in task_lower:
            direct_costs["labor"] += 2000000
        if "training" in task_lower or "handover" in task_lower:
            indirect_costs["training_handover"] += 400000
        if "management" in task_lower:
            indirect_costs["project_management"] += 600000

    # تحديد التفاصيل المفقودة
    if "location" not in text.lower():
        missing_details.append("الموقع الدقيق للمشروع غير محدد.")
    if "timeline" not in text.lower():
        missing_details.append("الجدول الزمني للمشروع غير محدد.")

    return tasks, dict(direct_costs), dict(indirect_costs), missing_details

# دالة لجلب بيانات السوق (MarketGPT)
def fetch_market_data():
    # هنا محاكاة لجلب بيانات السوق (لأننا لا نستطيع الوصول إلى API حقيقي)
    market_data = {
        "inflation_rate": 0.03,  # 3%
        "interest_rate": 0.05,   # 5%
        "market_stability": "Stable with moderate inflation",
        "price_trends": {
            "transformers": {"base_price": 4000000, "trend": 0.1},  # ارتفاع 10%
            "gis_system": {"base_price": 2500000, "trend": 0.05},
            "labor": {"base_price": 2000000, "trend": 0.02}
        }
    }
    return market_data

# دالة لتقدير التكلفة كمسعّر ذكي حذر
def cautious_pricing(direct_costs, indirect_costs, tasks, market_data, user_inputs=None):
    total_cost = 0.0
    reasoning = []

    # تطبيق التكاليف المباشرة
    for item, cost in direct_costs.items():
        adjusted_cost = cost
        # النظام يقرر بحرية ما إذا كان سيطبق التضخم أو اتجاهات السوق
        if item in market_data["price_trends"]:
            trend = market_data["price_trends"][item]["trend"]
            if len(tasks) > 5:  # إذا كان المشروع معقدًا، يمكن تطبيق الاتجاه
                adjusted_cost = cost * (1 + trend)
                reasoning.append(f"زيادة تكلفة {item} بنسبة {trend*100}% بسبب اتجاهات السوق.")
            else:
                reasoning.append(f"لم يتم تطبيق اتجاهات السوق على {item} لأن المشروع بسيط.")
        total_cost += adjusted_cost
        direct_costs[item] = adjusted_cost

    # تطبيق التكاليف غير المباشرة
    for item, cost in indirect_costs.items():
        total_cost += cost

    # النظام يقرر بحرية مستوى الحذر
    cautious_factor = 1.0
    if len(tasks) > 10:
        cautious_factor = 1.15  # زيادة 15% للمشاريع المعقدة
        reasoning.append("تم تطبيق معامل حذر 15% بسبب تعقيد المشروع (عدد المهام كبير).")
    elif user_inputs and "timeline" not in user_inputs:
        cautious_factor = 1.1  # زيادة 10% إذا كان الجدول الزمني مفقود
        reasoning.append("تم تطبيق معامل حذر 10% بسبب عدم وجود جدول زمني.")
    else:
        reasoning.append("لم يتم تطبيق معامل حذر إضافي لأن المشروع واضح.")

    total_cost *= cautious_factor

    # النظام يقرر بحرية ما إذا كان سيطبق الفائدة
    if user_inputs and "financing" in user_inputs and user_inputs["financing"].lower() == "yes":
        interest_cost = total_cost * market_data["interest_rate"]
        total_cost += interest_cost
        reasoning.append(f"تمت إضافة تكلفة فائدة {interest_cost:.2f} بنسبة {market_data['interest_rate']*100}% بسبب وجود تمويل.")
    else:
        reasoning.append("لم يتم تطبيق تكلفة الفائدة لأن التمويل غير مذكور.")

    return total_cost, reasoning

# واجهة Streamlit
st.title("Costimaizer - نظام تقدير التكاليف الذكي")

# رفع ملف
uploaded_file = st.file_uploader("ارفع ملف الصورة أو PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    # تحديد نوع الملف
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="الملف المرفوع", use_column_width=True)
        extracted_text = extract_text_from_image(image)

    st.subheader("النص المستخرج:")
    st.write(extracted_text)

    # تحليل النطاق
    tasks, direct_costs, indirect_costs, missing_details = analyze_scope(extracted_text)

    # عرض المهام المستخرجة
    st.subheader("المهام المستخرجة:")
    for task in tasks:
        st.write(f"- {task}")

    # عرض التفاصيل المفقودة وطرح أسئلة
    user_inputs = {}
    if missing_details:
        st.subheader("استفسارات النظام:")
        for detail in missing_details:
            st.write(detail)
            if "الموقع" in detail:
                user_inputs["location"] = st.text_input("ما هو الموقع الدقيق للمشروع؟")
            if "الجدول الزمني" in detail:
                user_inputs["timeline"] = st.text_input("ما هو الجدول الزمني للمشروع؟")
        user_inputs["financing"] = st.selectbox("هل المشروع ممول؟", ["Yes", "No"])
        if st.button("إرسال الإجابات"):
            st.success("تم حفظ الإجابات!")

    # جلب بيانات السوق
    market_data = fetch_market_data()

    # تقدير التكلفة
    if st.button("إظهار التقدير"):
        with st.spinner("جارٍ تقدير التكلفة..."):
            total_cost, reasoning = cautious_pricing(direct_costs, indirect_costs, tasks, market_data, user_inputs)

            # عرض النتائج
            st.subheader("التقدير النهائي للتكلفة:")
            st.write(f"${total_cost:,.2f}")

            st.subheader("تفاصيل التكاليف:")
            st.write("**التكاليف المباشرة:**")
            for item, cost in direct_costs.items():
                st.write(f"- {item}: ${cost:,.2f}")
            st.write("**التكاليف غير المباشرة:**")
            for item, cost in indirect_costs.items():
                st.write(f"- {item}: ${cost:,.2f}")

            st.subheader("تفسير القرارات:")
            for reason in reasoning:
                st.write(f"- {reason}")