import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pytesseract
from PIL import Image, ImageEnhance
import re
from pdf2image import convert_from_path
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import openai
import os
from datetime import datetime
import tempfile

# ملاحظات هامة للمطور:
# 1. هذا الكود يعتمد على ركائز أساسية: دوال OpenAI (ScopeGPT، MarketGPT، ReviewGPT)، دوال NLP، والتفاعل بين النماذج.
# 2. يُمنع حذف أو تعديل هذه الركائز دون استشارة المالك، لأنها أساس عمل التطبيق.
# 3. أي تعديلات يجب أن تكون تراكمية مع الحفاظ على الوظائف القائمة.

# إعداد مسار مؤقت لتخزين بيانات NLTK
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
try:
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
except Exception as e:
    st.error(f"Error creating NLTK data directory: {str(e)}")

# تحميل بيانات NLTK إذا لم تكن موجودة
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
        nltk.download('wordnet', download_dir=nltk_data_path)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")

# إعدادات OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-4-turbo"  # أحدث إصدار متاح في أبريل 2025

# إعدادات OCR
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# سجل العمليات
if "execution_log" not in st.session_state:
    st.session_state["execution_log"] = []

# دالة لإضافة إدخال إلى السجل
def log_action(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["execution_log"].append(f"[{timestamp}] {action}")

# ركيزة أساسية: تحسين جودة الصورة - لا يجوز حذفها
def preprocess_image(image):
    log_action("تحسين جودة الصورة قبل OCR")
    image = image.convert('L')  # تحويل إلى تدرج الرمادي
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # زيادة التباين
    log_action("تم تحسين جودة الصورة")
    return image

# ركيزة أساسية: استخراج النص من الصورة - لا يجوز حذفها
def extract_text_from_image(image):
    log_action("بدء استخراج النص من الصورة")
    try:
        image = preprocess_image(image)
        text = pytesseract.image_to_string(image, lang='ara+eng')
        log_action("اكتمال استخراج النص من الصورة")
        return text
    except Exception as e:
        log_action(f"خطأ في استخراج النص: {str(e)}")
        return f"خطأ في استخراج النص: {str(e)}"

# ركيزة أساسية: استخراج النص من PDF - لا يجوز حذفها
def extract_text_from_pdf(uploaded_file):
    log_action("بدء استخراج النص من PDF")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        images = convert_from_path(temp_file_path, size=(2000, 2000))
        text = ""
        for i, image in enumerate(images):
            st.write(f"معالجة الصفحة {i + 1} من {len(images)}...")
            log_action(f"معالجة الصفحة {i + 1} من {len(images)}")
            page_text = extract_text_from_image(image)
            text += page_text + "\n"

        os.remove(temp_file_path)
        log_action("اكتمل استخراج النص من PDF")
        return text
    except Exception as e:
        log_action(f"خطأ في قراءة الملف: {str(e)}")
        return f"خطأ في قراءة الملف: {str(e)}"

# ركيزة أساسية: استخراج المهام باستخدام NLP - لا يجوز حذفها
def extract_tasks(text):
    log_action("بدء استخراج المهام باستخدام NLP")
    sentences = text.split('\n')
    tasks = []
    for sentence in sentences:
        if len(sentence.strip()) > 10:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            if any(tag.startswith('VB') for word, tag in tagged) and any(tag.startswith('NN') for word, tag in tagged):
                tasks.append(sentence.strip())
    log_action(f"تم استخراج {len(tasks)} مهمة")
    return tasks

# ركيزة أساسية: تحليل نطاق العمل باستخدام ScopeGPT - لا يجوز حذفها
def analyze_scope(text):
    log_action("بدء تحليل نطاق العمل باستخدام OpenAI")
    tasks = extract_tasks(text)
    direct_cost_items = []
    indirect_cost_items = []
    missing_details = []

    prompt = f"""
    Analyze the following project scope text and identify direct and indirect cost items. Direct cost items are related to execution (e.g., equipment, labor), while indirect cost items are related to support (e.g., training, management). Also, identify any missing details that might affect cost estimation (e.g., location, timeline, budget, inflation, financing). Here is the text:\n\n{text}
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are ScopeGPT, an AI specializing in analyzing project scopes for cost estimation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        analysis = response.choices[0].message.content

        lines = analysis.split('\n')
        for line in lines:
            line = line.lower().strip()
            if line.startswith("direct cost item:"):
                direct_cost_items.append(line.replace("direct cost item:", "").strip())
            elif line.startswith("indirect cost item:"):
                indirect_cost_items.append(line.replace("indirect cost item:", "").strip())
            elif line.startswith("missing detail:"):
                missing_details.append(line.replace("missing detail:", "").strip())
        log_action("اكتمل تحليل نطاق العمل")
    except Exception as e:
        log_action(f"خطأ في تحليل نطاق العمل: {str(e)}")
        st.error(f"Error in scope analysis: {str(e)}")
    return tasks, direct_cost_items, indirect_cost_items, missing_details

# ركيزة أساسية: جلب بيانات السوق باستخدام MarketGPT - لا يجوز حذفها
def fetch_market_data(text, direct_cost_items, indirect_cost_items):
    log_action("بدء جلب بيانات السوق")
    market_data = {
        "inflation_rate": None,
        "interest_rate": None,
        "material_prices": {},
        "market_stability": None
    }
    reasoning = []

    prompt = f"""
    You are MarketGPT, an AI specializing in fetching market data for cost estimation. Based on the following project scope text, direct cost items, and indirect cost items, fetch the following economic factors from the internet for April 2025:
    - Inflation rate (global or local to the project context)
    - Interest rate (relevant to project financing)
    - Material prices for items mentioned in direct costs (e.g., transformers, GIS systems)
    - Market stability (e.g., stable, volatile)
    Here is the project scope text:\n\n{text}
    Direct cost items: {', '.join(direct_cost_items)}
    Indirect cost items: {', '.join(indirect_cost_items)}
    Provide a detailed reasoning for your findings.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are MarketGPT, an AI specializing in fetching market data for cost estimation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        market_response = response.choices[0].message.content

        lines = market_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Inflation rate:"):
                try:
                    rate = float(re.search(r'\d+\.?\d*', line).group()) / 100
                    market_data["inflation_rate"] = rate
                    reasoning.append(f"تم جلب معدل التضخم: {rate*100}% من الإنترنت بناءً على تحليل السوق.")
                except:
                    reasoning.append("لم يتم العثور على معدل التضخم في البيانات المستجلبة.")
            elif line.startswith("Interest rate:"):
                try:
                    rate = float(re.search(r'\d+\.?\d*', line).group()) / 100
                    market_data["interest_rate"] = rate
                    reasoning.append(f"تم جلب معدل الفائدة: {rate*100}% من الإنترنت بناءً على تحليل السوق.")
                except:
                    reasoning.append("لم يتم العثور على معدل الفائدة في البيانات المستجلبة.")
            elif line.startswith("Material price:"):
                try:
                    parts = line.split(':')
                    item = parts[1].split('=')[0].strip()
                    price = float(re.search(r'\d+\.?\d*', parts[1]).group())
                    market_data["material_prices"][item] = price
                    reasoning.append(f"تم جلب سعر المادة '{item}': ${price:,.2f} من الإنترنت.")
                except:
                    reasoning.append(f"لم يتم العثور على سعر المادة في البيانات المستجلبة: {line}")
            elif line.startswith("Market stability:"):
                market_data["market_stability"] = line.replace("Market stability:", "").strip()
                reasoning.append(f"تم تحديد استقرار السوق: {market_data['market_stability']} بناءً على تحليل السوق.")
        log_action("اكتمل جلب بيانات السوق")
    except Exception as e:
        log_action(f"خطأ في جلب بيانات السوق: {str(e)}")
        st.error(f"Error fetching market data: {str(e)}")
    return market_data, reasoning

# ركيزة أساسية: التحقق من النتائج بين النماذج - لا يجوز حذفها
def validate(scope_results, market_results):
    log_action("بدء التحقق من النتائج")
    discrepancies = []
    scope_tasks, scope_direct, scope_indirect, _ = scope_results
    market_data, _ = market_results

    for item in scope_direct:
        item_lower = item.lower()
        for material, price in market_data["material_prices"].items():
            if material.lower() in item_lower and price <= 0:
                discrepancies.append(f"تناقض: سعر المادة '{material}' في البنود المباشرة ({item}) غير منطقي: ${price:,.2f}")

    if market_data["inflation_rate"] is None:
        discrepancies.append("تناقض: لم يتم جلب معدل التضخم.")
    if market_data["interest_rate"] is None:
        discrepancies.append("تناقض: لم يتم جلب معدل الفائدة.")

    log_action(f"تم العثور على {len(discrepancies)} تناقض")
    return discrepancies

# ركيزة أساسية: التحكيم بين النماذج - لا يجوز حذفها
def arbitrate(discrepancies, scope_results, market_results):
    log_action("بدء التحكيم في التناقضات")
    reasoning = []
    scope_tasks, scope_direct, scope_indirect, _ = scope_results
    market_data, market_reasoning = market_results

    for discrepancy in discrepancies:
        if "غير منطقي" in discrepancy:
            item = re.search(r"'(.*?)'", discrepancy).group(1)
            prompt = f"Re-estimate the price of '{item}' for a project in April 2025, ensuring the price is reasonable and based on current market data."
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an AI specializing in market data estimation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                new_price = float(re.search(r'\d+\.?\d*', response.choices[0].message.content).group())
                market_data["material_prices"][item] = new_price
                reasoning.append(f"تم التحكيم: أعيد تقدير سعر '{item}' إلى ${new_price:,.2f} بناءً على تحليل السوق.")
            except Exception as e:
                reasoning.append(f"فشل في التحكيم لسعر '{item}': {str(e)}")
        elif "لم يتم جلب" in discrepancy:
            factor = "inflation rate" if "التضخم" in discrepancy else "interest rate"
            prompt = f"Fetch the {factor} for April 2025 based on global market data."
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an AI specializing in market data estimation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                rate = float(re.search(r'\d+\.?\d*', response.choices[0].message.content).group()) / 100
                market_data[factor.replace(" ", "_")] = rate
                reasoning.append(f"تم التحكيم: أعيد جلب {factor}: {rate*100}% بناءً على تحليل السوق.")
            except Exception as e:
                reasoning.append(f"فشل في التحكيم لـ {factor}: {str(e)}")

    log_action("اكتمل التحكيم في التناقضات")
    return market_data, reasoning

# ركيزة أساسية: تقدير التكلفة باستخدام ReviewGPT - لا يجوز حذفها
def cautious_pricing(scope_results, market_results):
    log_action("بدء تقدير التكلفة")
    total_cost = 0.0
    direct_costs = defaultdict(float)
    indirect_costs = defaultdict(float)
    reasoning = []

    scope_tasks, scope_direct, scope_indirect, missing_details = scope_results
    market_data, market_reasoning = market_results

    discrepancies = validate(scope_results, market_results)
    if discrepancies:
        market_data, arbitration_reasoning = arbitrate(discrepancies, scope_results, market_results)
        reasoning.extend(arbitration_reasoning)

    prompt = f"""
    You are ReviewGPT, an AI specializing in cost estimation and cautious pricing. Based on the following project scope, direct cost items, indirect cost items, and market data, estimate the total cost of the project. Ensure the estimation is cautious but reasonable, and provide a detailed breakdown of direct and indirect costs, along with reasoning for each step. The scope does not contain prices, so rely on market data and context to estimate costs. Apply inflation and interest rates if available, and determine a cautious factor based on the complexity (number of tasks: {len(scope_tasks)}) and missing details ({len(missing_details)}).

    Project scope text: {', '.join(scope_tasks)}
    Direct cost items: {', '.join(scope_direct)}
    Indirect cost items: {', '.join(scope_indirect)}
    Market data:
    - Inflation rate: {market_data['inflation_rate']*100 if market_data['inflation_rate'] else 'Not available'}%
    - Interest rate: {market_data['interest_rate']*100 if market_data['interest_rate'] else 'Not available'}%
    - Material prices: {', '.join([f'{k}: ${v:,.2f}' for k, v in market_data['material_prices'].items()])}
    - Market stability: {market_data['market_stability']}
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are ReviewGPT, an AI specializing in cost estimation and cautious pricing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        cost_response = response.choices[0].message.content

        lines = cost_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Direct cost:"):
                try:
                    parts = line.split(':')
                    item = parts[1].split('=')[0].strip()
                    cost = float(re.search(r'\d+\.?\d*', parts[1]).group())
                    direct_costs[item] = cost
                    reasoning.append(f"تم تقدير تكلفة البند المباشر '{item}': ${cost:,.2f} بناءً على تحليل السوق.")
                except:
                    reasoning.append(f"فشل في تقدير تكلفة البند المباشر: {line}")
            elif line.startswith("Indirect cost:"):
                try:
                    parts = line.split(':')
                    item = parts[1].split('=')[0].strip()
                    cost = float(re.search(r'\d+\.?\d*', parts[1]).group())
                    indirect_costs[item] = cost
                    reasoning.append(f"تم تقدير تكلفة البند غير المباشر '{item}': ${cost:,.2f} بناءً على تحليل السوق