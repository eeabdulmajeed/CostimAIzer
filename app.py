import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pytesseract
from PIL import Image, ImageEnhance
import re
from pdf2image import convert_from_bytes
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import openai
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pytesseract
from PIL import Image, ImageEnhance
import re
from pdf2image import convert_from_bytes
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import openai
import os

# إعداد مسار مؤقت لتخزين بيانات NLTK
nltk_data_path = "/tmp/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# تحميل بيانات NLTK إذا لم تكن موجودة
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)

# إعدادات OpenAI (استخدام أحدث إصدار من GPT)
openai.api_key = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-4-turbo"  # أحدث إصدار متاح في أبريل 2025

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
        if len(sentence.strip()) > 10:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            if any(tag.startswith('VB') for word, tag in tagged) and any(tag.startswith('NN') for word, tag in tagged):
                tasks.append(sentence.strip())
    return tasks

# دالة لتحليل نطاق العمل باستخدام OpenAI (ScopeGPT)
def analyze_scope(text):
    tasks = extract_tasks(text)
    direct_cost_items = []
    indirect_cost_items = []
    missing_details = []

    # تحديد البنود المباشرة وغير المباشرة باستخدام OpenAI
    prompt = f"""
    Analyze the following project scope text and identify direct and indirect cost items. Direct cost items are related to execution (e.g., equipment, labor), while indirect cost items are related to support (e.g., training, management). Also, identify any missing details that might affect cost estimation (e.g., location, timeline, budget, inflation, financing). Here is the text:\n\n{text}
    """
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are ScopeGPT, an AI specializing in analyzing project scopes for cost estimation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    analysis = response.choices[0].message.content

    # معالجة رد OpenAI
    lines = analysis.split('\n')
    for line in lines:
        line = line.lower().strip()
        if line.startswith("direct cost item:"):
            direct_cost_items.append(line.replace("direct cost item:", "").strip())
        elif line.startswith("indirect cost item:"):
            indirect_cost_items.append(line.replace("indirect cost item:", "").strip())
        elif line.startswith("missing detail:"):
            missing_details.append(line.replace("missing detail:", "").strip())

    return tasks, direct_cost_items, indirect_cost_items, missing_details

# دالة لجلب بيانات السوق باستخدام OpenAI (MarketGPT)
def fetch_market_data(text, direct_cost_items, indirect_cost_items):
    market_data = {
        "inflation_rate": None,
        "interest_rate": None,
        "material_prices": {},
        "market_stability": None
    }
    reasoning = []

    # جلب العوامل الاقتصادية باستخدام OpenAI
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
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are MarketGPT, an AI specializing in fetching market data for cost estimation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    market_response = response.choices[0].message.content

    # معالجة رد OpenAI
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

    return market_data, reasoning

# دالة للتحقق من النتائج بين النماذج (Validate)
def validate(scope_results, market_results):
    discrepancies = []
    scope_tasks, scope_direct, scope_indirect, _ = scope_results
    market_data, _ = market_results

    # التحقق من توافق البنود المباشرة مع أسعار المواد
    for item in scope_direct:
        item_lower = item.lower()
        for material, price in market_data["material_prices"].items():
            if material.lower() in item_lower and price <= 0:
                discrepancies.append(f"تناقض: سعر المادة '{material}' في البنود المباشرة ({item}) غير منطقي: ${price:,.2f}")

    # التحقق من العوامل الاقتصادية
    if market_data["inflation_rate"] is None:
        discrepancies.append("تناقض: لم يتم جلب معدل التضخم.")
    if market_data["interest_rate"] is None:
        discrepancies.append("تناقض: لم يتم جلب معدل الفائدة.")

    return discrepancies

# دالة للتحكيم بين النماذج (Arbitrate)
def arbitrate(discrepancies, scope_results, market_results):
    reasoning = []
    scope_tasks, scope_direct, scope_indirect, _ = scope_results
    market_data, market_reasoning = market_results

    # التحكيم في التناقضات
    for discrepancy in discrepancies:
        if "غير منطقي" in discrepancy:
            # إذا كان هناك سعر غير منطقي، اطلب من OpenAI إعادة تقدير السعر
            item = re.search(r"'(.*?)'", discrepancy).group(1)
            prompt = f"Re-estimate the price of '{item}' for a project in April 2025, ensuring the price is reasonable and based on current market data."
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
        elif "لم يتم جلب" in discrepancy:
            # إذا لم يتم جلب معدل التضخم أو الفائدة، اطلب من OpenAI إعادة المحاولة
            factor = "inflation rate" if "التضخم" in discrepancy else "interest rate"
            prompt = f"Fetch the {factor} for April 2025 based on global market data."
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

    return market_data, reasoning

# دالة لتقدير التكلفة باستخدام OpenAI (ReviewGPT)
def cautious_pricing(scope_results, market_results):
    total_cost = 0.0
    direct_costs = defaultdict(float)
    indirect_costs = defaultdict(float)
    reasoning = []

    scope_tasks, scope_direct, scope_indirect, missing_details = scope_results
    market_data, market_reasoning = market_results

    # التحقق من النتائج باستخدام Validate
    discrepancies = validate(scope_results, market_results)
    if discrepancies:
        market_data, arbitration_reasoning = arbitrate(discrepancies, scope_results, market_results)
        reasoning.extend(arbitration_reasoning)

    # تقدير التكاليف باستخدام OpenAI
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
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are ReviewGPT, an AI specializing in cost estimation and cautious pricing."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    cost_response = response.choices[0].message.content

    # معالجة رد OpenAI
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
                reasoning.append(f"تم تقدير تكلفة البند غير المباشر '{item}': ${cost:,.2f} بناءً على تحليل السوق.")
            except:
                reasoning.append(f"فشل في تقدير تكلفة البند غير المباشر: {line}")
        elif line.startswith("Total cost:"):
            try:
                total_cost = float(re.search(r'\d+\.?\d*', line).group())
                reasoning.append(f"تم تقدير التكلفة الإجمالية: ${total_cost:,.2f} بناءً على تحليل شامل.")
            except:
                reasoning.append("فشل في تقدير التكلفة الإجمالية.")

    reasoning.extend(market_reasoning)
    return total_cost, direct_costs, indirect_costs, reasoning

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

    # تحليل النطاق باستخدام ScopeGPT
    with st.spinner("جارٍ تحليل نطاق العمل..."):
        scope_results = analyze_scope(extracted_text)
    tasks, direct_cost_items, indirect_cost_items, missing_details = scope_results

    # عرض المهام المستخرجة
    st.subheader("المهام المستخرجة:")
    for task in tasks:
        st.write(f"- {task}")

    # عرض التفاصيل المفقودة
    if missing_details:
        st.subheader("التفاصيل المفقودة:")
        for detail in missing_details:
            st.write(f"- {detail}")

    # جلب بيانات السوق باستخدام MarketGPT
    with st.spinner("جارٍ جلب بيانات السوق..."):
        market_results = fetch_market_data(extracted_text, direct_cost_items, indirect_cost_items)

    # تقدير التكلفة باستخدام ReviewGPT
    if st.button("إظهار النتائج"):
        with st.spinner("جارٍ تقدير التكلفة..."):
            total_cost, direct_costs, indirect_costs, reasoning = cautious_pricing(scope_results, market_results)

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
# تحميل بيانات NLTK تلقائيًا عند بدء التطبيق
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# إعدادات OpenAI (استخدام أحدث إصدار من GPT)
openai.api_key = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-4-turbo"  # أحدث إصدار متاح في أبريل 2025

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
        if len(sentence.strip()) > 10:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            if any(tag.startswith('VB') for word, tag in tagged) and any(tag.startswith('NN') for word, tag in tagged):
                tasks.append(sentence.strip())
    return tasks

# دالة لتحليل نطاق العمل باستخدام OpenAI (ScopeGPT)
def analyze_scope(text):
    tasks = extract_tasks(text)
    direct_cost_items = []
    indirect_cost_items = []
    missing_details = []

    # تحديد البنود المباشرة وغير المباشرة باستخدام OpenAI
    prompt = f"""
    Analyze the following project scope text and identify direct and indirect cost items. Direct cost items are related to execution (e.g., equipment, labor), while indirect cost items are related to support (e.g., training, management). Also, identify any missing details that might affect cost estimation (e.g., location, timeline, budget, inflation, financing). Here is the text:\n\n{text}
    """
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are ScopeGPT, an AI specializing in analyzing project scopes for cost estimation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    analysis = response.choices[0].message.content

    # معالجة رد OpenAI
    lines = analysis.split('\n')
    for line in lines:
        line = line.lower().strip()
        if line.startswith("direct cost item:"):
            direct_cost_items.append(line.replace("direct cost item:", "").strip())
        elif line.startswith("indirect cost item:"):
            indirect_cost_items.append(line.replace("indirect cost item:", "").strip())
        elif line.startswith("missing detail:"):
            missing_details.append(line.replace("missing detail:", "").strip())

    return tasks, direct_cost_items, indirect_cost_items, missing_details

# دالة لجلب بيانات السوق باستخدام OpenAI (MarketGPT)
def fetch_market_data(text, direct_cost_items, indirect_cost_items):
    market_data = {
        "inflation_rate": None,
        "interest_rate": None,
        "material_prices": {},
        "market_stability": None
    }
    reasoning = []

    # جلب العوامل الاقتصادية باستخدام OpenAI
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
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are MarketGPT, an AI specializing in fetching market data for cost estimation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    market_response = response.choices[0].message.content

    # معالجة رد OpenAI
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

    return market_data, reasoning

# دالة للتحقق من النتائج بين النماذج (Validate)
def validate(scope_results, market_results):
    discrepancies = []
    scope_tasks, scope_direct, scope_indirect, _ = scope_results
    market_data, _ = market_results

    # التحقق من توافق البنود المباشرة مع أسعار المواد
    for item in scope_direct:
        item_lower = item.lower()
        for material, price in market_data["material_prices"].items():
            if material.lower() in item_lower and price <= 0:
                discrepancies.append(f"تناقض: سعر المادة '{material}' في البنود المباشرة ({item}) غير منطقي: ${price:,.2f}")

    # التحقق من العوامل الاقتصادية
    if market_data["inflation_rate"] is None:
        discrepancies.append("تناقض: لم يتم جلب معدل التضخم.")
    if market_data["interest_rate"] is None:
        discrepancies.append("تناقض: لم يتم جلب معدل الفائدة.")

    return discrepancies

# دالة للتحكيم بين النماذج (Arbitrate)
def arbitrate(discrepancies, scope_results, market_results):
    reasoning = []
    scope_tasks, scope_direct, scope_indirect, _ = scope_results
    market_data, market_reasoning = market_results

    # التحكيم في التناقضات
    for discrepancy in discrepancies:
        if "غير منطقي" in discrepancy:
            # إذا كان هناك سعر غير منطقي، اطلب من OpenAI إعادة تقدير السعر
            item = re.search(r"'(.*?)'", discrepancy).group(1)
            prompt = f"Re-estimate the price of '{item}' for a project in April 2025, ensuring the price is reasonable and based on current market data."
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
        elif "لم يتم جلب" in discrepancy:
            # إذا لم يتم جلب معدل التضخم أو الفائدة، اطلب من OpenAI إعادة المحاولة
            factor = "inflation rate" if "التضخم" in discrepancy else "interest rate"
            prompt = f"Fetch the {factor} for April 2025 based on global market data."
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

    return market_data, reasoning

# دالة لتقدير التكلفة باستخدام OpenAI (ReviewGPT)
def cautious_pricing(scope_results, market_results):
    total_cost = 0.0
    direct_costs = defaultdict(float)
    indirect_costs = defaultdict(float)
    reasoning = []

    scope_tasks, scope_direct, scope_indirect, missing_details = scope_results
    market_data, market_reasoning = market_results

    # التحقق من النتائج باستخدام Validate
    discrepancies = validate(scope_results, market_results)
    if discrepancies:
        market_data, arbitration_reasoning = arbitrate(discrepancies, scope_results, market_results)
        reasoning.extend(arbitration_reasoning)

    # تقدير التكاليف باستخدام OpenAI
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
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are ReviewGPT, an AI specializing in cost estimation and cautious pricing."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    cost_response = response.choices[0].message.content

    # معالجة رد OpenAI
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
                reasoning.append(f"تم تقدير تكلفة البند غير المباشر '{item}': ${cost:,.2f} بناءً على تحليل السوق.")
            except:
                reasoning.append(f"فشل في تقدير تكلفة البند غير المباشر: {line}")
        elif line.startswith("Total cost:"):
            try:
                total_cost = float(re.search(r'\d+\.?\d*', line).group())
                reasoning.append(f"تم تقدير التكلفة الإجمالية: ${total_cost:,.2f} بناءً على تحليل شامل.")
            except:
                reasoning.append("فشل في تقدير التكلفة الإجمالية.")

    reasoning.extend(market_reasoning)
    return total_cost, direct_costs, indirect_costs, reasoning

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

    # تحليل النطاق باستخدام ScopeGPT
    with st.spinner("جارٍ تحليل نطاق العمل..."):
        scope_results = analyze_scope(extracted_text)
    tasks, direct_cost_items, indirect_cost_items, missing_details = scope_results

    # عرض المهام المستخرجة
    st.subheader("المهام المستخرجة:")
    for task in tasks:
        st.write(f"- {task}")

    # عرض التفاصيل المفقودة
    if missing_details:
        st.subheader("التفاصيل المفقودة:")
        for detail in missing_details:
            st.write(f"- {detail}")

    # جلب بيانات السوق باستخدام MarketGPT
    with st.spinner("جارٍ جلب بيانات السوق..."):
        market_results = fetch_market_data(extracted_text, direct_cost_items, indirect_cost_items)

    # تقدير التكلفة باستخدام ReviewGPT
    if st.button("إظهار النتائج"):
        with st.spinner("جارٍ تقدير التكلفة..."):
            total_cost, direct_costs, indirect_costs, reasoning = cautious_pricing(scope_results, market_results)

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