import streamlit as st
from datetime import datetime

# ملاحظات هامة للمطور:
# هذا الملف هو الصفحة الرئيسية للتطبيق. لا يحتوي على منطق الخدمات (مثل تقدير التكلفة) لأنها موجودة في صفحات منفصلة.

st.title("CostimAIzer - تقدير التكاليف")
st.subheader("الصفحة الرئيسية")

# Dashboard
st.subheader("إحصائيات الأعمال")
st.write("عدد التقديرات: 10")  # مثال
st.write("متوسط التكلفة: 45,000 ريال")  # مثال

# سجل العمليات (مشترك بين جميع الصفحات عبر st.session_state)
if "execution_log" not in st.session_state:
    st.session_state["execution_log"] = []

def log_action(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["execution_log"].append(f"[{timestamp}] {action}")

# عرض السجل وزر التحميل
st.subheader("سجل التنفيذ")
if st.session_state["execution_log"]:
    log_text = "\n".join(st.session_state["execution_log"])
    st.text_area("سجل العمليات", log_text, height=200, key="log_text_area_main")
    st.download_button(
        label="تحميل سجل التنفيذ",
        data=log_text,
        file_name="execution_log.txt",
        mime="text/plain",
        key="download_log_main"
    )
else:
    st.write("لا توجد عمليات مسجلة بعد.")