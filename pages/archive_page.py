import streamlit as st

# ملاحظات هامة للمطور:
# هذا الملف مخصص لصفحة أرشفة وتدريب الأسعار التاريخية فقط.
# ملف train_costimaize.py موجود في الأرشيف وسيُستخدم لتدريب النظام على بيانات الأسعار التاريخية.

st.title("أرشفة وتدريب الأسعار التاريخية")
st.write("سيتم استخدام ملف `train_costimaize.py` لتدريب النظام.")
st.write("الرجاء رفع بيانات الأسعار التاريخية لاحقًا.")

# عرض السجل وزر التحميل (مشترك عبر st.session_state)
st.subheader("سجل التنفيذ")
if st.session_state["execution_log"]:
    log_text = "\n".join(st.session_state["execution_log"])
    st.text_area("سجل العمليات", log_text, height=200, key="log_text_area_archive")
    st.download_button(
        label="تحميل سجل التنفيذ",
        data=log_text,
        file_name="execution_log.txt",
        mime="text/plain",
        key="download_log_archive"
    )
else:
    st.write("لا توجد عمليات مسجلة بعد.")
