if st.session_state.page == "estimate_cost":
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

        # زر "Proceed to Estimate Cost" مع تحسين التحكم في الضغط المتكرر
        if st.button("Proceed to Estimate Cost", key="proceed_button", disabled=st.session_state.is_processing):
            if not st.session_state.is_processing:
                st.session_state.is_processing = True
                st.info("Processing... Please wait.")
                try:
                    with st.spinner("Estimating cost..."):
                        result = estimator.analyze_and_estimate_multi_gpt(task_description)
                        st.session_state.estimation_result = result
                        st.session_state.page = "estimation_result"
                        st.experimental_rerun()  # تحديث الواجهة فورًا
                except Exception as e:
                    logger.error("Error during estimation: %s", str(e))
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    st.session_state.is_processing = False
                    st.experimental_rerun()  # تحديث الحالة
            else:
                st.warning("Processing is already in progress. Please wait.")