import os
import re
import hashlib
from io import BytesIO

import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from modules.agent import build_analysis_agent
from modules.data_profiler import (
    get_dataset_overview,
    get_column_information,
    get_missing_values,
    get_duplicate_info,
    get_numerical_statistics,
    get_categorical_statistics,
)
from modules.analysis_tools import (
    get_correlation_analysis,
    get_outlier_analysis,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)

st.title("AI Data Analyst Agent")
st.caption(
    "Explore datasets, ask questions, and discover meaningful insights using AI."
)


# ==================================================
# API CONFIGURATION
# ==================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

if not groq_api_key:
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
    )


# ==================================================
# FILE UPLOAD - ONE FILE AT A TIME
# ==================================================

st.markdown(
    """
    <style>
    [data-testid="stFileUploader"] button[aria-label="Add files"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload one CSV or Excel dataset",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    key="dataset_uploader",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    dataset_id = hashlib.sha256(file_bytes).hexdigest()

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))

    if df.empty:
        st.warning("The uploaded dataset contains no records.")
        st.stop()

    if len(df.columns) == 0:
        st.warning("The uploaded dataset has no columns.")
        st.stop()

except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()


st.success(f"Successfully loaded: {uploaded_file.name}")


# ==================================================
# RESET SESSION DATA WHEN DATASET CHANGES
# ==================================================

if st.session_state.get("dataset_id") != dataset_id:
    st.session_state.dataset_id = dataset_id
    st.session_state.agent_messages = []
    st.session_state.chat_history = []
    st.session_state.agent = None
    st.session_state.agent_key = None
    st.session_state.generated_insights = None

st.session_state.setdefault("agent_messages", [])
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("generated_insights", None)


# ==================================================
# DATASET PREVIEW
# ==================================================

st.subheader("Dataset Preview")
st.caption("Showing the first 10 rows of the uploaded dataset.")
st.table(df.head(10))


# ==================================================
# LLM INITIALIZATION
# ==================================================

agent = None
insights_llm = None

if groq_api_key:
    try:
        if (
            st.session_state.get("agent") is None
            or st.session_state.get("agent_key") != groq_api_key
        ):
            agent_llm = ChatGroq(
                model="qwen/qwen3.8-27b",
                api_key=groq_api_key,
                temperature=0,
                max_tokens=512,
            )

            st.session_state.agent = build_analysis_agent(df, agent_llm)
            st.session_state.agent_key = groq_api_key

        agent = st.session_state.agent

        insights_llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=groq_api_key,
            temperature=0,
            max_tokens=900,
        )

    except Exception as e:
        st.error(f"AI initialization failed: {e}")


# ==================================================
# MAIN NAVIGATION
# ==================================================

section = st.radio(
    "Analysis Mode",
    ["Statistical Analysis", "AI-Powered Analysis"],
    horizontal=True,
    label_visibility="collapsed",
    key="analysis_mode",
)


# ==================================================
# STATISTICAL ANALYSIS
# ==================================================

if section == "Statistical Analysis":

    profile_tab, visualization_tab, sql_tab = st.tabs(
        ["Dataset Profile", "Visualizations", "SQL Query"]
    )

    # --------------------------------------------------
    # DATASET PROFILE
    # --------------------------------------------------

    with profile_tab:
        st.subheader("Dataset Overview")

        overview = get_dataset_overview(df)
        duplicates = get_duplicate_info(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{overview['rows']:,}")
        col2.metric("Columns", f"{overview['columns']:,}")
        col3.metric("Duplicate Rows", f"{duplicates['duplicate_rows']:,}")
        col4.metric("Memory Usage", f"{overview['memory_usage_mb']} MB")

        st.divider()

        st.subheader("Column Information")
        st.dataframe(
            get_column_information(df),
            use_container_width=True,
        )

        st.subheader("Missing Values")
        missing = get_missing_values(df)

        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing, use_container_width=True)

        st.subheader("Numerical Statistics")
        numerical_stats = get_numerical_statistics(df)

        if numerical_stats.empty:
            st.info("No numerical columns found.")
        else:
            st.dataframe(numerical_stats, use_container_width=True)

        st.subheader("Categorical Statistics")
        categorical_stats = get_categorical_statistics(df)

        if categorical_stats.empty:
            st.info("No categorical columns found.")
        else:
            st.dataframe(categorical_stats, use_container_width=True)

        st.subheader("Outlier Analysis")
        outliers = get_outlier_analysis(df)

        if outliers.empty:
            st.info("No numerical columns available.")
        else:
            st.dataframe(outliers, use_container_width=True)

    # --------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------

    with visualization_tab:
        st.subheader("Data Visualization")

        column = st.selectbox(
            "Select column to visualize",
            options=list(df.columns),
            key="visualization_column",
        )

        is_numeric = pd.api.types.is_numeric_dtype(df[column])

        if is_numeric:
            chart_options = ["Histogram", "Bar", "Line", "Box"]
        else:
            chart_options = ["Bar"]

        chart_type = st.selectbox(
            "Chart Type",
            options=chart_options,
            key="visualization_type",
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        if chart_type == "Histogram":
            df[column].dropna().hist(ax=ax, bins=20)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

        elif chart_type == "Bar":
            df[column].value_counts(dropna=False).head(20).plot(
                kind="bar",
                ax=ax,
            )
            ax.set_title(f"Frequency of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")

        elif chart_type == "Line":
            df[column].plot(ax=ax)
            ax.set_title(f"{column} over observations")
            ax.set_xlabel("Observation")
            ax.set_ylabel(column)

        elif chart_type == "Box":
            df[[column]].plot(kind="box", ax=ax)
            ax.set_title(f"Box Plot of {column}")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------
    # SQL QUERY
    # --------------------------------------------------

    with sql_tab:
        st.subheader("Query Your Dataset Using SQL")
        st.write("Use the table name `data` to query your dataset.")

        sql = st.text_area(
            "SQL Query",
            "SELECT * FROM data LIMIT 5",
            height=120,
            key="sql_query",
        )

        if st.button("Run SQL Query"):
            query = sql.strip().rstrip(";").strip()

            is_select = bool(
                re.match(r"^SELECT\b", query, flags=re.IGNORECASE)
            )
            has_multiple_statements = ";" in query

            if not query:
                st.warning("Enter a SQL query.")
            elif not is_select or has_multiple_statements:
                st.error("Only a single SELECT query is allowed.")
            else:
                con = None
                try:
                    con = duckdb.connect(
                        database=":memory:",
                        config={"enable_external_access": "false"},
                    )
                    con.register("data", df)
                    result = con.execute(query).df()

                    st.success(
                        f"Query executed successfully. {len(result)} rows returned."
                    )
                    st.dataframe(
                        result.head(1000),
                        use_container_width=True,
                    )

                    if len(result) > 1000:
                        st.info("Displaying the first 1,000 rows.")

                except Exception as e:
                    st.error(f"SQL execution failed: {e}")
                finally:
                    if con is not None:
                        con.close()


# ==================================================
# AI-POWERED ANALYSIS
# ==================================================

else:

    questions_tab, insights_tab = st.tabs(
        ["Ask Questions", "AI Insights"]
    )

    # --------------------------------------------------
    # ASK QUESTIONS
    # --------------------------------------------------

    with questions_tab:
        st.subheader("Ask Questions About Your Data")
        st.caption(
            "The AI agent selects analytical tools to answer questions about your dataset."
        )

        if agent is None:
            st.info("Enter your Groq API key to use AI-powered analysis.")
        else:
            if st.button("Clear Conversation", key="clear_conversation"):
                st.session_state.agent_messages = []
                st.session_state.chat_history = []
                st.rerun()

            # Render the conversation first so the input stays below the
            # latest response, including after a new question is submitted.
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input(
                "Ask a question about your dataset..."
            )

            if question:
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )

                with st.spinner("Analyzing your data..."):
                    try:
                        messages = (
                            st.session_state.agent_messages
                            + [{"role": "user", "content": question}]
                        )

                        response = agent.invoke({"messages": messages})
                        answer = response["messages"][-1].content
                        st.session_state.agent_messages = response["messages"]

                    except Exception as e:
                        answer = f"Analysis failed: {e}"

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

                # Rerun to render both new messages before the chat input.
                st.rerun()

    # --------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------

    with insights_tab:
        st.subheader("AI-Generated Insights")
        st.caption(
            "Discover important patterns and practical recommendations from your dataset."
        )

        if insights_llm is None:
            st.info("Enter your Groq API key to generate AI insights.")

        else:
            if st.button("Generate AI Insights", key="generate_insights"):
                with st.spinner("Analyzing your data..."):
                    try:
                        # Code-based calculations
                        numerical_stats = get_numerical_statistics(df)
                        categorical_stats = get_categorical_statistics(df)
                        missing = get_missing_values(df)
                        outliers = get_outlier_analysis(df)
                        correlation = get_correlation_analysis(df)

                        correlation_pairs = []

                        if not correlation.empty:
                            for i in range(len(correlation.columns)):
                                for j in range(i + 1, len(correlation.columns)):
                                    value = correlation.iloc[i, j]

                                    if pd.notna(value):
                                        correlation_pairs.append(
                                            {
                                                "Variables": (
                                                    f"{correlation.columns[i]} "
                                                    f"and {correlation.columns[j]}"
                                                ),
                                                "Correlation": round(value, 3),
                                            }
                                        )

                        correlation_pairs.sort(
                            key=lambda item: abs(item["Correlation"]),
                            reverse=True,
                        )

                        numerical_text = (
                            numerical_stats.head(15).to_string()
                            if not numerical_stats.empty
                            else "No numerical columns."
                        )
                        categorical_text = (
                            categorical_stats.head(15).to_string(index=False)
                            if not categorical_stats.empty
                            else "No categorical columns."
                        )
                        missing_text = (
                            missing.head(15).to_string()
                            if not missing.empty
                            else "No missing values."
                        )
                        outlier_text = (
                            outliers.head(15).to_string(index=False)
                            if not outliers.empty
                            else "No numerical outlier information."
                        )
                        correlation_text = (
                            str(correlation_pairs[:8])
                            if correlation_pairs
                            else "No numerical correlations available."
                        )

                        # LLM interpretation of computed results
                        insights_prompt = f"""
You are an experienced data analyst.

Analyze the computed statistics below and generate useful insights.

NUMERICAL STATISTICS:
{numerical_text}

CATEGORICAL STATISTICS:
{categorical_text}

CORRELATION ANALYSIS:
{correlation_text}

OUTLIER ANALYSIS:
{outlier_text}

MISSING VALUES:
{missing_text}

YOUR TASK:
Write BOTH sections below. Do not omit either section.

# Key Findings
- Give exactly 3 meaningful findings.
- Each finding should be concise and supported by the supplied statistics.

# Business Recommendations
- Give exactly 3 practical recommendations.
- Each recommendation must be a complete, actionable sentence.
- Link each recommendation to one or more findings.
- If the dataset does not establish a direct business context, recommend a practical next analysis or data-quality action instead.

INSTRUCTIONS:
- Use actual column names and computed values where relevant.
- Do not invent statistics or claim correlation proves causation.
- Do not repeat row count, column count, or memory usage.
- Avoid long introductions and conclusions.
- Keep each finding and recommendation to 1–2 sentences.
- Finish all 3 recommendations. Never stop after the Key Findings section.
"""

                        response = insights_llm.invoke(insights_prompt)
                        insights = response.content

                        st.session_state.generated_insights = insights

                    except Exception as e:
                        st.error(f"Could not generate insights: {e}")

            if st.session_state.generated_insights:
                st.markdown(st.session_state.generated_insights)
