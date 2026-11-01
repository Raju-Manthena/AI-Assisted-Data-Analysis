
import pandas as pd

from langchain.agents import create_agent
from langchain.tools import tool

from modules.data_profiler import (
    get_dataset_overview,
    get_column_information,
    get_missing_values,
    get_duplicate_info,
    get_numerical_statistics,
    get_categorical_statistics
)

from modules.analysis_tools import (
    get_correlation_analysis,
    get_outlier_analysis,
    group_and_aggregate,
    filter_data
)


def build_analysis_agent(df, llm):
    """
    Build a data analyst agent with controlled analytical tools.
    """

    @tool
    def dataset_overview() -> str:
        """Get the number of rows, columns, total cells and memory usage."""
        result = get_dataset_overview(df)
        return str(result)

    @tool
    def column_information() -> str:
        """Get column names, data types, non-null counts, missing counts and unique values."""
        result = get_column_information(df)
        return result.to_string(index=False)

    @tool
    def missing_values() -> str:
        """Identify missing values and their percentages in each column."""
        result = get_missing_values(df)

        if result.empty:
            return "No missing values found in the dataset."

        return result.to_string()

    @tool
    def duplicate_analysis() -> str:
        """Count duplicate rows and calculate their percentage in the dataset."""
        result = get_duplicate_info(df)
        return str(result)

    @tool
    def numerical_statistics() -> str:
        """Calculate count, mean, standard deviation, minimum, quartiles and maximum for numerical columns."""
        result = get_numerical_statistics(df)

        if result.empty:
            return "No numerical columns found."

        return result.to_string()

    @tool
    def categorical_statistics() -> str:
        """Summarize categorical columns, including unique values and most frequent categories."""
        result = get_categorical_statistics(df)

        if result.empty:
            return "No categorical columns found."

        return result.to_string(index=False)

    @tool
    def correlation_analysis(method: str = "pearson") -> str:
        """Calculate correlations between numerical columns. Supported methods: pearson, spearman and kendall."""
        if method not in ["pearson", "spearman", "kendall"]:
            return "Invalid method. Use pearson, spearman or kendall."

        result = get_correlation_analysis(df, method)

        if result.empty:
            return "Not enough numerical columns for correlation analysis."

        return result.to_string()

    @tool
    def outlier_analysis() -> str:
        """Detect numerical outliers using the IQR method and report their counts and percentages."""
        result = get_outlier_analysis(df)

        if result.empty:
            return "No numerical columns available for outlier analysis."

        return result.to_string(index=False)

    @tool
    def group_analysis(
        group_by: str,
        metric: str,
        aggregation: str = "mean"
    ) -> str:
        """Group records by a column and calculate mean, sum, median, minimum, maximum or count for a metric."""
        try:
            result = group_and_aggregate(
                df, group_by, metric, aggregation
            )

            return (
                f"Grouping column: {group_by}\n"
                f"Metric: {metric}\n"
                f"Aggregation: {aggregation}\n"
                f"Number of groups: {len(result)}\n\n"
                f"Complete results (include every group):\n"
                f"{result.to_string(index=False)}"
            )

        except Exception as e:
            return f"Analysis failed: {str(e)}"

    @tool
    def data_filter(
        column: str,
        operator: str,
        value: str
    ) -> str:
        """Filter records using ==, !=, >, >=, <, <= or contains. Return matching record count and up to 20 matching rows."""
        try:
            result = filter_data(df, column, operator, value)

            output = (
                f"Matching records: {len(result)}\n\n"
                f"First 20 matching records:\n"
                f"{result.head(20).to_string(index=False)}"
            )

            return output

        except Exception as e:
            return f"Filtering failed: {str(e)}"

    tools = [
        dataset_overview,
        column_information,
        missing_values,
        duplicate_analysis,
        numerical_statistics,
        categorical_statistics,
        correlation_analysis,
        outlier_analysis,
        group_analysis,
        data_filter
    ]

    system_prompt = """
    You are an AI Data Analyst.

    Your task is to answer questions about the uploaded dataset accurately.

    Instructions:

    1. Use the available analytical tools to retrieve actual dataset information.
    2. Never invent numerical results or dataset statistics.
    3. Use the appropriate tool for each question.
    4. For dataset overview questions, use the overview, column
       information, missing values and numerical statistics tools.
    5. For duplicate questions, use duplicate_analysis.
    6. For correlation questions, use correlation_analysis.
    7. For outlier questions, use outlier_analysis.
    8. For grouping and filtering questions, use the respective tools.
    9. Explain the results clearly in simple language.
    10. If a requested analysis cannot be performed, explain why.
    11. Do not claim that correlation proves causation.
    12. Do not make business recommendations unsupported by the data.
    13. When reporting grouped results, include every group returned
        by the tool, including groups labelled "Missing".
    14. Never claim missing categories were excluded unless the tool
        explicitly indicates that they were excluded.
    15. Do not omit categories simply because they represent missing values.
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent
