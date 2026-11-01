
import pandas as pd
from streamlit import metric


def get_correlation_analysis(df, method="pearson"):
    """
    Calculate correlations between numerical columns.
    """
    numerical_df = df.select_dtypes(include="number")

    if numerical_df.shape[1] < 2:
        return pd.DataFrame()

    return numerical_df.corr(method=method).round(3)


def get_outlier_analysis(df):
    """
    Detect outliers using the IQR method.
    """
    numerical_df = df.select_dtypes(include="number")

    results = []

    for column in numerical_df.columns:
        series = numerical_df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound) | (series > upper_bound)
        ]

        results.append({
            "Column": column,
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": len(outliers),
            "Outlier Percentage": round(
                len(outliers) / len(series) * 100, 2
            )
        })

    return pd.DataFrame(results)


def group_and_aggregate(df, group_by, metric, aggregation="mean"):
    """
    Group data by a column and calculate an aggregate.
    """
    if group_by not in df.columns:
        raise ValueError(f"Column not found: {group_by}")

    if metric not in df.columns:
        raise ValueError(f"Column not found: {metric}")

    allowed_aggregations = [
        "mean", "sum", "median", "min", "max", "count"
    ]

    if aggregation not in allowed_aggregations:
        raise ValueError("Unsupported aggregation function.")

    if aggregation != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError("Selected metric must be numerical.")

    result = (
        df.groupby(group_by, dropna=False)[metric]
        .agg(aggregation)
        .reset_index()
    )

    result[group_by] = result[group_by].astype("object")
    result[group_by] = result[group_by].where(
        result[group_by].notna(), "Missing"
    )

    return result.sort_values(
        by=metric, ascending=False
    ).reset_index(drop=True)


def filter_data(df, column, operator, value):
    """
    Filter dataset records using supported comparison operators.
    """
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    series = df[column]

    if operator == "contains":
        mask = series.astype("string").str.contains(
            str(value), case=False, na=False
        )

    elif operator in ["==", "!="]:
        if pd.api.types.is_numeric_dtype(series):
            value = pd.to_numeric(value)

        if operator == "==":
            mask = series == value
        else:
            mask = series != value

    elif operator in [">", ">=", "<", "<="]:
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                "Comparison operators require a numerical column."
            )

        value = float(value)

        if operator == ">":
            mask = series > value
        elif operator == ">=":
            mask = series >= value
        elif operator == "<":
            mask = series < value
        else:
            mask = series <= value

    else:
        raise ValueError("Unsupported filter operator.")

    return df.loc[mask].copy()
