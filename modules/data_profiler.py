
import pandas as pd


def get_dataset_overview(df):
    """
    Return basic information about the dataset.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_cells": df.size,
        "memory_usage_mb": float(round(
    df.memory_usage(deep=True).sum() / (1024 ** 2), 2
))
    }


def get_column_information(df):
    """
    Return information about each column.
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notna().sum().values,
        "Null Count": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })


def get_missing_values(df):
    """
    Return missing-value counts and percentages.
    """
    missing_count = df.isna().sum()

    result = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": (missing_count / len(df) * 100)
        if len(df) > 0 else 0
    })

    return result[result["Missing Count"] > 0].sort_values(
        by="Missing Count", ascending=False
    )


def get_duplicate_info(df):
    """
    Return the number and percentage of duplicate rows.
    """
    duplicate_count = int(df.duplicated().sum())

    duplicate_percentage = (
        duplicate_count / len(df) * 100 if len(df) > 0 else 0
    )

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2)
    }


def get_numerical_statistics(df):
    """
    Return descriptive statistics for numerical columns.
    """
    numerical_df = df.select_dtypes(include="number")

    if numerical_df.empty:
        return pd.DataFrame()

    return numerical_df.describe().T.round(2)


def get_categorical_statistics(df):
    """
    Return basic statistics for categorical columns.
    """
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    if categorical_df.empty:
        return pd.DataFrame()

    results = []

    for column in categorical_df.columns:
        mode_values = categorical_df[column].mode(dropna=True)

        results.append({
            "Column": column,
            "Unique Values": categorical_df[column].nunique(),
            "Most Frequent Value": (
                mode_values.iloc[0] if not mode_values.empty else None
            ),
            "Frequency": (
                int(categorical_df[column].value_counts().iloc[0])
                if not categorical_df[column].value_counts().empty else 0
            )
        })

    return pd.DataFrame(results)
