import pandas as pd

from modules.data_profiler import (
    get_dataset_overview,
    get_column_information,
    get_missing_values,
    get_duplicate_info,
    get_numerical_statistics,
    get_categorical_statistics,
)

# Sample dataset
df = pd.read_csv("taxi_trip_pricing.csv")

print("\n--- Dataset Overview ---")
print(get_dataset_overview(df))

print("\n--- Column Information ---")
print(get_column_information(df))

print("\n--- Missing Values ---")
print(get_missing_values(df))

print("\n--- Duplicate Information ---")
print(get_duplicate_info(df))

print("\n--- Numerical Statistics ---")
print(get_numerical_statistics(df))

print("\n--- Categorical Statistics ---")
print(get_categorical_statistics(df))
