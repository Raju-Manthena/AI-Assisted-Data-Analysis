
import pandas as pd

from modules.analysis_tools import (
    get_correlation_analysis,
    get_outlier_analysis,
    group_and_aggregate,
    filter_data
)

# Load the taxi dataset
df = pd.read_csv("taxi_trip_pricing.csv")


# 1. Correlation Analysis
print("\n--- Correlation Analysis ---")

correlation = get_correlation_analysis(df)
print(correlation)


# 2. Outlier Analysis
print("\n--- Outlier Analysis ---")

outliers = get_outlier_analysis(df)
print(outliers)


# 3. Group and Aggregate
print("\n--- Average Trip Price by Time of Day ---")

grouped = group_and_aggregate(
    df,
    group_by="Time_of_Day",
    metric="Trip_Price",
    aggregation="mean"
)

print(grouped)


# 4. Data Filtering
print("\n--- Trips with Distance Greater Than 50 km ---")

filtered = filter_data(
    df,
    column="Trip_Distance_km",
    operator=">",
    value=50
)

print(filtered.head(10))

print("\nNumber of matching trips:", len(filtered))


result = filter_data(
    df,
    column="Weather",
    operator="==",
    value="Clear"
)

print("\nClear weather records:", len(result))



result = filter_data(
    df,
    column="Time_of_Day",
    operator="contains",
    value="morning"
)

print("\nMorning trips:", len(result))



result = group_and_aggregate(
    df,
    group_by="Time_of_Day",
    metric="Trip_Price",
    aggregation="median"
)

print("\nMedian fare by time of day:")
print(result)
