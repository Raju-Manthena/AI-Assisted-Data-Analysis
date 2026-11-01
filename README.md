# AI-Assisted Data Analysis

An AI-powered data analysis application that combines traditional statistical analysis, SQL querying, and Generative AI to help users explore datasets and generate actionable insights.

## Overview

AI-Assisted Data Analysis is a Streamlit-based application that allows users to upload datasets and interact with their data through statistical analysis, visualizations, SQL queries, and natural-language questions.

The application integrates a single-agent tool-calling architecture using LangChain and Groq, allowing users to ask analytical questions without manually writing Python code or SQL queries.

## Features

### 1. Dataset Profiling
- Dataset overview, including rows, columns, and memory usage.
- Column information and data types.
- Missing value analysis.
- Duplicate record detection.
- Numerical and categorical statistical summaries.
- Outlier detection using the IQR method.

### 2. Data Visualization
- Generate visualizations by selecting columns.
- Supports bar charts, line charts, histograms, and box plots.
- Helps explore numerical and categorical data distributions.

### 3. SQL Querying
- Query uploaded datasets using SQL.
- Powered by DuckDB.
- Supports data filtering, aggregation, and other analytical queries.

### 4. AI-Powered Data Analysis
- Ask questions about uploaded datasets using natural language.
- A single AI agent dynamically selects relevant Python analytical tools.
- Generates responses based on computed results.
- Supports dataset profiling, correlation analysis, outlier detection, grouping, and filtering.

### 5. Automated Insights Generation
- Generates key findings from the uploaded dataset.
- Provides actionable recommendations.
- Uses Python-based statistical analysis combined with LLM-generated interpretations.
  
## System Architecture

The application is organized into two main components:

**1. Statistical Analysis**

- Performs dataset profiling, including missing value analysis, duplicate detection, and statistical summaries.
- Provides data visualizations to explore patterns and distributions.
- Supports SQL-based querying using DuckDB.

**2. AI-Powered Analysis**

- Uses a single-agent tool-calling architecture to answer natural-language questions about datasets.
- Dynamically selects Python analytical tools based on user queries.
- Generates automated insights, key findings, and recommendations by combining statistical results with LLM-based interpretation.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas |
| Data Visualization | Matplotlib |
| SQL Engine | DuckDB |
| Agent Framework | LangChain |
| LLM Integration | Groq |
| Language Model | Qwen |
| File Support | CSV, Excel |

## Project Structure

```text
AI-Assisted-Data-Analysis/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── agent.py
│   ├── analysis_tools.py
│   └── data_profiler.py
│
└── tests/
    ├── test_agent.py
    ├── test_analysis.py
    └── test_profiler.py
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- A Groq API key.

### 1. Clone the repository

```bash
git clone https://github.com/Raju-Manthena/AI-Assisted-Data-Analysis.git
cd AI-Assisted-Data-Analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Usage

1. Launch the application.
2. Upload a CSV or Excel dataset.
3. Explore the dataset using statistical profiling and visualizations.
4. Execute SQL queries against the uploaded data.
5. Ask natural-language questions through the AI Data Analyst Agent.
6. Generate automated findings and recommendations using AI Insights.

## Future Improvements

- Interactive visualizations generated from natural-language questions.
- Support for additional file formats and larger datasets.
- Enhanced analytical capabilities.
- Improved agent evaluation and testing.

## Author

**Satyanarayana Raju Manthena**

Integrated Dual Degree – Computer Science and Engineering  
Indian Institute of Technology Bhubaneswar

