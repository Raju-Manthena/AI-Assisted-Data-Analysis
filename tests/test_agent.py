
import os
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from modules.agent import build_analysis_agent


# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")


# Load dataset
df = pd.read_csv("taxi_trip_pricing.csv")

print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")


# Initialize Groq
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=api_key,
    temperature=0,
    max_tokens=512
)


# Build agent
agent = build_analysis_agent(df, llm)

print("\nAgent initialized successfully.")


# Questions to test
questions = [
    "How many records are present in the dataset?",
    "Are there any duplicate records in the dataset?",
    "What is the relationship between trip distance and trip price?",
    "What is the average trip price for each time of day?"
]


# Execute questions
for i, question in enumerate(questions, start=1):

    print("\n" + "=" * 70)
    print(f"TEST {i}: {question}")
    print("=" * 70)

    try:
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        answer = response["messages"][-1].content

        print("\nANSWER:")
        print(answer)

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
