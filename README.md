# Investment-Recommendation-Engine
## 🏷️ Project Title
**Investment Recommendation Engine Using Python (Hybrid ML + Rule-Based System)**
---

## Project Overview
This project is a full end-to-end Investment Recommendation Engine built with Python, designed to help individuals identify the most suitable investment avenues based on their profile, behaviour, and financial goals.

Unlike generic recommendation systems, this project blends:

- Data-driven analytics — using actual investor survey patterns
- Risk-based rule logic — mirroring how financial advisors think
- Explainable insights — visual breakdowns of why recommendations were chosen

The engine takes simple user inputs such as age, investment duration, expected returns, and risk appetite, then intelligently matches them to similar investor profiles from the dataset. From there, it generates:

- personalised investment suggestions
- visual comparisons of “base” vs “risk-adjusted” scores
- transparent logic behind each recommendation

I built this project to demonstrate how analytics, business logic, and user-centric thinking can be combined into a clear, practical tool. It reflects my passion for transforming raw data into meaningful insights and helping people make better financial decisions.

---

## 📌 Dataset Overview
The project is powered by Finance_Trends.csv, a large dataset containing real survey responses about investor preferences, behaviour, and financial motivations.

---

## 📁 Data Source
This dataset was provided as part of a financial behaviour research activity and captures a wide cross-section of retail investors `Finance_Trends.csv`. Although the dataset was gotten from kaggle, it mirrors the structure of typical investor profiling research used in financial services.
-[download here](https://www.kaggle.com/datasets/ayeshasiddiqa123/finance-trends-2020-2025?resource=download)

# Dashboard
Finance Trends <img width="1470" height="956" alt="Screenshot 2025-11-20 at 16 19 15" src="https://github.com/user-attachments/assets/4828c2d5-b999-4f99-87c6-861b20adcb05" />


## 📐 Size & Structure
- Rows: ~12,000+ individual responses
- Columns: 24 variables (mix of numeric ratings & categorical fields)

## 🧩 Key Features & Variables
The dataset captures a wide array of investment-related attributes, including:

### Demographics
- gender
- age

### Investment Preference Scores
Numeric ratings (1–10 scale):
- Mutual_Funds
- Equity_Market
- Debentures
- Government_Bonds
- Fixed_Deposits
- PPF
- Gold

These represent how strongly each investor prefers a given product.

### Behaviour & Attitudes
- Duration (preferred investment horizon: "<1 year", "1–3 years", etc.)
- Expect (expected annual returns: "10%-20%", "20%-30%", etc.)
- Invest_Monitor (how often they monitor investments)
- Objective (investment goals)
- Factor (key decision drivers)

## 🧼 Data Cleaning & Preprocessing
To ensure reliable analysis, several preprocessing steps were applied:
- Stripped whitespace from all string-based columns
- Converted mixed-type fields into consistent categorical values
- Normalised duration and expected-return fields for mapping
- Handled potentially inconsistent text fields in reasons & motivations
- Verified numeric integrity of preference scores

## 💡 Interesting Dataset Insights
During exploration, several patterns emerged:
- Equity & Mutual Funds consistently scored highest among younger, high-risk investors.
- Fixed Deposits, PPF, and Government Bonds were most preferred by low-risk and short-duration investors.
- Higher expected returns (30%+) were strongly associated with increased equity market preferences.
- Long-term investors (>5 years) leaned more towards Gold and Mutual Funds.
- Clear behavioural clustering exists based on duration, expected return, and risk appetite—ideal for a recommendation engine.

## 🎯 Project Objectives
This project was built with several clear, practical goals in mind:
### Build an Explainable Investment Recommendation Engine
Create a system that doesn’t just recommend — but shows why it recommends specific investment products.
### Combine Data-Driven Insights with Realistic Financial Logic
Blend machine-learning-style segmentation with rule-based financial reasoning, replicating how real advisors assess risk.
### Provide an Interactive, User-Friendly Experience
Allow users to input details such as age, duration, expected return, and risk appetite via a CLI or notebook interface.
### Deliver Clear Visual Insights for Stakeholders
Generate visual breakdowns (bar charts, grouped comparisons, etc.) that help users understand the scoring behaviour behind the scenes.
### Demonstrate Practical Python, Data Science, and Analytical Skills
Showcase a real-world project employing:
- Pandas
- NumPy
- Seaborn / Matplotlib
- OOP design principles
- Data cleaning & preprocessing
- User input handling
- Business and financial reasoning
