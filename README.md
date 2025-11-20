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

# Behaviour & Attitudes
- Duration (preferred investment horizon: "<1 year", "1–3 years", etc.)
- Expect (expected annual returns: "10%-20%", "20%-30%", etc.)
- Invest_Monitor (how often they monitor investments)
- Objective (investment goals)
- Factor (key decision drivers)

# Motivations
Textual reasons such as:
- Reason_Mutual
- Reason_Bonds
- Reason_FD
- Reason_Equity

### Dataset Contains:
- **Demographics:** age, gender  
- **Investment avenues:** Mutual Funds, Equity, Debentures, Gold, PPF, Fixed Deposits, Government Bonds  
- **Behavioural attributes:** monitoring habits, investment duration  
- **Expected returns:** 10%–20%, 20%–30%, 30%–40%  
- **Motivations:** reasons for choosing each investment  
- **Information sources:** newspapers, consultants, television, online media

### Dataset Objective
To capture investor behaviour patterns, analyse financial preference trends, and support modelling tasks such as segmentation and personalised recommendation systems.

---

## 🎯 Project Objective
To design a personalised investment recommendation system that uses behavioural survey data to match investors with suitable financial products based on risk appetite, investment horizon, and expected returns. The aim is to deliver explainable, data-driven recommendations through a modular, reusable, and interactive Python application.

---

## 🚀 Key Features

### 🔹 Personalised Investment Recommendations
Generates investment suggestions tailored to user inputs (risk appetite, expected returns, and duration).

### 🔹 Hybrid Recommendation Logic
Combines statistical segmentation with rule-based risk scoring for more accurate recommendations.

### 🔹 Explainable AI Approach
Uses investor motivation data (e.g., “Assured Returns”, “Better Returns”) to justify recommendations.

### 🔹 Dual Interface System
- **Command-Line Interface (CLI)**
- **Streamlit Web Application**

### 🔹 Modular Architecture
Contains a reusable **InvestmentRecommender** class that handles data cleaning, segmentation, scoring, and explanation modules.

---

## 🛠 Tech Stack
- **Python 3**
- **Pandas**
- **NumPy**
- **Streamlit**
- **CLI (built-in Python)**

