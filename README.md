# Investment-Recommendation-Engine
## 🏷️ Project Title
**Investment Recommendation Engine Using Python (Hybrid ML + Rule-Based System)**
---

## 📝 Short Description
This project is a full end-to-end Investment Recommendation Engine built with Python, designed to help individuals identify the most suitable investment avenues based on their profile, behaviour, and financial goals.

Unlike generic recommendation systems, this project blends:

- **Data-driven analytics — using actual investor survey patterns
- **Risk-based rule logic — mirroring how financial advisors think
- **Explainable insights — visual breakdowns of why recommendations were chosen

---

## 📌 Project Overview
This project delivers an end-to-end **Investment Recommendation Engine** developed in Python. Using a dataset of investor behaviours and financial preferences,
the system evaluates user inputs such as investment duration, expected returns, and risk appetite to generate **data-backed investment recommendations**.

The recommendation engine integrates:

- **Segment-based statistical analysis** (mean preference scoring)
- **Rule-based risk modelling**
- **Explanation generation** via investor motivation fields
- **Two user interfaces:**
  - Command-Line Interface (CLI)
  - Streamlit Web Application

The goal is to create a transparent, user-friendly tool that mirrors real-world financial advisory logic using data science techniques.

---

## 📁 Data Source
this data was gotten from kaggle
-[download here](https://www.kaggle.com/datasets/ayeshasiddiqa123/finance-trends-2020-2025?resource=download)

**File:** `Finance_Trends.csv`  
**Type:** Financial Behaviour & Investment Preference Survey  

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

