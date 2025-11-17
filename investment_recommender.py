##########################
from __future__ import annotations
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

# ---------- Helper mapping functions ----------
def map_duration_years_to_category(years: float) -> str:
    """
    Map numeric duration in years to the categorical duration labels 
    used in the dataset.
    """
    if years < 1:
        return "Less than 1 year"
    elif years < 3:
        return "1-3 years"
    elif years < 5:
        return "3-5 years"
    else:
        return "More than 5 years"

def map_expected_return_to_category(expected_return_pct: float) -> str:
    """
    Map numeric expected return (e.g., 15 for 15%) to the 
    categorical return ranges used in the dataset.
    """
    if expected_return_pct < 20:
        return "10%-20%"
    elif expected_return_pct < 30:
        return "20%-30%"
    else:
        return "30%-40%"

# ---------- Core Recommendation Class ----------
class InvestmentRecommender:
    """
    A simple hybrid (data-driven + rule-based) investment recommendation engine
    built on top of survey data from Finance_Trends.csv.
    """

    INVESTMENT_COLS = [
        "Mutual_Funds",
        "Equity_Market",
        "Debentures",
        "Government_Bonds",
        "Fixed_Deposits",
        "PPF",
        "Gold",
    ]

    # Map avenue to reason column (where available)
    REASON_COL_MAP = {
        "Mutual_Funds": "Reason_Mutual",
        "Equity_Market": "Reason_Equity",
        "Debentures": "Reason_Bonds",
        "Fixed_Deposits": "Reason_FD",
        "Government_Bonds": None,
        "PPF": None,
        "Gold": None,
    }

    def __init__(self, df: pd.DataFrame):
        self.df = self._clean_dataframe(df)

    @classmethod
    def from_csv(cls, path: str) -> "InvestmentRecommender":
        """Convenience constructor to load data directly from CSV."""
        df = pd.read_csv(path)
        return cls(df)

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text_cols = ["Duration", "Expect", "gender", "Investment_Avenues", "Avenue"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    def describe_data(self) -> None:
        print("📊 Dataset summary:")
        print(f"Rows: {len(self.df):,}")
        print("\nDuration distribution:")
        print(self.df["Duration"].value_counts(), "\n")
        print("Expected return distribution:")
        print(self.df["Expect"].value_counts(), "\n")

    def _get_segment(self, duration_cat: str, expect_cat: str) -> pd.DataFrame:
        seg = self.df[
            (self.df["Duration"] == duration_cat)
            & (self.df["Expect"] == expect_cat)
        ]
        if seg.empty:
            print("⚠️ No exact matches for this segment. Falling back to the entire dataset.\n")
            return self.df
        else:
            print(f"Matched {len(seg):,} similar investors in the dataset.\n")
            return seg

    def _compute_segment_scores(self, seg: pd.DataFrame) -> Dict[str, float]:
        means = seg[self.INVESTMENT_COLS].mean()
        return means.sort_values(ascending=False).to_dict()

    @staticmethod
    def _apply_risk_adjustments(base_scores: Dict[str, float], risk_level: str) -> Dict[str, float]:
        risk_level = risk_level.lower().strip()
        risk_boost = {
            "low": {"Fixed_Deposits": 1.0, "PPF": 1.0, "Government_Bonds": 0.7},
            "medium": {"Mutual_Funds": 0.5, "Government_Bonds": 0.5, "PPF": 0.5},
            "high": {"Equity_Market": 1.0, "Mutual_Funds": 0.7, "Gold": 0.5},
        }
        adjusted = base_scores.copy()
        if risk_level in risk_boost:
            for avenue, bonus in risk_boost[risk_level].items():
                if avenue in adjusted:
                    adjusted[avenue] = adjusted[avenue] + bonus
        return adjusted

    def _explain_recommendation(self, avenue: str, seg: pd.DataFrame) -> str:
        reason_col = self.REASON_COL_MAP.get(avenue)
        if not reason_col or reason_col not in seg.columns:
            return "This avenue scored highly among similar investors in terms of preference and suitability."
        reasons = seg[reason_col].dropna().astype(str).str.strip()
        if reasons.empty:
            return "This avenue scored highly among similar investors, but no specific reasons were recorded in the data."
        top_reason = reasons.value_counts().idxmax()
        return f"Common reason among similar investors: '{top_reason}'."

    def recommend(
        self, age: int, gender: str, duration_years: float,
        expected_return_pct: float, risk_level: str = "medium",
        top_n: int = 3, verbose: bool = True
    ) -> List[Tuple[str, float]]:
        gender = str(gender).strip().title()
        risk_level = str(risk_level).strip().lower()
        duration_cat = map_duration_years_to_category(duration_years)
        expect_cat = map_expected_return_to_category(expected_return_pct)

        if verbose:
            print("📌 Interpreted investor profile:")
            print(f"   Age: {age}")
            print(f"   Gender: {gender}")
            print(f"   Duration preference: {duration_cat}")
            print(f"   Expected return: {expect_cat}")
            print(f"   Risk level: {risk_level}\n")

        seg = self._get_segment(duration_cat, expect_cat)
        base_scores = self._compute_segment_scores(seg)
        adjusted_scores = self._apply_risk_adjustments(base_scores, risk_level)
        sorted_recs = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
        top_recs = sorted_recs[:top_n]

        if verbose:
            print("✅ Recommended investment avenues:\n")
            for avenue, score in top_recs:
                base = base_scores.get(avenue, np.nan)
                explanation = self._explain_recommendation(avenue, seg)
                print(f" - {avenue}")
                print(f"   Adjusted score: {score:.2f} (segment mean: {base:.2f})")
                print(f"   Why: {explanation}\n")

        return top_recs


# --------------- CLI ---------------

def _get_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

def _get_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def _get_choice(prompt: str, choices):
    choices_lower = [c.lower() for c in choices]
    while True:
        value = input(prompt).strip().lower()
        if value in choices_lower:
            return choices[choices_lower.index(value)]
        print(f"Choose one of: {', '.join(choices)}")

def main():
    print("=== Investment Recommendation CLI ===\n")
    recommender = InvestmentRecommender.from_csv("Finance_Trends.csv")

    age = _get_int("Enter your age (e.g. 30): ")
    gender = input("Enter your gender (Male/Female): ").strip()
    duration_years = _get_float("Investment duration in years (e.g. 2.5): ")
    expected_return_pct = _get_float("Desired annual return % (e.g. 15): ")
    risk_level = _get_choice("Risk appetite (low/medium/high): ", ["low", "medium", "high"])
    top_n = _get_int("How many recommendations? (e.g. 3): ")

    print("\nGenerating recommendations...\n")
    recommender.recommend(
        age=age,
        gender=gender,
        duration_years=duration_years,
        expected_return_pct=expected_return_pct,
        risk_level=risk_level,
        top_n=top_n,
        verbose=True,
    )

if __name__ == "__main__":
    main()
