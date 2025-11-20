import pandas as pd

# ============================================================
# INVESTMENT RECOMMENDER CLASS
# ============================================================

class InvestmentRecommender:
    def __init__(self, dataframe):
        """
        Initialize the recommender with a cleaned pandas DataFrame.
        """
        self.df = dataframe
        self.investment_cols = [
            "Mutual_Funds", "Equity_Market", "Debentures", "Government_Bonds",
            "Fixed_Deposits", "PPF", "Gold"
        ]

    @classmethod
    def from_csv(cls, file_path: str):
        """
        Load dataset from CSV and perform necessary cleaning.
        """
        df = pd.read_csv(file_path)

        # Clean up whitespace from important string columns
        text_cols = [
            "gender", "Investment_Avenues", "Duration", "Expect", "Avenue",
            "Reason_Equity", "Reason_Mutual", "Reason_Bonds", "Reason_FD", "Source"
        ]

        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return cls(df)

    # -----------------------------------------------------------
    # Mapping functions
    # -----------------------------------------------------------

    @staticmethod
    def map_duration_years(years: float) -> str:
        if years < 1:
            return "Less than 1 year"
        elif years < 3:
            return "1-3 years"
        elif years < 5:
            return "3-5 years"
        else:
            return "More than 5 years"

    @staticmethod
    def map_expected_return(expected_pct: float) -> str:
        if expected_pct < 20:
            return "10%-20%"
        elif expected_pct < 30:
            return "20%-30%"
        else:
            return "30%-40%"

    # -----------------------------------------------------------
    # Main Recommendation Method
    # -----------------------------------------------------------

    def recommend(
        self,
        age,
        gender,
        duration_years,
        expected_return_pct,
        risk_level="medium",
        top_n=3,
        verbose=True
    ):
        """
        Hybrid recommendation engine combining:
        - Data-driven segmentation
        - Rule-based risk adjustment
        """

        duration_cat = self.map_duration_years(duration_years)
        expect_cat = self.map_expected_return(expected_return_pct)
        gender = gender.strip().title()
        risk_level = risk_level.lower()

        # Filter similar investors
        seg = self.df[
            (self.df["Duration"] == duration_cat) &
            (self.df["Expect"] == expect_cat)
        ]

        if seg.empty:
            seg = self.df.copy()

        # Base mean preference scores
        base_scores = seg[self.investment_cols].mean().to_dict()

        # Risk adjustment rules
        risk_boosts = {
            "low": {"Fixed_Deposits": 1, "PPF": 1, "Government_Bonds": 0.7},
            "medium": {"Mutual_Funds": 0.5, "PPF": 0.5, "Government_Bonds": 0.5},
            "high": {"Equity_Market": 1, "Mutual_Funds": 0.7, "Gold": 0.5}
        }

        # Apply adjustments
        adjusted_scores = base_scores.copy()
        if risk_level in risk_boosts:
            for prod, boost in risk_boosts[risk_level].items():
                adjusted_scores[prod] = adjusted_scores.get(prod, 0) + boost

        # Sort and return top recommendations
        final_recs = sorted(
            adjusted_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        if verbose:
            print("=== Recommendation Results ===\n")
            for prod, score in final_recs:
                print(f"• {prod}: {score:.2f} (base: {base_scores[prod]:.2f})")
            print()

        return final_recs


# ============================================================
# HELPER INPUT FUNCTIONS FOR CLI
# ============================================================

def _get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Invalid integer. Try again.\n")

def _get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Invalid number. Try again.\n")

def _get_choice(prompt, valid_choices):
    valid_choices = [v.lower() for v in valid_choices]
    while True:
        response = input(prompt).strip().lower()
        if response in valid_choices:
            return response
        else:
            print(f"❌ Please choose from: {valid_choices}\n")


# ============================================================
# CLI MAIN EXECUTION
# ============================================================

def main():
    print("\n=== Investment Recommendation CLI ===\n")

    # Load the recommendation engine
    recommender = InvestmentRecommender.from_csv("Finance_Trends.csv")

    # Collect user input
    age = _get_int("Enter your age (e.g. 30): ")
    gender = input("Enter your gender (Male/Female): ").strip()
    duration_years = _get_float("How many years do you plan to invest? (e.g. 3.5): ")
    expected_return_pct = _get_float("Expected annual return (%), e.g. 15: ")
    risk_level = _get_choice("Risk appetite (low/medium/high): ", ["low", "medium", "high"])
    top_n = _get_int("How many top recommendations? (e.g. 3): ")

    print("\nGenerating your personalized investment recommendations...\n")

    # Generate recommendations
    recommender.recommend(
        age=age,
        gender=gender,
        duration_years=duration_years,
        expected_return_pct=expected_return_pct,
        risk_level=risk_level,
        top_n=top_n,
        verbose=True
    )


if __name__ == "__main__":
    main()
