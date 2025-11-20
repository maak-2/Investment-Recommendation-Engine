# Investment Recommendation Engine Using Python (Hybrid ML + Rule-Based System)
# ============================================================
# IMPORT LIBRARIES & LOAD DATA
# ============================================================

# pandas: data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# ------------------------------------------------------------
# Load the dataset
# ------------------------------------------------------------
data_path = "Finance_Trends.csv"

# Reading the CSV file into a DataFrame
finance_df = pd.read_csv(data_path)

# Display the first few rows to understand the structure
finance_df.head()


# ============================================================
# 2. BASIC DATA INSPECTION & CLEANING
# ============================================================

# Inspect data types and missing values
finance_df.info()

# Check basic statistics for numeric columns 
finance_df.describe()

# View the column names for reference
finance_df.columns


# ------------------------------------------------------------
# Clean up whitespace in string columns
# ------------------------------------------------------------

text_columns = [
    'gender', 'Investment_Avenues', 'Duration', 'Invest_Monitor',
    'Expect', 'Avenue', 'What are your savings objectives?',
    'Reason_Equity', 'Reason_Mutual', 'Reason_Bonds',
    'Reason_FD', 'Source'
]

for col in text_columns:
    # Convert to string (in case there are missing or numeric types mixed in)
    finance_df[col] = finance_df[col].astype(str).str.strip()

# Check unique values after cleaning for key categorical columns
print("Unique values for Duration:")
print(finance_df['Duration'].value_counts(), '\n')

print("Unique values for Expect (Expected Return):")
print(finance_df['Expect'].value_counts(), '\n')

print("Unique values for gender:")
print(finance_df['gender'].value_counts(), '\n')

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) - OVERVIEW
# ============================================================

# List of numeric preference score columns for investment products.
investment_columns = [
    'Mutual_Funds',
    'Equity_Market',
    'Debentures',
    'Government_Bonds',
    'Fixed_Deposits',
    'PPF',
    'Gold'
]

# confirming these columns exist
finance_df[investment_columns].head()

# ------------------------------------------------------------
# 3.1 Average preference score per investment product
# ------------------------------------------------------------
# Visualization choice:
# - Bar chart is chosen because i want to compare the average score
#   of each product side-by-side.
# - This gives me a quick sense of which products are most liked overall.
average_scores = finance_df[investment_columns].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=average_scores.index, y=average_scores.values)
plt.title("Average Preference Score per Investment Product")
plt.ylabel("Average Score")
plt.xlabel("Investment Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3.2 Distribution of Risk Appetite (if available) and Duration
# ------------------------------------------------------------
# I use a countplot to see how many respondents fall into each duration
# category. This helps me understand typical investment horizons.

plt.figure(figsize=(8, 4))
sns.countplot(x='Duration', data=finance_df, order=finance_df['Duration'].value_counts().index)
plt.title("Distribution of Investment Duration Preferences")
plt.xlabel("Duration")
plt.ylabel("Count of Respondents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3.3 Expected Return vs. Popularity of Equity / Mutual Funds
# ------------------------------------------------------------
# Here i look at how expected return relates to average preference
# for high-risk products like Equity Market and Mutual Funds.

group_by_expect = finance_df.groupby('Expect')[['Equity_Market', 'Mutual_Funds']].mean()

plt.figure(figsize=(8, 4))
group_by_expect.plot(kind='bar')
plt.title("Average Preference for Equity & Mutual Funds by Expected Return")
plt.ylabel("Average Score")
plt.xlabel("Expected Return Category")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Note: i used a grouped bar chart because i want to compare
# the two products across the same expected-return categories.


# ============================================================
# 4. HELPER FUNCTIONS: MAPPING USER INPUTS TO DATA CATEGORIES
# ============================================================

# These functions convert continuous / raw user inputs into
# categorical labels consistent with the dataset, so that i can
# match users to similar investors.

def map_duration_years_to_category(years: float) -> str:
    """
    Map a numeric duration in years to the survey's Duration categories.

    Parameters
    ----------
    years : float
        Investment duration in years (e.g., 0.5, 2, 4, 7)

    Returns
    -------
    str
        One of:
        - "Less than 1 year"
        - "1-3 years"
        - "3-5 years"
        - "More than 5 years"
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
    Map a numeric expected return (percentage) to the survey's 'Expect' categories.

    Parameters
    ----------
    expected_return_pct : float
        Expected annual return in percentage (e.g., 12 for 12%, 25 for 25%).

    Returns
    -------
    str
        One of:
        - "10%-20%"
        - "20%-30%"
        - "30%-40%"
    """
    if expected_return_pct < 20:
        return "10%-20%"
    elif expected_return_pct < 30:
        return "20%-30%"
    else:
        return "30%-40%"
    
    
# ============================================================
# 5. HYBRID RECOMMENDATION ENGINE
# ============================================================

# 1. Data-driven approach:
#    - Filter the dataset to investors with similar Duration and Expected Return.
#    - Compute mean preference scores for each product in that segment.
# 2. Rule-based risk adjustment:
#    - If user risk is low: boost scores for safer products (PPF, FD, Bonds).
#    - If user risk is high: boost scores for riskier products (Equity, Mutual Funds).



def recommend_investments(
    age: int,
    gender: str,
    duration_years: float,
    expected_return_pct: float,
    risk_level: str = "medium",
    top_n: int = 3,
    verbose: bool = True
):
    """
    Recommend investment products based on user profile using a hybrid
    (data + rule-based) approach.

    Parameters
    ----------
    age : int
        Age of the investor (currently not heavily used, but could be extended).
    gender : str
        Gender of the investor ("Male", "Female", etc.). Not filtered on here,
        but kept to show extensibility.
    duration_years : float
        Investment horizon in years.
    expected_return_pct : float
        Desired annual return in percentage.
    risk_level : str, optional
        One of "low", "medium", "high". Adjusts product scores accordingly.
    top_n : int, optional
        Number of top recommendations to return. Default is 3.
    verbose : bool, optional
        If True, prints explanation to console.

    Returns
    -------
    list of tuples
        List of (product_name, adjusted_score) sorted by adjusted score descending.
    dict
        Dictionary of base (segment mean) scores for each product.
    dict
        Dictionary of adjusted scores for each product.

    Notes
    -----
    - The function first maps duration and expected return into the dataset's
      categorical labels.
    - It then filters the dataset to investors with matching Duration and Expect.
    - It calculates the mean preference score for each investment product in
      this segment.
    - Finally, it applies small boosts or penalties based on the user's risk level.
    """

    # --------------------------------------------------------
    # 5.1 Normalize inputs
    # --------------------------------------------------------
    gender_clean = gender.strip().title()   #"male" -> "Male"
    risk_level_clean = risk_level.strip().lower()

    # Map numeric inputs to dataset categories
    duration_category = map_duration_years_to_category(duration_years)
    expect_category = map_expected_return_to_category(expected_return_pct)

    if verbose:
        print("📌 Interpreted user profile:")
        print(f"   - Age: {age}")
        print(f"   - Gender: {gender_clean}")
        print(f"   - Duration (years): {duration_years} → '{duration_category}'")
        print(f"   - Expected Return (%): {expected_return_pct} → '{expect_category}'")
        print(f"   - Risk Level: {risk_level_clean}")
        print()

    # --------------------------------------------------------
    # 5.2 Segment the dataset by Duration & Expect
    # --------------------------------------------------------
    segment_df = finance_df[
        (finance_df['Duration'] == duration_category) &
        (finance_df['Expect'] == expect_category)
    ]

    # If no exact rows match this segment (unlikely but possible),
    # fall back to using the entire dataset so we still return something.
    if segment_df.empty:
        if verbose:
            print("⚠️ No exact segment match found. Using entire dataset instead.\n")
        segment_df = finance_df.copy()
    else:
        if verbose:
            print(f"Matched {len(segment_df)} similar investors for this profile.\n")

    # --------------------------------------------------------
    # 5.3 Compute mean preference score for each product
    # --------------------------------------------------------
    base_mean_scores = segment_df[investment_columns].mean().to_dict()

    if verbose:
        print("📊 Base mean scores (before risk adjustment):")
        for product, score in sorted(base_mean_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {product}: {score:.2f}")
        print()

    # --------------------------------------------------------
    # 5.4 Apply risk-based rule adjustments
    # --------------------------------------------------------
    # Risk boosts are small numeric bonuses added to the base mean score
    # to reflect the user's risk appetite.
    risk_boost_map = {
        "low": {
            'Fixed_Deposits': 1.0,
            'PPF': 1.0,
            'Government_Bonds': 0.7
        },
        "medium": {
            'Mutual_Funds': 0.5,
            'Government_Bonds': 0.5,
            'PPF': 0.5
        },
        "high": {
            'Equity_Market': 1.0,
            'Mutual_Funds': 0.7,
            'Gold': 0.5
        }
    }

    # Start with a copy of mean scores
    adjusted_scores = base_mean_scores.copy()

    # Apply boosts if risk level is recognized
    if risk_level_clean in risk_boost_map:
        boosts_for_level = risk_boost_map[risk_level_clean]
        for product, bonus in boosts_for_level.items():
            if product in adjusted_scores:
                adjusted_scores[product] = adjusted_scores[product] + bonus

    # --------------------------------------------------------
    # 5.5 Sort products by adjusted score & select top N
    # --------------------------------------------------------
    sorted_recommendations = sorted(
        adjusted_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_recommendations = sorted_recommendations[:top_n]

    if verbose:
        print("✅ Final Recommendations (after risk adjustment):")
        for product, adj_score in top_recommendations:
            base_score = base_mean_scores[product]
            print(f"   - {product}: adjusted {adj_score:.2f} (base {base_score:.2f})")
        print()

    return top_recommendations, base_mean_scores, adjusted_scores


# ============================================================
# 6. EXAMPLE USAGE OF THE RECOMMENDER
# ============================================================

# profile 1: Young, high-risk, long-term investor
top_recs_1, base_scores_1, adj_scores_1 = recommend_investments(
    age=28,
    gender="Male",
    duration_years=6,        # long-term horizon
    expected_return_pct=25,  # wants around 20%-30% returns
    risk_level="high",
    top_n=3,
    verbose=True
)

# profile 2: Older, low-risk, short-term investor
top_recs_2, base_scores_2, adj_scores_2 = recommend_investments(
    age=50,
    gender="Female",
    duration_years=0.5,      # short-term horizon
    expected_return_pct=12,  # wants around 10%-20%
    risk_level="low",
    top_n=3,
    verbose=True
)

# ============================================================
# 7. VISUAL EXPLANATION OF A RECOMMENDATION
# ============================================================

# Visualization goal is to:
# - Show the user how their recommended products compare to others.
# - Help stakeholders visually understand the effect of the model.

# I create a bar chart comparing base vs adjusted scores.


def plot_recommendation_explanation(
    base_scores: dict,
    adjusted_scores: dict,
    title: str = "Base vs Adjusted Scores for Investment Products"
):
    """
    Visualize how base (segment mean) scores differ from adjusted scores
    after applying risk-based rules.

    Parameters
    ----------
    base_scores : dict
        Product → base score (segment mean).
    adjusted_scores : dict
        Product → score after risk adjustment.
    title : str
        Title for the plot.

    Notes
    -----
    - We use a grouped bar chart because this is a classic way to compare
      two metrics (base vs adjusted) across a category (products).
    """
    products = list(base_scores.keys())
    base_values = [base_scores[p] for p in products]
    adjusted_values = [adjusted_scores[p] for p in products]

    x_positions = np.arange(len(products))  # positions along the x-axis
    bar_width = 0.35                        # width of each bar

    plt.figure(figsize=(10, 5))

    # Bar for base scores
    plt.bar(x_positions - bar_width/2, base_values, width=bar_width, label='Base (Segment Mean)')

    # Bar for adjusted scores
    plt.bar(x_positions + bar_width/2, adjusted_values, width=bar_width, label='Adjusted (Risk-Weighted)')

    plt.xticks(x_positions, products, rotation=45)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# explain the high-risk user's recommendation visually
plot_recommendation_explanation(base_scores_1, adj_scores_1,
                                title="High-Risk Investor: Base vs Adjusted Scores")

# explain the low-risk user's recommendation visually
plot_recommendation_explanation(base_scores_2, adj_scores_2,
                                title="Low-Risk Investor: Base vs Adjusted Scores")
plt.figure(figsize=(8,6))
corr = finance_df[investment_columns].corr()
sns.heatmap(corr, annot=False)
plt.title("Correlation Between Investment Preference Scores")
plt.tight_layout()
plt.show()

finance_df['age_band'] = pd.cut(
    finance_df['age'],
    bins=[18, 25, 35, 50, 70],
    labels=['18-25', '26-35', '36-50', '51-70']
)

age_pref = finance_df.groupby('age_band')[investment_columns].mean()

age_pref.T.plot(kind='bar', figsize=(10,6))
plt.title("Average Investment Preference by Age Band")
plt.ylabel("Average Score")
plt.xlabel("Investment Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# INVESTMENT RECOMMENDER CLASS 
# ============================================================

class InvestmentRecommender:
    def __init__(self, dataframe):
        """
        Initialize with a cleaned Pandas DataFrame.
        """
        self.df = dataframe
        self.investment_cols = [
            'Mutual_Funds', 'Equity_Market', 'Debentures', 'Government_Bonds',
            'Fixed_Deposits', 'PPF', 'Gold'
        ]

    @classmethod
    def from_csv(cls, file_path: str):
        """
        Class method to load CSV and automatically clean text fields.
        """
        df = pd.read_csv(file_path)

        # Clean up whitespace for important columns
        text_cols = [
            'gender', 'Investment_Avenues', 'Duration', 'Expect', 'Avenue',
            'Reason_Equity', 'Reason_Mutual', 'Reason_Bonds', 'Reason_FD', 'Source'
        ]

        for col in text_cols:
            if col in df.columns:
                # Convert to string and strip leading/trailing spaces
                df[col] = df[col].astype(str).str.strip()

        return cls(df)


    # -----------------------------------------------
    # Mapping functions 
    # -----------------------------------------------

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

    # -----------------------------------------------
    # Main recommendation method
    # -----------------------------------------------

    def recommend(
        self, age, gender, duration_years, expected_return_pct,
        risk_level="medium", top_n=3, verbose=True
    ):
        """
        Hybrid recommendation engine (same logic as notebook).
        """

        duration_cat = self.map_duration_years(duration_years)
        expect_cat = self.map_expected_return(expected_return_pct)

        # Filter dataset for similar investors
        seg = self.df[
            (self.df["Duration"] == duration_cat) &
            (self.df["Expect"] == expect_cat)
        ]

        if seg.empty:
            seg = self.df.copy()

        # Compute mean scores
        base_scores = seg[self.investment_cols].mean().to_dict()

        # Risk-based adjustments
        risk_boosts = {
            "low": {"Fixed_Deposits": 1, "PPF": 1, "Government_Bonds": 0.7},
            "medium": {"Mutual_Funds": 0.5, "PPF": 0.5, "Government_Bonds": 0.5},
            "high": {"Equity_Market": 1, "Mutual_Funds": 0.7, "Gold": 0.5},
        }

        adjusted_scores = base_scores.copy()
        if risk_level in risk_boosts:
            for product, bonus in risk_boosts[risk_level].items():
                adjusted_scores[product] += bonus

        # Sort by adjusted score
        final_recs = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if verbose:
            print("=== Recommendation Result ===\n")
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
            print("Please enter a valid integer.\n")


def _get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.\n")


def _get_choice(prompt, valid_choices):
    valid_choices = [v.lower() for v in valid_choices]
    while True:
        value = input(prompt).strip().lower()
        if value in valid_choices:
            return value
        else:
            print(f"Invalid choice. Please choose from: {valid_choices}\n")
            

# ============================================================
# CLI RUNNER
# ============================================================

def main():
    print("\n=== Investment Recommendation CLI ===\n")

    # Load class-based recommender
    recommender = InvestmentRecommender.from_csv("Finance_Trends.csv")

    # Collecting user input
    age = _get_int("Enter your age (e.g. 30): ")
    gender = input("Enter your gender (e.g. Male/Female): ").strip()

    duration_years = _get_float(
        "For how many years do you plan to invest? (e.g. 2.5): "
    )

    expected_return_pct = _get_float(
        "What average annual return (%) are you aiming for? (e.g. 15): "
    )

    risk_level = _get_choice(
        "What is your risk appetite? (low / medium / high): ",
        ["low", "medium", "high"]
    )

    top_n = _get_int(
        "How many top investment avenues would you like to see? (e.g. 3): "
    )

    print("\nCalculating recommendations...\n")

    # Get recommendations
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

