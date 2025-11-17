import streamlit as st
from investment_recommender import (
    InvestmentRecommender,
    map_duration_years_to_category,
    map_expected_return_to_category,
)


# ---------- Load Recommender ----------
@st.cache_data
def load_recommender():
    return InvestmentRecommender.from_csv("Finance_Trends.csv")


# ---------- Streamlit App ----------
def main():
    st.set_page_config(
        page_title="Investment Recommendation Engine",
        page_icon="📈",
        layout="centered",
    )

    st.title("📈 Investment Recommendation Engine")
    st.write(
        "Provide your investment preferences and receive personalised, "
        "data-backed investment recommendations."
    )

    recommender = load_recommender()

    # Sidebar user input
    with st.sidebar:
        st.header("Investor Profile")

        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)

        duration_years = st.number_input(
            "Investment Horizon (years)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )

        expected_return_pct = st.number_input(
            "Target Annual Return (%)",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
        )

        risk_level = st.selectbox(
            "Risk Appetite",
            ["low", "medium", "high"],
            index=1,
        )

        top_n = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=7,
            value=3,
        )

        show_segment = st.checkbox("Show Segment Stats", value=False)

    # Run recommendation
    if st.button("Get Recommendations"):
        st.subheader("🎯 Recommended Investment Options")

        # Generate recommendations (silent mode)
        results = recommender.recommend(
            age=age,
            gender=gender,
            duration_years=duration_years,
            expected_return_pct=expected_return_pct,
            risk_level=risk_level,
            top_n=top_n,
            verbose=False,
        )

        # Segment & base scores
        duration_cat = map_duration_years_to_category(duration_years)
        expect_cat = map_expected_return_to_category(expected_return_pct)
        seg = recommender._get_segment(duration_cat, expect_cat)
        base_scores = recommender._compute_segment_scores(seg)

        for avenue, score in results:
            base = base_scores.get(avenue, float("nan"))
            explanation = recommender._explain_recommendation(avenue, seg)

            with st.container():
                st.markdown(f"### 📌 {avenue}")
                st.markdown(
                    f"**Adjusted Score:** `{score:.2f}`  "
                    f"(Segment Mean: `{base:.2f}`)"
                )
                st.markdown(f"**Why this option?** {explanation}")
                st.markdown("---")

        if show_segment:
            st.subheader("📊 Segment Statistics")
            st.write(
                "These are investors in the dataset with similar duration "
                "and expected return preferences."
            )
            st.dataframe(seg.describe(include='all'))


if __name__ == "__main__":
    main()
