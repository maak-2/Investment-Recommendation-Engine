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
        layout="wide",
    )

    # Header
    st.title("📈 Investment Recommendation Engine")
    st.markdown(
        """
        This app uses a **survey-based financial behaviour dataset** to suggest
        suitable investment avenues based on your **risk appetite, time horizon, and return expectations**.
        """
    )

    # Load model
    recommender = load_recommender()

    # Layout: two columns
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🧑‍💼 Investor Profile")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=0,
        )

        duration_years = st.number_input(
            "Investment Horizon (years)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="How long you plan to keep this investment."
        )

        expected_return_pct = st.number_input(
            "Target Annual Return (%)",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Average annual return you are aiming for."
        )

        risk_level = st.selectbox(
            "Risk Appetite",
            options=["low", "medium", "high"],
            index=1,
            help="Higher risk usually means higher potential return, but also higher chance of loss."
        )

        top_n = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=7,
            value=3,
        )

        show_segment = st.checkbox(
            "Show underlying segment statistics",
            value=False,
        )

        run_button = st.button("🔍 Get Recommendations")

    with col_right:
        st.subheader("🎯 Results")

        if run_button:
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

            if not results:
                st.warning("No recommendations could be generated.")
            else:
                for avenue, score in results:
                    base = base_scores.get(avenue, float("nan"))
                    explanation = recommender._explain_recommendation(avenue, seg)

                    with st.container():
                        st.markdown(f"### 📌 {avenue}")
                        st.markdown(
                            f"- **Adjusted Score:** `{score:.2f}`  "
                            f"(Segment Mean: `{base:.2f}`)"
                        )
                        st.markdown(f"- **Why this option?** {explanation}")
                        st.markdown("---")

                if show_segment:
                    st.markdown("### 📊 Segment Statistics")
                    st.write(
                        "These are investors in the dataset with **similar duration** "
                        "and **expected return preferences** to you."
                    )
                    st.dataframe(seg.describe(include="all"))
        else:
            st.info("Fill in your profile on the left and click **'Get Recommendations'**.")


if __name__ == "__main__":
    main()
