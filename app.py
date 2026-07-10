"""
app.py
------
Streamlit app for the Credit Scoring Model, with Supabase for:
- Authentication (email/password signup, login, logout)
- Storing each user's prediction history (Postgres table via Supabase)

SETUP (one-time):
1. Create a free project at https://supabase.com
2. In the SQL editor, run:

    create table predictions (
        id uuid primary key default gen_random_uuid(),
        user_id uuid references auth.users not null,
        created_at timestamptz default now(),
        input_data jsonb not null,
        prob_default float not null,
        decision text not null
    );

    alter table predictions enable row level security;

    create policy "Users can insert their own predictions"
        on predictions for insert
        with check (auth.uid() = user_id);

    create policy "Users can view their own predictions"
        on predictions for select
        using (auth.uid() = user_id);

3. In your Supabase project settings -> API, copy the Project URL and the
   anon/public API key.
4. Create a .env file in the project root (add it to .gitignore -- do NOT
   commit it) with:

    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_KEY=your-anon-public-key

5. Add to requirements.txt: supabase, python-dotenv

Run:
    streamlit run app.py
"""

import json
import os
import math

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

from src.utils import MODELS_DIR, REPORTS_DIR, load_model
from src.features import engineer_features

load_dotenv()  # reads .env into os.environ, if present

st.set_page_config(page_title="Credit Scoring Model", page_icon="\U0001F4B3", layout="wide")


# ---------------------------------------------------------------------------
# Design system: fonts, palette, and layout overrides for Streamlit's
# default chrome. base = '#EEF2EF' (soft sage, not white/not dark), accent
# = muted emerald '#2F6F52'. Headings in a serif for an editorial, trustworthy
# feel; numbers in a monospace so data reads as data.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --bg:#d9e4dd;
    --panel:#cbd8cf;
    --card:#ffffff;
    --border:#aab7ae;
    --text:#111827;
    --text-soft:#374151;
}
.stApp{
    background:#d9e4dd !important;
}

label,
p,
span,
div,
h1,
h2,
h3{
    color:#111827 !important;
}

input{
    background:white !important;
    color:#111827 !important;
}
.stApp { background-color: var(--bg); }
html, body, [class*="css"]  { color: var(--text); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background-color: var(--accent);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 500;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { background-color: var(--accent-hover); color: #FFFFFF; }

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--card);
    border: 1px solid var(--border) !important;
    border-radius: 12px;
}

/* Sidebar-style left panel */
.left-panel {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    min-height: 520px;
    display: flex;
    flex-direction: column;
}
.left-panel-spacer { flex-grow: 1; }
.history-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 0.4rem 0;
    border-top: 1px solid var(--border);
    color: var(--text-soft);
}
.history-row .amt { color: var(--text); font-weight: 600; }
.stTabs{
    margin-top:-10px;
}

.stTabs [data-baseweb="tab-list"]{
    gap:0;
}

.stTabs [data-baseweb="tab"]{
    background:#4682B4;
    border:1px solid #D9E2DC;
    border-bottom:none;
    border-radius:10px 10px 0 0;
    padding:12px 28px;
    font-weight:600;
}

div[data-testid="stForm"]{
    background:white;
    border:1px solid #D9E2DC;
    border-top:none;
    border-radius:0 0 14px 14px;
    padding:25px;
}
/* Mono numbers */
.mono-number { font-family: 'IBM Plex Mono', monospace; }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
    if not url or not key:
        st.error(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to "
            "a .env file in the project root (see the setup instructions at the top of app.py)."
        )
        st.stop()
    return create_client(url, key)


@st.cache_resource
def get_model():
    try:
        return load_model("best_model.pkl")
    except FileNotFoundError:
        return None


@st.cache_data
def get_metrics_summary():
    path = REPORTS_DIR / "metrics_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def risk_band(prob_default: float) -> tuple[str, str]:
    """Map a predicted default probability to a decision recommendation + hex color."""
    if prob_default < 0.30:
        return "Approve", "var(--approve)"
    elif prob_default < 0.50:
        return "Manual Review", "var(--review)"
    else:
        return "Decline", "var(--decline)"


def render_gauge(prob_default: float, decision: str, color: str) -> str:
    """
    Build a semicircular risk gauge as inline SVG: three colored zones
    (approve / review / decline, matching risk_band's thresholds) with a
    needle pointing at the predicted probability.
    """
    cx, cy, r = 110, 110, 90
    zone_bounds = [0.0, 0.20, 0.50, 1.0]  # matches risk_band thresholds
    zone_colors = ["#2F396F", "#B8862F", "#A6432E"]

    def point(v):
        theta = math.radians(180 - 180 * v)
        return cx + r * math.cos(theta), cy - r * math.sin(theta)

    arcs = ""
    for i in range(3):
        x1, y1 = point(zone_bounds[i])
        x2, y2 = point(zone_bounds[i + 1])
        arcs += f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f}" fill="none" stroke="{zone_colors[i]}" stroke-width="14" stroke-linecap="round" opacity="0.85"/>'

    needle_x, needle_y = point(min(max(prob_default, 0.0), 1.0))
    needle_tip_x = cx + (needle_x - cx) * 0.8
    needle_tip_y = cy + (needle_y - cy) * 0.8

    color_solid = {"var(--approve)": "#2F6F52", "var(--review)": "#B8862F", "var(--decline)": "#A6432E"}.get(color, "#1E2A22")

    return f"""
    <div style="text-align:center;">
      <svg width="240" height="150" viewBox="0 0 220 130">
        {arcs}
        <line x1="{cx}" y1="{cy}" x2="{needle_tip_x:.1f}" y2="{needle_tip_y:.1f}" stroke="#1E2A22" stroke-width="3" stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="7" fill="#1E2A22"/>
      </svg>
      <div class="mono-number" style="font-size:2rem; font-weight:600; color:{color_solid}; margin-top:-0.5rem;">
        {prob_default:.1%}
      </div>
      <div style="font-family:'Source Serif 4',serif; font-size:1.15rem; font-weight:600; color:{color_solid};">
        {decision}
      </div>
    </div>
    """
# Auth
def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None


def sign_up(supabase: Client, email: str, password: str):
    try:
        supabase.auth.sign_up({"email": email, "password": password})
        st.success("Account created. Check your email to confirm, then log in below.")
    except Exception as e:
        st.error(f"Sign up failed: {e}")


def log_in(supabase: Client, email: str, password: str):
    try:
        result = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user = result.user
        st.session_state.access_token = result.session.access_token
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")


def log_out(supabase: Client):
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.session_state.access_token = None
    st.rerun()


def render_auth_screen(supabase: Client):
    inject_css()

    left, center, right = st.columns([1.2, 2, 1.2])

    with center:

        st.markdown(
            """
            <div style="
                background:#00BFFF;
                padding:35px;
                border-radius:16px;
                box-shadow:0 8px 25px rgba(0,0,0,.08);
                border:1px solid #D9E2DC;
            ">
            <h1 style="text-align:center;margin-bottom:5px;">
            💳 Credit Scoring
            </h1>

            <p style="
                text-align:center;
                color:#5B6B60;
                margin-bottom:25px;
            ">
            Log in or create an account to assess credit risk.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                    type="primary",
                )

                if submitted:
                    log_in(supabase, email, password)

        with signup_tab:
            with st.form("signup_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")

                submitted = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                    type="primary",
                )

                if submitted:
                    sign_up(supabase, email, password)


# Prediction history
def save_prediction(supabase: Client, user_id: str, input_data: dict, prob_default: float, decision: str):
    try:
        supabase.table("predictions").insert(
            {
                "user_id": user_id,
                "input_data": input_data,
                "prob_default": float(prob_default),
                "decision": decision,
            }
        ).execute()
    except Exception as e:
        st.warning(f"Prediction succeeded, but saving history failed: {e}")


def load_prediction_history(supabase: Client, user_id: str) -> pd.DataFrame:
    try:
        result = (
            supabase.table("predictions")
            .select("created_at, prob_default, decision")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return pd.DataFrame(result.data)
    except Exception as e:
        st.warning(f"Could not load prediction history: {e}")
        return pd.DataFrame()


# Main app (shown after login)
def render_app(supabase: Client):
    user = st.session_state.user

    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.title("\U0001F4B3 Credit Scoring Model")
        st.caption(f"Logged in as {user.email}")
    with top_col2:
        if st.button("Log Out", use_container_width=True):
            log_out(supabase)

    model = get_model()
    if model is None:
        st.error(
            "No trained model found at `models/best_model.pkl`. "
            "Run `python -m src.train` first to train and save a model."
        )
        st.stop()

    metrics_summary = get_metrics_summary()
    if metrics_summary:
        with st.expander("\U0001F4CA Model performance (held-out test set)"):
            best_model_name = metrics_summary["best_model"]
            st.write(f"**Active model:** `{best_model_name}`")
            comp_df = pd.DataFrame(metrics_summary["comparison"])
            st.dataframe(comp_df.style.format(precision=3), use_container_width=True)

    st.subheader("Applicant Details(prefer german credit dataset for features understanding)")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        credit_amount = st.number_input("Credit Amount", min_value=100, value=5000)
        month_duration = st.number_input("Duration (Months)", 1, 120, 24)
        n_credits = st.number_input("Existing Credits", 1, 10, 1)
        status_account = st.text_input("Status Account")
        status_savings = st.text_input("Savings Status")
        credit_history = st.text_input("Credit History")
        purpose = st.text_input("Purpose")
        housing = st.text_input("Housing")
        job = st.text_input("Job")
    with col2:
        years_employment = st.text_input("Years Employment")
        residence_since = st.number_input("Residence Since", 1, 10, 2)
        status_and_sex = st.text_input("Status And Gender")
        n_guarantors = st.number_input("Number of Guarantors", 0, 5, 0)
        secondary_obligor = st.text_input("Secondary Obligor")
        telephone = st.text_input("Telephone")
        other_installment_plans = st.text_input("Other Installment Plans")
        collateral = st.text_input("Collateral")
        payment_to_income_ratio = st.number_input(
        "Payment / Income Ratio",
        min_value=0.0,
        value=0.30,
        step=0.01,
    )
    is_foreign_worker = st.selectbox(
        "Foreign Worker",
        ["yes", "no"]
    )

    if st.button("Predict Credit Risk", type="primary", use_container_width=True):
        input_data = {
            "age": age,
            "credit_amount": credit_amount,
            "month_duration": month_duration,
            "n_credits": n_credits,
            "status_account": status_account,
            "status_savings": status_savings,
            "credit_history": credit_history,
            "purpose": purpose,
            "housing": housing,
            "job": job,
            "years_employment": years_employment,
            "residence_since": residence_since,
            "status_and_sex": status_and_sex,
            "n_guarantors": n_guarantors,
            "secondary_obligor": secondary_obligor,
            "telephone": telephone,
            "other_installment_plans": other_installment_plans,
            "collateral": collateral,
            "payment_to_income_ratio": payment_to_income_ratio,
            "is_foreign_worker": is_foreign_worker,
        }
        applicant = pd.DataFrame([input_data])
        applicant_features = engineer_features(applicant)
        prob_default = model.predict_proba(applicant_features)[0, 1]

        decision, color = risk_band(prob_default)

        st.divider()
        st.metric("Predicted Probability of Default", f"{prob_default:.1%}")
        st.markdown(f"### Recommendation: :{color}[{decision}]")
        st.progress(min(max(prob_default, 0.0), 1.0))

        save_prediction(supabase, user.id, input_data, prob_default, decision)

        with st.expander("Applicant feature snapshot"):
            st.dataframe(applicant_features.T.rename(columns={0: "value"}), use_container_width=True)

    st.divider()
    st.subheader("Your Recent Predictions")
    history_df = load_prediction_history(supabase, user.id)
    if history_df.empty:
        st.caption("No predictions yet.")
    else:
        st.dataframe(history_df, use_container_width=True)

# Entry point
def main():
    init_session_state()
    supabase = get_supabase_client()

    if st.session_state.user is None:
        render_auth_screen(supabase)
    else:
        render_app(supabase)


if __name__ == "__main__":
    main()