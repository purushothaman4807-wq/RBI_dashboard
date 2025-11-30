
# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="RBI Macro Dashboard", layout="wide", page_icon="🏦")
st.markdown("<h1 style='text-align:center;color:#0B63A8;'>RBI Macro Economic Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Inflation • Riskometer • Monetary Policy • Liquidity (India + US) — forecasting & exports (text report)")

# -------------------- SIDEBAR NAVIGATION --------------------
menu = st.sidebar.radio("Navigation", [
    "Inflation (India + US)",
    "Riskometer",
    "Monetary Policy Impact",
    "Liquidity Data (India + US)",
    "Correlation & Forecasts",
    "Brief Report"
])

# -------------------- CONFIG / KEYS --------------------
FRED_API_KEY = st.secrets.get("fred_api_key")  # optional: set this in Streamlit secrets to enable FRED

# -------------------- HELPERS --------------------
def get_fred(series_id):
    """Fetch data from FRED API safely (monthly/daily depending on series)."""
    if not FRED_API_KEY:
        # return empty DF if key missing (app remains fully functional)
        return pd.DataFrame(columns=["date", "value"])
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        observations = r.json().get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["date", "value"])
        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).reset_index(drop=True)
        return df[["date", "value"]]
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

def india_cpi():
    """Fetch India CPI (World Bank series FP.CPI.TOTL) by year. Returns annual data."""
    try:
        url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL?format=json&per_page=500"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if not payload or len(payload) < 2:
            return pd.DataFrame(columns=["date", "value"])
        data = payload[1]
        df = pd.DataFrame(data)
        df = df.rename(columns={"date": "year", "value": "value"})
        df["date"] = pd.to_datetime(df["year"], format="%Y", errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        df = df[["date", "value"]].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

def fetch_usdinr():
    """Fetch USD -> INR latest exchange rate using exchangerate.host (free)."""
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR", timeout=10)
        r.raise_for_status()
        rate = r.json().get("rates", {}).get("INR", None)
        return rate
    except Exception:
        return None

def linear_forecast(df, periods=12, freq='M'):
    """
    Simple linear forecast using numpy.polyfit.
    Expects df with ['date','value'] sorted asc.
    Returns combined DataFrame with 'is_forecast' boolean.
    """
    if df.empty or len(df) < 3:
        out = df.copy()
        out["is_forecast"] = False
        return out
    df = df.sort_values("date").reset_index(drop=True)
    x = np.array([d.toordinal() for d in df["date"]])
    y = df["value"].values.astype(float)
    p = np.polyfit(x, y, deg=1)
    slope, intercept = p[0], p[1]
    last = df["date"].iloc[-1]
    future_dates = []
    if freq == 'M':
        for i in range(1, periods + 1):
            future_dates.append((last + pd.DateOffset(months=i)).to_pydatetime())
    else:
        for i in range(1, periods + 1):
            future_dates.append((last + timedelta(days=30 * i)).to_pydatetime())
    x_future = np.array([d.toordinal() for d in future_dates])
    y_future = intercept + slope * x_future
    fut_df = pd.DataFrame({"date": future_dates, "value": y_future})
    fut_df["is_forecast"] = True
    hist_df = df.copy()
    hist_df["is_forecast"] = False
    out = pd.concat([hist_df, fut_df], ignore_index=True)
    return out

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -------------------- PAGES --------------------

# Page 1: Inflation
if menu == "Inflation (India + US)":
    st.header("📌 Inflation Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇺🇸 US CPI (Live FRED)")
        us_df = get_fred("CPIAUCSL")
        if not us_df.empty:
            st.line_chart(us_df.set_index("date"))
            st.metric("Latest US CPI", f"{us_df['value'].iloc[-1]:.2f}")
            us_proj = linear_forecast(us_df, periods=12, freq='M')
            fig = px.line(us_proj, x="date", y="value", color="is_forecast", labels={"value": "CPI", "is_forecast": "Forecast"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("US CPI not available (FRED key missing or API error).")

    with col2:
        st.subheader("🇮🇳 India CPI (World Bank)")
        ind_df = india_cpi()
        if not ind_df.empty:
            st.line_chart(ind_df.set_index("date"))
            st.metric("Latest India CPI (annual)", f"{ind_df['value'].iloc[-1]:.2f}")
            ind_proj = linear_forecast(ind_df, periods=5, freq='M')
            fig2 = px.line(ind_proj, x="date", y="value", color="is_forecast", labels={"value": "CPI", "is_forecast": "Forecast"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("India CPI not available (World Bank API issue).")

    st.subheader("🧮 Inflation Calculator")
    initial = st.number_input("Initial Price", value=100.0)
    inflation = st.number_input("Inflation Rate (%)", value=6.0)
    years = st.number_input("Years", value=5, min_value=1)
    future_price = initial * ((1 + inflation / 100) ** years)
    st.success(f"Future Price after {years} years → ₹{future_price:.2f}")

# Page 2: Riskometer
if menu == "Riskometer":
    st.header("📌 Portfolio Riskometer")
    st.write("Enter your allocations. If they don't sum to 100%, you can normalize them to 100% automatically.")

    col1, col2, col3 = st.columns(3)
    with col1:
        eq = st.number_input("Equity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, format="%.1f")
    with col2:
        debt = st.number_input("Debt (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, format="%.1f")
    with col3:
        gold = st.number_input("Gold/Commodities (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, format="%.1f")

    total = eq + debt + gold
    st.write(f"**Total allocation:** {total:.1f}%")

    # If total is zero, avoid division by zero
    if total == 0:
        st.warning("All allocations are zero. Please enter allocations or use the quick presets below.")
        if st.button("Use balanced preset (50E/30D/20G)"):
            st.experimental_rerun()  # user will need to re-enter; this keeps UI simple
    else:
        normalize = False
        if abs(total - 100.0) > 1e-6:
            st.warning("Allocation does not sum to 100%. You can normalize (scale) allocations so they sum to 100%.")
            if st.button("Normalize allocations to 100%"):
                normalize = True

        if normalize:
            eq_n = (eq / total) * 100.0
            debt_n = (debt / total) * 100.0
            gold_n = (gold / total) * 100.0
            st.success(f"Allocations normalized → Equity: {eq_n:.1f}%, Debt: {debt_n:.1f}%, Gold: {gold_n:.1f}%")
            eq, debt, gold = eq_n, debt_n, gold_n
            total = eq + debt + gold

        # compute risk score using (possibly normalized) allocations
        # risk_score = equity*0.7 + gold*0.2 + debt*0.1  (scale to 0-100)
        risk_score = eq * 0.7 + gold * 0.2 + debt * 0.1

        # show gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Portfolio Risk Score (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "red"}
                ],
                "bar": {"color": "darkblue"},
                "threshold": {"value": risk_score, "line": {"color": "red", "width": 4}}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # message
        if risk_score < 30:
            st.success(f"Score: {risk_score:.1f} → LOW RISK")
        elif risk_score < 60:
            st.warning(f"Score: {risk_score:.1f} → MODERATE RISK")
        else:
            st.error(f"Score: {risk_score:.1f} → HIGH RISK")

        st.subheader("Allocation breakdown")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Equity", f"{eq:.1f}%")
        col_b.metric("Debt", f"{debt:.1f}%")
        col_c.metric("Gold", f"{gold:.1f}%")

# Page 3: Monetary Policy Impact
if menu == "Monetary Policy Impact":
    st.header("📌 Monetary Policy Risk Analysis")
    st.subheader("📈 Risks when Interest Rates Increase")
    st.write("- Borrowing cost rises\n- GDP growth slows\n- Bond prices fall\n- Stock market correction\n- EM currency depreciation")
    st.subheader("💧 Risks when Liquidity Increases")
    st.write("- Inflation rises\n- Asset bubble risk\n- Currency weakens\n- Excessive credit growth")
    st.subheader("🔥 Risks when Inflation Rises")
    st.write("- Purchasing power falls\n- Corporate margins shrink\n- Monetary tightening expected")
    st.subheader("🇺🇸 Risks when US CPI Rises")
    st.write("- USD strengthens\n- FPI outflows from India\n- RBI may be forced to hike")

# Page 4: Liquidity
if menu == "Liquidity Data (India + US)":
    st.header("📌 Liquidity (India + US)")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🇺🇸 US Fed Balance Sheet (WALCL via FRED)")
        fed_df = get_fred("WALCL")
        if not fed_df.empty:
            st.line_chart(fed_df.set_index("date"))
            latest_val = fed_df["value"].iloc[-1] / 1e6  # convert to millions
            st.metric("Latest Fed Balance Sheet (Trillion $)", f"${latest_val/1000:.2f}T")
            fed_proj = linear_forecast(fed_df, periods=12, freq='M')
            figf = px.line(fed_proj, x="date", y="value", color="is_forecast", labels={"value": "WALCL", "is_forecast": "Forecast"})
            st.plotly_chart(figf, use_container_width=True)
        else:
            st.info("Fed balance sheet data not available (FRED key missing or API issue).")

    with col4:
        st.subheader("🇮🇳 India Liquidity — Upload CSV")
        st.info("CSV must include `date` and `value` columns. Date can be daily/monthly/year (ISO). Example columns: date,value")
        file = st.file_uploader("Upload CSV with 'date' & 'value' columns", type=['csv'])
        uploaded_df = pd.DataFrame()
        if file:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if 'date' not in df.columns or 'value' not in df.columns:
                    st.error("CSV must contain 'date' and 'value' columns (case-insensitive).")
                else:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['date', 'value']).sort_values('date').reset_index(drop=True)
                    if df.empty:
                        st.error("After parsing, no valid date/value rows found.")
                    else:
                        uploaded_df = df.copy()
                        st.line_chart(df.set_index("date"))
                        st.metric("Latest India Liquidity", f"₹{df['value'].iloc[-1]:,.2f}")
                        st.download_button("Download uploaded CSV", data=df_to_csv_bytes(df), file_name="india_liquidity_uploaded.csv", mime="text/csv")
                        proj = linear_forecast(df, periods=12, freq='M')
                        figp = px.line(proj, x="date", y="value", color="is_forecast", labels={"value": "Liquidity", "is_forecast": "Forecast"})
                        st.plotly_chart(figp, use_container_width=True)
            except Exception as e:
                st.error(f"CSV parsing error: {e}")

    # show USD/INR
    rate = fetch_usdinr()
    if rate:
        st.sidebar.metric("USD → INR", f"{rate:.2f}")

# Page 5: Correlation & Forecasts
if menu == "Correlation & Forecasts":
    st.header("📌 Correlations & Simple Forecasts")
    st.write("Upload one or more CSV files. Each should contain `date` and `value` columns. We'll merge by date and show correlations.")
    uploaded = st.file_uploader("Upload multiple CSVs (hold Ctrl/Cmd to select more)", accept_multiple_files=True, type=['csv'])
    dfs = {}
    if uploaded:
        for f in uploaded:
            try:
                name = f.name.rsplit('.', 1)[0]
                d = pd.read_csv(f)
                d.columns = [c.strip().lower() for c in d.columns]
                if 'date' not in d.columns or 'value' not in d.columns:
                    st.warning(f"Skipping {f.name}: needs 'date' & 'value' columns.")
                    continue
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d['value'] = pd.to_numeric(d['value'], errors='coerce')
                d = d.dropna(subset=['date', 'value'])
                d = d[['date', 'value']].rename(columns={'value': name})
                dfs[name] = d
            except Exception as e:
                st.warning(f"Couldn't parse {f.name}: {e}")
        if dfs:
            merged = None
            for name, d in dfs.items():
                if merged is None:
                    merged = d.copy()
                else:
                    merged = pd.merge(merged, d, on='date', how='outer')
            merged = merged.sort_values('date').set_index('date').interpolate().dropna(axis=0, how='all')
            st.write("Merged preview (interpolated):")
            st.dataframe(merged.tail(10))
            if merged.shape[1] > 1:
                corr = merged.corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Heatmap')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload at least 2 series to compute correlation.")
            forecast_horizon = st.slider("Forecast horizon (months)", 1, 24, 12)
            combined = pd.DataFrame(index=merged.index)
            for col in merged.columns:
                s = merged[[col]].dropna().reset_index().rename(columns={'date': 'date', col: 'value'})
                if not s.empty:
                    pf = linear_forecast(s, periods=forecast_horizon)
                    pf = pf.set_index('date')['value'].rename(col)
                    combined = combined.join(pf, how='outer')
            st.line_chart(combined)
            st.download_button("Download merged dataset (CSV)", data=df_to_csv_bytes(merged.reset_index()), file_name="merged_timeseries.csv")

# Page 6: Brief Report (text only)
if menu == "Brief Report":
    st.header("📋 Brief Text Report — Dashboard Summary")
    st.write("This is a concise, text-only summary of the dashboard's key points and the riskometer result.")

    # gather a few latest values where available
    us_df = get_fred("CPIAUCSL")
    fed_df = get_fred("WALCL")
    ind_df = india_cpi()

    lines = []
    lines.append("RBI Macro Dashboard — Brief Summary")
    lines.append(f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # US CPI
    if not us_df.empty:
        lines.append(f"- Latest US CPI (CPIAUCSL): {us_df['value'].iloc[-1]:.2f} (as of {us_df['date'].iloc[-1].date()})")
    else:
        lines.append("- Latest US CPI: not available (FRED key missing or API error)")

    # India CPI
    if not ind_df.empty:
        lines.append(f"- Latest India CPI (World Bank annual): {ind_df['value'].iloc[-1]:.2f} (year {ind_df['date'].iloc[-1].year})")
    else:
        lines.append("- Latest India CPI: not available (World Bank API issue)")

    # Fed balance sheet
    if not fed_df.empty:
        fed_latest = fed_df['value'].iloc[-1]
        lines.append(f"- Latest Fed Balance Sheet (WALCL): {fed_latest:,.0f}")
    else:
        lines.append("- Fed Balance Sheet: not available (FRED key missing or API error)")

    # USD/INR
    usd_inr = fetch_usdinr()
    if usd_inr:
        lines.append(f"- USD → INR (spot): {usd_inr:.2f}")
    else:
        lines.append("- USD → INR: not available (exchange API error)")

    # simple interpretation heuristics
    lines.append("")
    lines.append("Interpretation / Notes:")
    if not ind_df.empty:
        try:
            latest_ind = ind_df['value'].iloc[-1]
            if latest_ind > 6.0:
                lines.append("- Inflation: India CPI is elevated (>6%). RBI likely to remain vigilant on tightening.")
            else:
                lines.append("- Inflation: India CPI is within moderate range (≤6%).")
        except Exception:
            lines.append("- Inflation: unable to compute interpretation.")
    else:
        lines.append("- Inflation: India CPI data not available for interpretation.")

    lines.append("- Liquidity: check the Liquidity Data page for uploaded series and recent trend. Surplus liquidity typically eases short-term rates; tight liquidity raises them.")
    lines.append("- Riskometer: check the Riskometer page. You can normalize allocations to 100% if needed.")

    # show report text
    report_text = "\n".join(lines)
    st.text_area("Brief Report (text only)", value=report_text, height=360)

# -------------------- END --------------------
