import streamlit as st
from datetime import date, timedelta
from src.data_loader import load_prices

st.set_page_config(page_title="SignalScope (Minimal)")
st.title("📈 SignalScope — Minimal App")

ticker = st.sidebar.text_input("Ticker", "URA")
years = st.sidebar.slider("Years of history", 1, 10, 5)
run = st.sidebar.button("Run / Refresh", type="primary")

def get_data():
    end = date.today()
    start = end - timedelta(days=365*years)
    return load_prices(ticker, start.isoformat(), end.isoformat())

if run or "df" not in st.session_state:
    st.session_state["df"] = get_data()

df = st.session_state.get("df")
if df is None or df.empty:
    st.warning("No data returned. Try a different ticker or shorter range.")
else:
    st.line_chart(df.set_index("Date")["Close"])
    st.write(df.tail())

