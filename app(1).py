import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Stock Volatility Signal Lab",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Download daily adjusted OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(
            f"No data was returned for {ticker}. Check the ticker and date range."
        )

    # yfinance can return MultiIndex columns even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns from downloaded data: {sorted(missing)}")

    data = data.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"
    return data


def build_strategy(
    prices: pd.DataFrame,
    volatility_window: int,
    short_ma: int,
    long_ma: int,
    low_quantile: float,
    high_quantile: float,
    use_volatility_filter: bool,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, float, float]:
    """Create volatility regimes, trading signals, and a simple backtest."""
    df = prices.copy()

    df["Return"] = df["Close"].pct_change()
    df["SMA_Short"] = df["Close"].rolling(short_ma).mean()
    df["SMA_Long"] = df["Close"].rolling(long_ma).mean()

    # Annualized rolling volatility.
    df["Volatility"] = (
        df["Return"].rolling(volatility_window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    valid_vol = df["Volatility"].dropna()
    if valid_vol.empty:
        raise ValueError(
            "Not enough observations to calculate volatility. "
            "Choose an earlier start date or a shorter volatility window."
        )

    low_threshold = float(valid_vol.quantile(low_quantile))
    high_threshold = float(valid_vol.quantile(high_quantile))

    df["Volatility_Regime"] = np.select(
        [
            df["Volatility"] < low_threshold,
            df["Volatility"] < high_threshold,
        ],
        ["Low", "Medium"],
        default="High",
    )
    df.loc[df["Volatility"].isna(), "Volatility_Regime"] = np.nan

    df["Cross_Above"] = (
        (df["Close"] > df["SMA_Short"])
        & (df["Close"].shift(1) <= df["SMA_Short"].shift(1))
    )
    df["Cross_Below"] = (
        (df["Close"] < df["SMA_Short"])
        & (df["Close"].shift(1) >= df["SMA_Short"].shift(1))
    )

    bull_trend = df["SMA_Short"] > df["SMA_Long"]
    bear_trend = df["SMA_Short"] < df["SMA_Long"]

    buy_condition = df["Cross_Above"] & bull_trend
    short_condition = df["Cross_Below"] & bear_trend

    if use_volatility_filter:
        # Avoid buying during the highest-volatility regime.
        buy_condition &= df["Volatility"] <= high_threshold
        # Require at least medium volatility for a short entry.
        short_condition &= df["Volatility"] >= low_threshold

    df["Entry_Signal"] = 0
    df.loc[buy_condition, "Entry_Signal"] = 1
    df.loc[short_condition, "Entry_Signal"] = -1

    # Hold the latest direction until the opposite signal appears.
    df["Target_Position"] = df["Entry_Signal"].replace(0, np.nan).ffill().fillna(0)

    # Avoid look-ahead bias: today's signal becomes tomorrow's position.
    df["Position"] = df["Target_Position"].shift(1).fillna(0)

    turnover = df["Position"].diff().abs().fillna(df["Position"].abs())
    cost_rate = transaction_cost_bps / 10_000
    df["Strategy_Return"] = df["Position"] * df["Return"] - turnover * cost_rate
    df["Buy_Hold_Return"] = df["Return"]

    df["Strategy_Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    df["Buy_Hold_Equity"] = (1 + df["Buy_Hold_Return"].fillna(0)).cumprod()

    df["Signal_Label"] = np.select(
        [df["Entry_Signal"] == 1, df["Entry_Signal"] == -1],
        ["BUY", "SHORT"],
        default="HOLD",
    )

    return df, low_threshold, high_threshold


def annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    growth = (1 + clean).prod()
    years = len(clean) / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return np.nan
    return growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return clean.std(ddof=1) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    vol = annualized_volatility(clean)
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    return (annualized_return(clean) - risk_free_rate) / vol


def fmt_pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def fmt_num(value: float, decimals: int = 2) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:,.{decimals}f}"



def calculate_signal_statistics(df: pd.DataFrame, forward_days: int):
    analysis = df.copy()
    analysis["Forward_Return"] = (
        analysis["Close"].shift(-forward_days) / analysis["Close"] - 1
    )

    buy = analysis[
        (analysis["Entry_Signal"] == 1) & analysis["Forward_Return"].notna()
    ].copy()
    short = analysis[
        (analysis["Entry_Signal"] == -1) & analysis["Forward_Return"].notna()
    ].copy()

    def summarize(returns):
        if len(returns) == 0:
            return {
                "count": 0,
                "win_rate": np.nan,
                "average_return": np.nan,
                "best_return": np.nan,
                "worst_return": np.nan,
            }
        return {
            "count": len(returns),
            "win_rate": float((returns > 0).mean()),
            "average_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "worst_return": float(returns.min()),
        }

    return {
        "BUY": summarize(buy["Forward_Return"]),
        "SHORT": summarize(-short["Forward_Return"]),
    }


st.title("Stock Volatility Signal Lab")
st.caption(
    "Interactive research dashboard for volatility regimes, BUY/SHORT signals, "
    "and a simple historical backtest."
)

with st.sidebar:
    st.header("Settings")

    ticker = st.text_input(
        "Company ticker",
        value="AAPL",
        help="Examples: AAPL, NVDA, TSLA, 2330.TW, 0700.HK",
    ).strip().upper()

    today = dt.date.today()
    start_date = st.date_input("Start date", value=dt.date(2020, 1, 1))
    end_date = st.date_input("End date", value=today, max_value=today)

    st.subheader("Indicator parameters")
    volatility_window = st.number_input(
        "Volatility window (trading days)", min_value=5, max_value=252, value=20
    )
    short_ma = st.number_input(
        "Short moving average", min_value=5, max_value=200, value=20
    )
    long_ma = st.number_input(
        "Long moving average", min_value=10, max_value=400, value=50
    )

    low_percentile = st.slider(
        "Low-volatility percentile", min_value=10, max_value=45, value=33
    )
    high_percentile = st.slider(
        "High-volatility percentile", min_value=55, max_value=90, value=67
    )

    use_volatility_filter = st.checkbox(
        "Use volatility filter for signals", value=True
    )

    transaction_cost_bps = st.number_input(
        "Estimated cost per position change (bps)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

    st.divider()
    st.subheader("Win-rate analysis")
    forward_days = st.selectbox(
        "Select evaluation period",
        options=[5, 10, 20, 30, 60],
        index=2,
        format_func=lambda x: f"{x} trading days",
    )

    run_button = st.button("Run analysis", type="primary", use_container_width=True)

if not ticker:
    st.info("Enter a ticker in the sidebar.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()

if short_ma >= long_ma:
    st.error("The short moving average must be smaller than the long moving average.")
    st.stop()

minimum_rows = max(volatility_window, long_ma) + 10

try:
    with st.spinner(f"Downloading and analyzing {ticker}..."):
        raw = download_stock(ticker, start_date, end_date)

        if len(raw) < minimum_rows:
            st.error(
                f"Only {len(raw)} trading days are available. "
                f"Use at least {minimum_rows} trading days for these settings."
            )
            st.stop()

        df, low_threshold, high_threshold = build_strategy(
            prices=raw,
            volatility_window=int(volatility_window),
            short_ma=int(short_ma),
            long_ma=int(long_ma),
            low_quantile=low_percentile / 100,
            high_quantile=high_percentile / 100,
            use_volatility_filter=use_volatility_filter,
            transaction_cost_bps=float(transaction_cost_bps),
        )
except Exception as exc:
    st.error(str(exc))
    st.stop()

latest = df.dropna(subset=["Close", "Volatility"]).iloc[-1]
latest_signal_row = df[df["Entry_Signal"] != 0].tail(1)

if latest_signal_row.empty:
    latest_signal_text = "No historical entry signal"
    latest_signal_date = "—"
else:
    latest_signal_text = latest_signal_row["Signal_Label"].iloc[0]
    latest_signal_date = latest_signal_row.index[-1].strftime("%Y-%m-%d")

current_position = int(df["Target_Position"].iloc[-1])
position_text = {1: "LONG", -1: "SHORT", 0: "CASH"}[current_position]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest adjusted close", fmt_num(float(latest["Close"])))
c2.metric("Annualized volatility", fmt_pct(float(latest["Volatility"])))
c3.metric("Volatility regime", str(latest["Volatility_Regime"]))
c4.metric("Model position", position_text)
c5.metric("Latest entry signal", latest_signal_text, latest_signal_date)

st.info(
    f"Current volatility bands for this selected history: "
    f"Low < {low_threshold:.2%}; Medium = {low_threshold:.2%}–{high_threshold:.2%}; "
    f"High > {high_threshold:.2%}."
)


signal_stats = calculate_signal_statistics(df, int(forward_days))
buy_stats = signal_stats["BUY"]
short_stats = signal_stats["SHORT"]

st.divider()
st.subheader(f"{forward_days}-Trading-Day Signal Performance")
st.caption(
    f"Only the selected {forward_days}-trading-day evaluation period is displayed."
)

buy_col, short_col = st.columns(2)

with buy_col:
    st.markdown("### ▲ BUY")
    b1, b2 = st.columns(2)
    b1.metric(f"{forward_days}-day win rate", fmt_pct(buy_stats["win_rate"]))
    b2.metric("Valid historical signals", str(buy_stats["count"]))
    b3, b4, b5 = st.columns(3)
    b3.metric("Average return", fmt_pct(buy_stats["average_return"]))
    b4.metric("Best return", fmt_pct(buy_stats["best_return"]))
    b5.metric("Worst return", fmt_pct(buy_stats["worst_return"]))

with short_col:
    st.markdown("### ▼ SHORT")
    s1, s2 = st.columns(2)
    s1.metric(f"{forward_days}-day win rate", fmt_pct(short_stats["win_rate"]))
    s2.metric("Valid historical signals", str(short_stats["count"]))
    s3, s4, s5 = st.columns(3)
    s3.metric("Average short return", fmt_pct(short_stats["average_return"]))
    s4.metric("Best short return", fmt_pct(short_stats["best_return"]))
    s5.metric("Worst short return", fmt_pct(short_stats["worst_return"]))

buy_points = df[df["Entry_Signal"] == 1]
short_points = df[df["Entry_Signal"] == -1]

price_fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.68, 0.32],
)

price_fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ),
    row=1,
    col=1,
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA_Short"],
        mode="lines",
        name=f"SMA {short_ma}",
        line=dict(width=1.4),
    ),
    row=1,
    col=1,
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA_Long"],
        mode="lines",
        name=f"SMA {long_ma}",
        line=dict(width=1.4),
    ),
    row=1,
    col=1,
)

price_fig.add_trace(
    go.Scatter(
        x=buy_points.index,
        y=buy_points["Close"],
        mode="markers",
        name="BUY",
        marker=dict(symbol="triangle-up", size=18, color="#00FF00", line=dict(color="black", width=2)),
        customdata=np.column_stack(
            [
                buy_points["Volatility"].values,
                buy_points["Volatility_Regime"].values,
            ]
        ) if not buy_points.empty else None,
        hovertemplate=(
            "BUY<br>Date=%{x|%Y-%m-%d}<br>Close=%{y:.2f}"
            "<br>Volatility=%{customdata[0]:.2%}"
            "<br>Regime=%{customdata[1]}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)

price_fig.add_trace(
    go.Scatter(
        x=short_points.index,
        y=short_points["Close"],
        mode="markers",
        name="SHORT",
        marker=dict(symbol="triangle-down", size=18, color="#FF00FF", line=dict(color="black", width=2)),
        customdata=np.column_stack(
            [
                short_points["Volatility"].values,
                short_points["Volatility_Regime"].values,
            ]
        ) if not short_points.empty else None,
        hovertemplate=(
            "SHORT<br>Date=%{x|%Y-%m-%d}<br>Close=%{y:.2f}"
            "<br>Volatility=%{customdata[0]:.2%}"
            "<br>Regime=%{customdata[1]}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Volatility"],
        mode="lines",
        name="Annualized volatility",
        line=dict(width=1.5),
        fill="tozeroy",
    ),
    row=2,
    col=1,
)

price_fig.add_hline(
    y=low_threshold,
    line_dash="dash",
    annotation_text=f"Low threshold {low_threshold:.1%}",
    row=2,
    col=1,
)
price_fig.add_hline(
    y=high_threshold,
    line_dash="dash",
    annotation_text=f"High threshold {high_threshold:.1%}",
    row=2,
    col=1,
)

price_fig.update_yaxes(title_text="Price", row=1, col=1)
price_fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)
price_fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
price_fig.update_layout(
    height=820,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=50, r=30, t=120, b=40),
)

st.plotly_chart(price_fig, use_container_width=True)

st.subheader("Backtest")
st.caption(
    "Signals are generated using closing data, then applied from the next trading day. "
    "The strategy remains long or short until an opposite signal occurs."
)

equity_fig = go.Figure()
equity_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Strategy_Equity"],
        mode="lines",
        name="Signal strategy",
    )
)
equity_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Buy_Hold_Equity"],
        mode="lines",
        name="Buy and hold",
    )
)
equity_fig.update_layout(
    title="Growth of $1",
    xaxis_title="Date",
    yaxis_title="Portfolio value",
    hovermode="x unified",
    height=430,
    margin=dict(l=30, r=30, t=60, b=30),
)
st.plotly_chart(equity_fig, use_container_width=True)

strategy_metrics = {
    "Annualized return": annualized_return(df["Strategy_Return"]),
    "Annualized volatility": annualized_volatility(df["Strategy_Return"]),
    "Sharpe ratio": sharpe_ratio(df["Strategy_Return"]),
    "Maximum drawdown": max_drawdown(df["Strategy_Equity"]),
}
benchmark_metrics = {
    "Annualized return": annualized_return(df["Buy_Hold_Return"]),
    "Annualized volatility": annualized_volatility(df["Buy_Hold_Return"]),
    "Sharpe ratio": sharpe_ratio(df["Buy_Hold_Return"]),
    "Maximum drawdown": max_drawdown(df["Buy_Hold_Equity"]),
}

metrics_df = pd.DataFrame(
    {"Signal strategy": strategy_metrics, "Buy and hold": benchmark_metrics}
)
display_metrics = metrics_df.copy()
for row in ["Annualized return", "Annualized volatility", "Maximum drawdown"]:
    display_metrics.loc[row] = display_metrics.loc[row].map(fmt_pct)
display_metrics.loc["Sharpe ratio"] = display_metrics.loc["Sharpe ratio"].map(
    lambda x: fmt_num(float(x))
)
st.dataframe(display_metrics, use_container_width=True)

st.subheader("Signal history")
signal_table = (
    df[df["Entry_Signal"] != 0]
    .reset_index()
    .loc[
        :,
        [
            "Date",
            "Signal_Label",
            "Close",
            "Return",
            "Volatility",
            "Volatility_Regime",
            "SMA_Short",
            "SMA_Long",
        ],
    ]
    .sort_values("Date", ascending=False)
)

if signal_table.empty:
    st.warning("No BUY or SHORT entries were found with the selected settings.")
else:
    formatted_signals = signal_table.copy()
    formatted_signals["Date"] = formatted_signals["Date"].dt.strftime("%Y-%m-%d")
    formatted_signals["Close"] = formatted_signals["Close"].map(
        lambda x: round(float(x), 2)
    )
    formatted_signals["Return"] = formatted_signals["Return"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    formatted_signals["Volatility"] = formatted_signals["Volatility"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    formatted_signals["SMA_Short"] = formatted_signals["SMA_Short"].round(2)
    formatted_signals["SMA_Long"] = formatted_signals["SMA_Long"].round(2)

    st.dataframe(formatted_signals, use_container_width=True, hide_index=True)

    csv = signal_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download signal history as CSV",
        data=csv,
        file_name=f"{ticker}_signal_history.csv",
        mime="text/csv",
    )

with st.expander("How the model decides BUY and SHORT"):
    st.markdown(
        f"""
**BUY entry**

1. Price crosses above the {short_ma}-day moving average.
2. The {short_ma}-day moving average is above the {long_ma}-day moving average.
3. When the volatility filter is enabled, annualized volatility must not be
   above the high-volatility threshold.

**SHORT entry**

1. Price crosses below the {short_ma}-day moving average.
2. The {short_ma}-day moving average is below the {long_ma}-day moving average.
3. When the volatility filter is enabled, annualized volatility must be at
   least above the low-volatility threshold.

**Volatility**

Annualized rolling volatility is calculated as:

`rolling standard deviation of daily returns × sqrt(252)`

High volatility alone does **not** imply that price will fall. The model combines
volatility with trend and price-crossing conditions.
"""
    )

st.warning(
    "Educational research only. This dashboard does not predict prices or guarantee "
    "profit. Short selling can create losses greater than the original capital and "
    "may involve borrowing costs, margin requirements, and forced liquidation."
)
