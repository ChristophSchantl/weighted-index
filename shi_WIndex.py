# streamlit_app_metrics.py

from st_aggrid import AgGrid, GridOptionsBuilder  # optional; aktuell nicht aktiv verwendet
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import seaborn as sns
import scipy.optimize as opt
from datetime import datetime, timedelta

# -----------------------------
# Globales Styling
# -----------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')

st.set_page_config(
    page_title="SHI Zertifikate im Vergleich",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Parameter / Konventionen
# -----------------------------
TRADING_DAYS = 252
RF_ANNUAL = 0.02
RF_DAILY = (1.0 + RF_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

# -----------------------------
# Hilfsfunktionen — Laden & Preprocessing
# -----------------------------
def to_1d_series(ret: pd.Series | pd.DataFrame) -> pd.Series:
    """Sicherstellen, dass wir eine float Series mit DatetimeIndex bekommen."""
    if isinstance(ret, pd.DataFrame):
        ret = ret.iloc[:, 0]
    s = pd.to_numeric(ret, errors='coerce')
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s.astype(float)

def load_returns_from_csv(file) -> tuple[pd.Series, pd.Series]:
    """CSV mit Datumsindex + Close-Spalte -> Tagesreturns & kumulativ (aus returns)."""
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    close = pd.to_numeric(df['Close'], errors='coerce').ffill().dropna()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    r = close.pct_change().dropna()
    cum = (1.0 + r).cumprod()
    return r, cum

@st.cache_data(show_spinner=False)
def load_returns_from_yahoo_cached(ticker: str, start: datetime, end: datetime) -> tuple[pd.Series, pd.Series]:
    """Yahoo Download mit Cache. End inclusive -> end+1d."""
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)['Close'].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    r = df.pct_change().dropna()
    cum = (1.0 + r).cumprod()
    return r, cum

# -----------------------------
# Kennzahl-Funktionen (konsistent 252)
# -----------------------------
def annualize_geometric(returns: pd.Series) -> float:
    r = returns.dropna()
    N = len(r)
    if N == 0:
        return np.nan
    return float((1.0 + r).prod() ** (TRADING_DAYS / N) - 1.0)

def annualize_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))

def sortino_ratio(returns: pd.Series, rf_daily: float = RF_DAILY, annualization: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    downside = r[r < rf_daily]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    daily_excess = r.mean() - rf_daily
    return float(daily_excess / downside_std * np.sqrt(annualization))

def omega_ratio(returns: pd.Series, rf_daily: float = RF_DAILY) -> float:
    r = returns.dropna()
    gains = (r > rf_daily).sum()
    losses = (r <= rf_daily).sum()
    return np.nan if losses == 0 else float(gains / losses)

def tail_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    try:
        p95 = np.percentile(r, 95)
        p05 = np.percentile(r, 5)
        return float(p95 / abs(p05)) if p05 != 0 else np.nan
    except Exception:
        return np.nan

def format_percent(df: pd.DataFrame, cols: list[str], digits: int = 2) -> pd.DataFrame:
    """Nur Anzeige-Format; Berechnung bleibt ungerundet."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c] * 100).round(digits).astype(str) + '%'
    return out

# -----------------------------
# Kern-Metriken (konsistent aus Returns, gemeinsamer Index)
# -----------------------------
def calculate_metrics(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    1) baue harte Schnittmenge aller Returns
    2) rechne kumulative Pfade aus DIESEN Returns
    3) berechne Kennzahlen konsistent (252-Basis)
    """
    # Matrix bauen & harte Schnittmenge
    returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()})
    returns_df = returns_df.sort_index().dropna(how='any')

    metrics = {}
    for name in returns_df.columns:
        r = returns_df[name].astype(float)
        if r.empty:
            continue

        cum = (1.0 + r).cumprod()
        total_ret = float(cum.iloc[-1] - 1.0)
        ann_ret = annualize_geometric(r)
        ann_vol = annualize_vol(r)
        sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan

        dd = cum / cum.cummax() - 1.0
        mdd = float(dd.min()) if not dd.empty else np.nan
        calmar = ann_ret / abs(mdd) if (not np.isnan(mdd) and mdd < 0) else np.nan

        var_95 = float(r.quantile(0.05))
        cvar_95 = float(r[r <= var_95].mean())

        srt = sortino_ratio(r, rf_daily=RF_DAILY, annualization=TRADING_DAYS)
        omg = omega_ratio(r, rf_daily=RF_DAILY)
        tail = tail_ratio(r)

        win_rate = float((r > 0).mean())
        avg_win = float(r[r > 0].mean())
        avg_loss = float(r[r < 0].mean())
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan

        monthly_ret = r.resample('M').apply(lambda x: (1.0 + x).prod() - 1.0)
        positive_months = float((monthly_ret > 0).mean())

        metrics[name] = {
            'Total Return': total_ret,
            'Annual Return': ann_ret,
            'Annual Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': srt,
            'Max Drawdown': mdd,
            'Calmar Ratio': calmar,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Omega Ratio': omg,
            'Tail Ratio': tail,
            'Win Rate': win_rate,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Positive Months': positive_months
        }

    return pd.DataFrame(metrics).T

# -----------------------------
# Plots (nutzen konsistente Reihen)
# -----------------------------
def plot_performance_from_returns(returns_dict: dict[str, pd.Series]):
    """Plot Performance & Drawdown – intern werden cumulative Pfade aus returns gebaut."""
    returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()})
    returns_df = returns_df.sort_index().dropna(how='any')
    if returns_df.empty:
        st.warning("Keine gemeinsamen Datenpunkte.")
        return

    cum_df = (1.0 + returns_df).cumprod()

    # Performance
    fig, ax = plt.subplots(figsize=(6, 3))
    for name in cum_df.columns:
        y = cum_df[name]
        if name in ["Composite Index", "Eigener Index"]:
            ax.plot(y.index, y / y.iloc[0], label=name, linewidth=0.6, color="black")
        else:
            ax.plot(y.index, y / y.iloc[0], label=name, linewidth=0.4)
    ax.set_title("Kumulative Performance (Start = 1.0)", fontsize=8, pad=8)
    ax.set_xlabel("Datum", fontsize=6)
    ax.set_ylabel("Indexierte Entwicklung", fontsize=6)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)

    # Drawdown
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    for name in cum_df.columns:
        y = cum_df[name]
        dd = y / y.cummax() - 1.0
        if name in ["Composite Index", "Eigener Index"]:
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.18, color="black")
            ax2.plot(dd.index, dd.values, linewidth=0.3, color="black", label=name)
        else:
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.25)
            ax2.plot(dd.index, dd.values, linewidth=0.25, label=name)
    ax2.set_title("Drawdown-Verlauf", fontsize=8, pad=8)
    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6)
    ax2.tick_params(axis='x', labelsize=6)
    ax2.tick_params(axis='y', labelsize=6)
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig2)

def analyze_correlations(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()})
    returns_df = returns_df.sort_index().dropna(how='any')
    corr = returns_df.corr()
    if corr.empty:
        st.warning("Zu wenig Daten für Korrelationsmatrix!")
        return corr

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
        linewidths=0.5, ax=ax, annot_kws={"size": 5, "color": "black"}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)
    ax.set_title("Korrelationsmatrix der täglichen Renditen", fontsize=6, pad=6)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    st.pyplot(fig)
    return corr

def analyze_rolling_performance(returns_dict: dict[str, pd.Series], window: int = 126) -> pd.DataFrame:
    returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()})
    returns_df = returns_df.sort_index().dropna(how='any')
    if returns_df.empty:
        st.warning("Zu wenig Daten für rollierende Kennzahlen!")
        return pd.DataFrame()

    rolling_sharpe = pd.DataFrame(index=returns_df.index)
    for name in returns_df.columns:
        r = returns_df[name]
        if len(r) < window:
            continue
        rolling_mean = r.rolling(window).mean() * TRADING_DAYS
        rolling_std = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        rolling_sharpe[name] = (rolling_mean - RF_ANNUAL) / rolling_std

    if rolling_sharpe.dropna(how='all').empty:
        st.warning("Zu wenig Daten für rollierende Kennzahlen!")
        return rolling_sharpe

    fig, ax = plt.subplots(figsize=(6, 2.6))
    for name in rolling_sharpe.columns:
        ax.plot(rolling_sharpe.index, rolling_sharpe[name], label=name, linewidth=0.35)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage Fenster)", fontsize=8, pad=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=5)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)
    return rolling_sharpe

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.markdown('<h3 style="font-weight:400; margin-bottom:0.2rem;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
    st.caption("Performance-, Risiko- und Benchmarkanalyse (konsistente 252-Tage-Konvention)")

    with st.sidebar:
        st.header("Datenquellen auswählen")
        start = st.date_input("Startdatum", value=datetime(2023, 1, 1))
        end   = st.date_input("Enddatum",  value=datetime.today())

        uploaded_files = st.file_uploader(
            "Zusatzzertifikate/Strategien (CSV, Close-Spalte)",
            type="csv", accept_multiple_files=True
        )

        st.markdown("**Yahoo Finance Ticker (mehrere durch Komma/Zeile/Semikolon getrennt):**")
        tickers_input = st.text_area("Ticker", value="", placeholder="z.B. AAPL, MSFT, GOOG")

        tickers = []
        for line in tickers_input.splitlines():
            tickers += [t.strip() for t in line.replace(";", ",").split(",") if t.strip()]

        st.write("Verarbeitete Ticker:", tickers)

    # ----------------- Daten laden -----------------
    returns_dict: dict[str, pd.Series] = {}

    # Autoload (optional)
    autoload_files = [
        ("SHI_ALPHA_02JUN2025.csv", "SHI_ALPHA"),
        ("SHI_INCOME_02JUN2025.csv", "SHI_INCOME")
    ]
    for filename, displayname in autoload_files:
        try:
            r, _ = load_returns_from_csv(filename)
            r = r.loc[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
            if not r.empty:
                returns_dict[displayname] = r
        except Exception as e:
            st.warning(f"Fehler beim Laden von {filename}: {e}")

    # CSV Uploads
    if uploaded_files:
        for file in uploaded_files:
            try:
                r, _ = load_returns_from_csv(file)
                r = r.loc[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
                name = file.name.replace('.csv', '')
                if not r.empty:
                    returns_dict[name] = r
            except Exception as e:
                st.warning(f"Fehler beim Laden von {file.name}: {e}")

    # Yahoo
    for ticker in tickers:
        try:
            r, _ = load_returns_from_yahoo_cached(ticker, start, end)
            # Display-Name holen (optional; robust)
            try:
                info = yf.Ticker(ticker).fast_info
                display_name = getattr(info, "shortName", None) or ticker
            except Exception:
                display_name = ticker
            if not r.empty:
                returns_dict[display_name] = r
        except Exception as e:
            st.warning(f"Fehler beim Laden von {ticker}: {e}")

    # ----------------- Tabs -----------------
    tabs = st.tabs([
        "🚦 Metriken",
        "📈 Performance & Drawdown",
        "📉 Sharpe & Korrelation",
        "📊 Monatsrenditen",
        "🔀 Composite Index"
    ])

    # --- Metriken ---
    with tabs[0]:
        st.subheader("Erweiterte Risiko-Kennzahlen (252-Tage-Basis)")
        if not returns_dict:
            st.warning("Bitte Datenquelle(n) laden.")
        else:
            metrics = calculate_metrics(returns_dict)
            percent_cols = [
                'Total Return', 'Annual Return', 'Annual Volatility',
                'Max Drawdown', 'VaR (95%)', 'CVaR (95%)',
                'Win Rate', 'Avg Win', 'Avg Loss', 'Positive Months'
            ]
            metrics_fmt = metrics.copy()
            metrics_fmt_display = format_percent(metrics_fmt, percent_cols, digits=2)
            for col in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Tail Ratio', 'Profit Factor']:
                if col in metrics_fmt_display.columns:
                    metrics_fmt_display[col] = metrics_fmt_display[col].round(2)
            st.dataframe(metrics_fmt_display, use_container_width=True, height=360)

        with st.expander("ℹ️ Definitionen"):
            st.markdown(r"""
        **• Annual Return:** geometrisch annualisiert  
        \[
        R_{ann} = \Big(\prod_{t=1}^{N}(1+r_t)\Big)^{\frac{252}{N}} - 1
        \]
        
        **• Annual Volatility:**  
        \[
        \sigma_{ann} = \sigma \cdot \sqrt{252}
        \]
        
        **• Sharpe Ratio:**  
        \[
        \text{Sharpe} = \frac{R_{ann} - r_f}{\sigma_{ann}}, \quad r_f = 2\% \text{ p.a.}
        \]
        
        **• Sortino Ratio:** nur Downside-Std mit Schwelle = täglicher \(r_f\)
        
        **• Max Drawdown:**  
        \[
        \min \Big( \frac{Cum}{Cum^{max}} - 1 \Big)
        \]
        
        **• VaR / CVaR (95%):**  
        5 %-Quantil der Tagesrenditen bzw. dessen Durchschnitt
        """, unsafe_allow_html=False)


    # --- Performance & Drawdown ---
    with tabs[1]:
        st.subheader("Kumulative Performance & Drawdown (konsistente Schnittmenge)")
        if not returns_dict:
            st.warning("Bitte Datenquelle(n) laden.")
        else:
            plot_performance_from_returns(returns_dict)

    # --- Rolling Sharpe & Korrelation ---
    with tabs[2]:
        st.subheader("Rollierender Sharpe Ratio")
        if returns_dict:
            analyze_rolling_performance(returns_dict, window=126)
        st.subheader("Korrelation der Tagesrenditen")
        if returns_dict:
            analyze_correlations(returns_dict)

    # --- Monatsrenditen Heatmap ---
    with tabs[3]:
        st.subheader("Monatliche Renditen")
        if returns_dict:
            returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()})
            returns_df = returns_df.sort_index().dropna(how='any')
            if returns_df.empty:
                st.warning("Keine gemeinsame Schnittmenge.")
            else:
                monthly_returns = returns_df.resample('M').apply(lambda x: (1.0 + x).prod() - 1.0)
                if monthly_returns.empty:
                    st.warning("Keine Monatsrenditen im gewählten Zeitraum.")
                else:
                    fig, ax = plt.subplots(figsize=(7, max(2.2, len(monthly_returns.columns) * 0.33)))
                    sns.heatmap(
                        monthly_returns.T, annot=True, fmt='.1%', cmap='RdYlGn', center=0,
                        linewidths=0.5, ax=ax,
                        annot_kws={"size": 4, "color": "black", "fontname": "DejaVu Sans"},
                        cbar_kws={'label': '', 'shrink': 0.8}
                    )
                    ax.set_title("Monatliche Renditen", fontsize=8, pad=10)
                    ax.set_xlabel("", fontsize=5)
                    ax.set_xticklabels(
                        [pd.to_datetime(lbl.get_text()).strftime('%Y-%m') for lbl in ax.get_xticklabels()],
                        rotation=90, ha='right', fontsize=4
                    )
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning("Keine Daten vorhanden.")

    # --- Composite Index ---
    with tabs[4]:
        st.subheader("🔀 Composite Index aus gewählten Assets")
        asset_names = list(returns_dict.keys())
        num_assets = len(asset_names)

        if num_assets < 2:
            st.info("Bitte mindestens zwei Assets laden.")
        else:
            st.markdown("**Gewichte einstellen (Summe = 100%):**")
            # harte Schnittmenge für die Rechenbasis
            returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()}).sort_index().dropna(how='any')

            # Optimale Sharpe-Gewichte (Float) – nur zur Initialisierung/Anzeige
            def neg_sharpe(weights):
                mu = returns_df.mean() * TRADING_DAYS
                cov = returns_df.cov() * TRADING_DAYS
                port_ret = float(np.dot(mu, weights))
                port_vol = float(np.sqrt(weights.T @ cov.values @ weights))
                return -((port_ret - RF_ANNUAL) / port_vol) if port_vol > 0 else np.inf

            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = tuple((0.0, 1.0) for _ in range(num_assets))
            x0 = np.ones(num_assets) / num_assets
            res = opt.minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
            opt_w = res.x if res.success else x0
            opt_w_pct_display = (opt_w * 100).round(0).astype(int)

            # Slider
            sliders = []
            cols = st.columns(num_assets)
            residual = 100
            for i in range(num_assets - 1):
                key = f"weight_{asset_names[i]}_slider"
                default_val = int(opt_w_pct_display[i])
                val = cols[i].slider(asset_names[i], 0, residual, value=min(default_val, residual), step=1, key=key)
                sliders.append(val)
                residual -= val
            last_val = max(0, 100 - sum(sliders))
            sliders.append(last_val)
            cols[-1].number_input(f"{asset_names[-1]} (auto)", min_value=0, max_value=100, value=last_val, step=1, disabled=True)

            total_weight = sum(sliders)
            st.markdown(
                f"<div style='margin-top:10px;margin-bottom:4px;font-size:1.05em;'><b>Summe der Gewichte: "
                f"<span style='color:{'#3cb371' if total_weight == 100 else '#e74c3c'};'>{total_weight:.0f}%</span></b></div>",
                unsafe_allow_html=True
            )

            # Anzeige optimale Gewichte
            st.markdown("<b>Optimale Gewichtung (Sharpe):</b>", unsafe_allow_html=True)
            col_count = min(4, num_assets)
            cols2 = st.columns(col_count)
            for i, (asset, w) in enumerate(zip(asset_names, opt_w_pct_display)):
                with cols2[i % col_count]:
                    st.markdown(
                        f"""
                        <div style='border-radius:0.6em;border:1px solid #b4d5ee;background:#f8fbfd;
                                   padding:0.6em 0.8em;margin-bottom:0.5em;text-align:center;min-width:110px;'>
                            <div style='font-weight:600;'>{asset}</div>
                            <div style='font-size:1.2em;color:#146eb4;font-weight:700;'>{int(w)}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Composite – rechnen mit Float-Gewichten
            weights_np = np.array(sliders, dtype=float) / 100.0
            # numerische Toleranz: re-normalize
            s = weights_np.sum()
            if s > 0:
                weights_np = weights_np / s

            composite_returns = (returns_df * weights_np).sum(axis=1)
            compare_returns = {**returns_dict}
            compare_returns["Composite Index"] = composite_returns

            st.markdown("**Kumulative Performance (Composite vs. Einzelassets):**")
            plot_performance_from_returns(compare_returns)

            st.markdown("**Risikokennzahlen (Composite vs. Einzelassets):**")
            metrics_comp = calculate_metrics(compare_returns)
            percent_cols = [
                'Total Return', 'Annual Return', 'Annual Volatility',
                'Max Drawdown', 'VaR (95%)', 'CVaR (95%)',
                'Win Rate', 'Avg Win', 'Avg Loss', 'Positive Months'
            ]
            metrics_comp_display = format_percent(metrics_comp, percent_cols, digits=2)
            for col in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Tail Ratio', 'Profit Factor']:
                if col in metrics_comp_display.columns:
                    metrics_comp_display[col] = metrics_comp_display[col].round(2)
            st.dataframe(metrics_comp_display, use_container_width=True, height=360)

if __name__ == "__main__":
    main()
