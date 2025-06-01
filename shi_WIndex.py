from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import seaborn as sns
from datetime import datetime, timedelta

# ---- Styling & Optionen ----
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.float_format', '{:.2%}'.format)
#st.set_page_config(page_title="Strategie-Analyse & Risiko-Kennzahlen", layout="wide")

st.set_page_config(
    page_title="SHI Zertifikate im Vergleich",
    page_icon="📊", 
    layout="wide"
)


RISK_FREE_RATE = 0.02  # 2% p.a.

# --- Hilfsfunktionen ---
def to_1d_series(ret):
    if isinstance(ret, pd.DataFrame):
        ret = ret.iloc[:, 0]
    return pd.to_numeric(ret, errors='coerce').dropna()

def load_returns_from_csv(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    close = pd.to_numeric(df['Close'], errors='coerce').ffill().dropna()
    returns = close.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def load_returns_from_yahoo(ticker, start, end):
    df = yf.download(ticker, start=start, end=end+timedelta(days=1), progress=False)['Close'].dropna()
    returns = df.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def sortino_ratio(returns, risk_free=0.0, annualization=252):
    downside = returns[returns < risk_free]
    downside_std = downside.std(ddof=0)
    mean_ret = returns.mean()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    daily_sortino = (mean_ret - risk_free) / downside_std
    return daily_sortino * np.sqrt(annualization)

def omega_ratio(returns, risk_free=0.0):
    gain = (returns > risk_free).sum()
    loss = (returns <= risk_free).sum()
    if loss == 0:
        return np.nan
    return gain / loss

def tail_ratio(returns):
    try:
        return np.percentile(returns, 95) / abs(np.percentile(returns, 5))
    except Exception:
        return np.nan

def calculate_metrics(returns_dict, cumulative_dict):
    metrics = pd.DataFrame()
    for name in returns_dict:
        ret = returns_dict[name]
        cum = cumulative_dict[name]
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        if ret.empty or cum.empty:
            continue
        days = (cum.index[-1] - cum.index[0]).days
        total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1)
        annual_ret = float((1 + total_ret)**(365/days) - 1) if days > 0 else np.nan
        annual_vol = float(ret.std() * np.sqrt(252))
        sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
        sortino = sortino_ratio(ret, risk_free=0.0)
        drawdowns = (cum / cum.cummax() - 1)
        mdd = float(drawdowns.min()) if not drawdowns.empty else np.nan
        calmar = annual_ret / abs(mdd) if (not np.isnan(mdd) and mdd < 0) else np.nan
        var_95 = float(ret.quantile(0.05))
        cvar_95 = float(ret[ret <= var_95].mean())
        omega = omega_ratio(ret, risk_free=0.0)
        tail = tail_ratio(ret)
        win_rate = float(len(ret[ret > 0]) / len(ret))
        avg_win = float(ret[ret > 0].mean())
        avg_loss = float(ret[ret < 0].mean())
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan
        monthly_ret = ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = float((monthly_ret > 0).mean())
        metrics.loc[name, 'Total Return'] = total_ret
        metrics.loc[name, 'Annual Return'] = annual_ret
        metrics.loc[name, 'Annual Volatility'] = annual_vol
        metrics.loc[name, 'Sharpe Ratio'] = sharpe
        metrics.loc[name, 'Sortino Ratio'] = sortino
        metrics.loc[name, 'Max Drawdown'] = mdd
        metrics.loc[name, 'Calmar Ratio'] = calmar
        metrics.loc[name, 'VaR (95%)'] = var_95
        metrics.loc[name, 'CVaR (95%)'] = cvar_95
        metrics.loc[name, 'Omega Ratio'] = omega
        metrics.loc[name, 'Tail Ratio'] = tail
        metrics.loc[name, 'Win Rate'] = win_rate
        metrics.loc[name, 'Avg Win'] = avg_win
        metrics.loc[name, 'Avg Loss'] = avg_loss
        metrics.loc[name, 'Profit Factor'] = profit_factor
        metrics.loc[name, 'Positive Months'] = positive_months
    return metrics

# -- Plots & Analysefunktionen: ALLE klein, dezent, fein --
def plot_performance(cumulative_dict):
    fig, ax = plt.subplots(figsize=(6, 3))
    for name, cum in cumulative_dict.items():
        if cum is None or len(cum) == 0:
            continue
        ax.plot(cum.index, cum / cum.iloc[0], label=name, linewidth=0.5)
    ax.set_title("Kumulative Performance (Start = 1.0)", fontsize=8, pad=8)
    ax.set_xlabel("Datum", fontsize=5)
    ax.set_ylabel("Indexierte Entwicklung", fontsize=5)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=5)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    for name, cum in cumulative_dict.items():
        if cum is None or len(cum) == 0:
            continue
        drawdown = (cum / cum.cummax()) - 1
        if isinstance(drawdown, pd.DataFrame):
            drawdown = drawdown.iloc[:, 0]
        drawdown = pd.Series(drawdown.values, index=drawdown.index)
        drawdown = drawdown.dropna()
        if drawdown.empty or len(drawdown) < 2:
            continue
        x = np.array(drawdown.index)
        y = np.array(drawdown.values).flatten()
        if y.ndim > 1:
            y = y.flatten()
        if len(x) != len(y):
            continue
        ax2.fill_between(x, y, 0, alpha=0.3)
        ax2.plot(x, y, linewidth=0.5, label=name)
    ax2.set_title("Drawdown-Verlauf", fontsize=8, pad=8)
    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=5)
    ax2.tick_params(axis='x', labelsize=5)
    ax2.tick_params(axis='y', labelsize=5)
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig2)

def analyze_correlations(returns_dict):
    returns_clean = {}
    for name, ret in returns_dict.items():
        ret = to_1d_series(ret)
        returns_clean[name] = ret
    returns_df = pd.DataFrame(returns_clean)
    corr_matrix = returns_df.corr()
    if corr_matrix.empty:
        st.warning("Zu wenig Daten für Korrelationsmatrix!")
        return corr_matrix
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 5, "color": "black"}  # kleine Schriftgröße!
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)
    
    ax.set_title("Korrelationsmatrix der täglichen Renditen", fontsize=6, pad=6)
    ax.tick_params(axis='x', labelsize=4)      # X-Achse klein
    ax.tick_params(axis='y', labelsize=4)      # Y-Achse klein
    plt.tight_layout()
    st.pyplot(fig)
    return corr_matrix

def analyze_rolling_performance(returns_dict, window=126):
    rolling_sharpe = pd.DataFrame()
    for name, ret in returns_dict.items():
        ret = to_1d_series(ret)
        if len(ret) < window:
            continue
        rolling_mean = ret.rolling(window).mean() * 252
        rolling_std = ret.rolling(window).std() * np.sqrt(252)
        rolling_sharpe[name] = (rolling_mean - RISK_FREE_RATE) / rolling_std
    if rolling_sharpe.empty:
        st.warning("Zu wenig Daten für rollierende Kennzahlen!")
        return rolling_sharpe
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for name in rolling_sharpe:
        ax.plot(rolling_sharpe.index, rolling_sharpe[name], label=name, linewidth=0.5)
    ax.set_title(f"Rollierender Sharpe Ratio (126-Tage Fenster)", fontsize=8, pad=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)
    return rolling_sharpe

# --------- Streamlit App ---------
def main():
    #st.title("📊 SHI Zertifikate im Vergleich – Performance & Risikoanalyse")
    #st.caption("Strategievergleich mit Benchmarks, Fonds und alternativen Anlagemodellen")

    st.markdown('<h3 style="font-weight:400; margin-bottom:0.2rem;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
    st.caption("Performance-, Risiko- und Benchmarkanalyse auf Monatsbasis")


    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Datenquellen auswählen")
        start = st.date_input("Startdatum", value=datetime(2023, 1, 1))
        end = st.date_input("Enddatum", value=datetime.today())
        uploaded_files = st.file_uploader(
            "Zusatzzertifikate/Strategien (CSV, Close-Spalte)", 
            type="csv", accept_multiple_files=True)
        st.markdown("**Yahoo Finance Ticker (mehrere durch Komma, Zeile, oder Semikolon getrennt):**")
        tickers_input = st.text_area("Ticker", value="0P0000J5K3.F")
        tickers = []
        for line in tickers_input.splitlines():
            tickers += [t.strip() for t in line.replace(";", ",").split(",") if t.strip()]

    # --- Daten laden ---
    returns_dict, cumulative_dict = {}, {}

    # --- Autoload local CSVs ---
    autoload_files = [
        ("SHI_ALPHA_28Mai2025.csv", "SHI_ALPHA"),
        ("SHI_INCOME_28Mai2025.csv", "SHI_INCOME")
    ]
    for filename, displayname in autoload_files:
        try:
            ret, cum = load_returns_from_csv(filename)
            ret = ret.loc[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]
            cum = cum.loc[(cum.index >= pd.Timestamp(start)) & (cum.index <= pd.Timestamp(end))]
            returns_dict[displayname] = ret
            cumulative_dict[displayname] = cum
        except Exception as e:
            st.warning(f"Fehler beim Laden von {filename}: {e}")

    # --- CSV-Uploads ---
    if uploaded_files:
        for file in uploaded_files:
            name = file.name.replace('.csv', '')
            try:
                ret, cum = load_returns_from_csv(file)
                ret = ret.loc[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]
                cum = cum.loc[(cum.index >= pd.Timestamp(start)) & (cum.index <= pd.Timestamp(end))]
                returns_dict[name] = ret
                cumulative_dict[name] = cum
            except Exception as e:
                st.warning(f"Fehler beim Laden von {file.name}: {e}")

    # Yahoo Finance-Ticker
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            display_name = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            display_name = ticker
        try:
            ret, cum = load_returns_from_yahoo(ticker, start, end)
            returns_dict[display_name] = ret
            cumulative_dict[display_name] = cum
        except Exception as e:
            st.warning(f"Fehler beim Laden von {ticker}: {e}")

    # Synchronisiere Zeitachsen aller Serien
    if returns_dict:
        all_indexes = [set(r.index) for r in returns_dict.values() if len(r) > 0]
        if all_indexes:
            common_index = sorted(set.intersection(*all_indexes))
        else:
            common_index = []
        for name in returns_dict:
            returns_dict[name] = returns_dict[name].loc[common_index]
            cumulative_dict[name] = cumulative_dict[name].loc[common_index]

    # --- Tabs ---
    tabs = st.tabs([
    "🚦 Metriken",
    "📈 Performance & Drawdown",
    "📉 Sharpe & Korrelation",
    "📊 Monatsrenditen",
    "🔀 Composite Index"
])

    
    # --- Metrik-Tab ---
    with tabs[0]:
        st.subheader("Erweiterte Risikokennzahlen")
        if not returns_dict:
            st.warning("Bitte Datenquelle(n) hochladen oder Ticker eingeben.")
        else:
            metrics = calculate_metrics(returns_dict, cumulative_dict)
            percent_cols = [
                'Total Return', 'Annual Return', 'Annual Volatility', 'Max Drawdown', 'VaR (95%)',
                'CVaR (95%)', 'Win Rate', 'Avg Win', 'Avg Loss', 'Positive Months'
            ]
            metrics_fmt = metrics.copy()
            for col in percent_cols:
                if col in metrics_fmt.columns:
                    metrics_fmt[col] = (metrics_fmt[col]*100).round(2).astype(str) + '%'
            for col in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Tail Ratio', 'Profit Factor']:
                if col in metrics_fmt.columns:
                    metrics_fmt[col] = metrics_fmt[col].round(2)
            metrics_fmt.index = metrics_fmt.index.to_series().apply(lambda x: f"{x}")
            st.dataframe(metrics_fmt, use_container_width=True, height=350)

    # --- Performance-Tab ---
    with tabs[1]:
        st.subheader("Kumulative Performance & Drawdown")
        if not returns_dict:
            st.warning("Bitte Datenquelle(n) hochladen oder Ticker eingeben.")
        else:
            plot_performance(cumulative_dict)

    # --- Drawdown/Korrelation ---
    with tabs[2]:
        st.subheader("Rolling Sharpe Ratio")
        if not returns_dict:
            st.warning("Bitte Datenquelle(n) hochladen oder Ticker eingeben.")
        else:
            analyze_rolling_performance(returns_dict, window=126)
        st.subheader("Korrelation der Tagesrenditen")
        if returns_dict:
            analyze_correlations(returns_dict)

    # --- Monatsrenditen Heatmap ---
    
    with tabs[3]:
        st.subheader("Monatliche Renditen")
        if returns_dict:
            monthly_returns = pd.DataFrame({
                name: to_1d_series(ret).resample('M').apply(lambda x: (1 + x).prod() - 1)
                for name, ret in returns_dict.items()
            })
            if not monthly_returns.empty:
                fig, ax = plt.subplots(figsize=(7, max(2.2, len(monthly_returns.columns)*0.33)))

                heatmap = sns.heatmap(
                    monthly_returns.T,
                    annot=True,
                    fmt='-.1%',
                    cmap='RdYlGn',
                    center=0,
                    linewidths=0.5,
                    ax=ax,
                    annot_kws={"size": 4, "color": "black", "fontname": "DejaVu Sans"},
                    cbar_kws={'label': '', 'shrink': 0.8}
                )

                ax.set_title("Monatliche Renditen", fontsize=8, pad=10)
                ax.set_xlabel("", fontsize=5)
                ax.set_xticklabels(
                    [pd.to_datetime(label.get_text()).strftime('%Y-%m') for label in ax.get_xticklabels()],
                    rotation=45, ha='right', fontsize=4
                )
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)

                # Colorbar kleiner
                cbar = heatmap.collections[0].colorbar
                cbar.ax.tick_params(labelsize=5)

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Keine Monatsrenditen für diesen Zeitraum vorhanden.")
        else:
            st.warning("Keine Daten vorhanden.")


        # Tab 5: Composite Index
    with tabs[4]:
        st.subheader("🔀 Composite Index aus gewählten Assets")
        
        asset_names = list(returns_dict.keys())
        if len(asset_names) < 2:
            st.info("Bitte mindestens zwei Assets laden, um einen eigenen Index zu bauen.")
        else:
            st.write("**Gewichte für jedes Asset setzen (Summe = 100%):**")
            
            # Gewichtsauswahl pro Asset
            default = [round(1/len(asset_names)*100, 2) for _ in asset_names]
            weights = []
            weight_cols = st.columns(len(asset_names))
            for i, asset in enumerate(asset_names):
                with weight_cols[i]:
                    w = st.number_input(
                        f"{asset}", 
                        min_value=0.0, max_value=100.0, 
                        value=default[i], 
                        step=0.1, 
                        key=f"weight_{asset}"
                    )
                    weights.append(w)
            
            total_weight = sum(weights)
            st.write(f"Summe der Gewichte: **{total_weight:.2f}%**")
            
            if abs(total_weight-100) > 1e-2:
                st.warning("Die Summe der Gewichte muss genau 100% ergeben!")
            else:
                # Index-Konstruktion
                weights_np = np.array(weights) / 100  # Von Prozent auf Bruchteil
                # DataFrame aus Returns, auf gemeinsamen Index
                returns_df = pd.DataFrame({k: to_1d_series(v) for k,v in returns_dict.items()})
                returns_df = returns_df.dropna()  # Nur gemeinsame Zeitachse
                
                custom_index_returns = (returns_df * weights_np).sum(axis=1)
                custom_index_cum = (1 + custom_index_returns).cumprod()
                
                # Add to dicts for gemeinsamen Vergleich
                compare_cum = cumulative_dict.copy()
                compare_cum["Eigener Index"] = custom_index_cum
                
                compare_ret = returns_dict.copy()
                compare_ret["Eigener Index"] = custom_index_returns
                
                # Grafiken
                st.markdown("**Kumulative Performance (Composite Index vs. Einzelassets):**")
                plot_performance(compare_cum)
                
                # Metriken
                st.markdown("**Risikokennzahlen (Composite Index vs. Einzelassets):**")
                metrics = calculate_metrics(compare_ret, compare_cum)
                percent_cols = [
                    'Total Return', 'Annual Return', 'Annual Volatility', 'Max Drawdown', 'VaR (95%)',
                    'CVaR (95%)', 'Win Rate', 'Avg Win', 'Avg Loss', 'Positive Months'
                ]
                metrics_fmt = metrics.copy()
                for col in percent_cols:
                    if col in metrics_fmt.columns:
                        metrics_fmt[col] = (metrics_fmt[col]*100).round(2).astype(str) + '%'
                for col in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Tail Ratio', 'Profit Factor']:
                    if col in metrics_fmt.columns:
                        metrics_fmt[col] = metrics_fmt[col].round(2)
                metrics_fmt.index = metrics_fmt.index.to_series().apply(lambda x: f"{x}")
                st.dataframe(metrics_fmt, use_container_width=True, height=350)





if __name__ == "__main__":
    main()
