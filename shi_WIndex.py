from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import seaborn as sns
import scipy.optimize as opt
from datetime import datetime, timedelta

# ---- Styling & Optionen ----
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.float_format', '{:.2%}'.format)

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

# -- Plots & Analysefunktionen --
def plot_performance(cumulative_dict):
    fig, ax = plt.subplots(figsize=(6, 3))
    for name, cum in cumulative_dict.items():
        if cum is None or len(cum) == 0:
            continue
        line_kwargs = {'linewidth': 0.3, 'label': name}
        if name in ["Composite Index", "Eigener Index"]:
            line_kwargs.update({'color': 'black'})
        ax.plot(cum.index, cum / cum.iloc[0], **line_kwargs)
    ax.set_title("Kumulative Performance (Start = 1.0)", fontsize=8, pad=8)
    ax.set_xlabel("Datum", fontsize=5)
    ax.set_ylabel("Indexierte Entwicklung", fontsize=5)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=5)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)

    # Drawdown
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    for name, cum in cumulative_dict.items():
        if cum is None or len(cum) == 0:
            continue
        drawdown = (cum / cum.cummax()) - 1
        if isinstance(drawdown, pd.DataFrame):
            drawdown = drawdown.iloc[:, 0]
        drawdown = drawdown.dropna()
        fill_kwargs = {'alpha': 0.3}
        line_kwargs = {}
        if name in ["Composite Index", "Eigener Index"]:
            fill_kwargs.update({'color': 'black', 'alpha': 0.18})
            line_kwargs.update({'color': 'black', 'linewidth': 0.25, 'label': name})
        else:
            line_kwargs.update({'linewidth': 0.25, 'label': name})
        ax2.fill_between(drawdown.index, drawdown.values, 0, **fill_kwargs)
        ax2.plot(drawdown.index, drawdown.values, **line_kwargs)
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
    returns_clean = {name: to_1d_series(ret) for name, ret in returns_dict.items()}
    returns_df = pd.DataFrame(returns_clean)
    corr_matrix = returns_df.corr()
    if corr_matrix.empty:
        st.warning("Zu wenig Daten für Korrelationsmatrix!")
        return corr_matrix
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5,
        ax=ax, annot_kws={"size": 5, "color": "black"}
    )
    ax.collections[0].colorbar.ax.tick_params(labelsize=5)
    ax.set_title("Korrelationsmatrix der täglichen Renditen", fontsize=6, pad=6)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    st.pyplot(fig)
    return corr_matrix

def analyze_rolling_performance(returns_dict, window=126):
    rolling_sharpe = pd.DataFrame()
    for name, ret in returns_dict.items():
        s = to_1d_series(ret)
        if len(s) < window:
            continue
        rolling_mean = s.rolling(window).mean() * 252
        rolling_std = s.rolling(window).std() * np.sqrt(252)
        rolling_sharpe[name] = (rolling_mean - RISK_FREE_RATE) / rolling_std
    if rolling_sharpe.empty:
        st.warning("Zu wenig Daten für rollierende Kennzahlen!")
        return rolling_sharpe
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for name in rolling_sharpe:
        ax.plot(rolling_sharpe.index, rolling_sharpe[name], label=name, linewidth=0.3)
    ax.set_title(f"Rollierender Sharpe Ratio (126-Tage Fenster)", fontsize=8, pad=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=5)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)
    return rolling_sharpe

# --------- Streamlit App ---------
def main():
    st.markdown('<h3 style="font-weight:400; margin-bottom:0.2rem;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
    st.caption("Performance-, Risiko- und Benchmarkanalyse auf Monatsbasis")

    # Sidebar
    with st.sidebar:
        st.header("Datenquellen auswählen")
        start = st.date_input("Startdatum", value=datetime(2023, 1, 1))
        end   = st.date_input("Enddatum",  value=datetime.today())
        uploaded_files = st.file_uploader(
            "Zusatzzertifikate/Strategien (CSV, Close-Spalte)", 
            type="csv", accept_multiple_files=True)
        tickers_input = st.text_area(
            "Yahoo Finance Ticker (AAPL, MSFT, ...)", placeholder="z.B. AAPL, MSFT, GOOG"
        )
        tickers = [t.strip() for line in tickers_input.splitlines() for t in line.replace(";", ",").split(",") if t.strip()]
        st.write("Verarbeitete Ticker:", tickers)

    # Daten laden
    returns_dict, cumulative_dict = {}, {}
    autoload_files = [("SHI_ALPHA_02JUN2025.csv", "SHI_ALPHA"), ("SHI_INCOME_02JUN2025.csv", "SHI_INCOME")]
    for fname, name in autoload_files:
        try:
            r, c = load_returns_from_csv(fname)
            returns_dict[name] = r.loc[start:end]
            cumulative_dict[name] = c.loc[start:end]
        except Exception as e:
            st.warning(f"Fehler beim Laden von {fname}: {e}")
    if uploaded_files:
        for file in uploaded_files:
            name = file.name.replace('.csv', '')
            try:
                r, c = load_returns_from_csv(file)
                returns_dict[name] = r.loc[start:end]
                cumulative_dict[name] = c.loc[start:end]
            except Exception as e:
                st.warning(f"Fehler beim Laden von {file.name}: {e}")
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            disp = info.get("shortName", ticker)
        except:
            disp = ticker
        try:
            r, c = load_returns_from_yahoo(ticker, start, end)
            returns_dict[disp] = r
            cumulative_dict[disp] = c
        except Exception as e:
            st.warning(f"Fehler beim Laden von {ticker}: {e}")
    # Synchronisiere Index
    if returns_dict:
        idx = sorted(set.intersection(*[set(r.index) for r in returns_dict.values()]))
        for k in returns_dict:
            returns_dict[k] = returns_dict[k].loc[idx]
            cumulative_dict[k] = cumulative_dict[k].loc[idx]

    # Tabs
    tabs = st.tabs(["🚦 Metriken","📈 Performance & Drawdown","📉 Sharpe & Korrelation","📊 Monatsrenditen","🔀 Composite Index"])
    # Tab 0
    with tabs[0]:
        st.subheader("Erweiterte Risikokennzahlen")
        if not returns_dict:
            st.warning("Bitte Datenquelle(n) hochladen oder Ticker eingeben.")
        else:
            metrics = calculate_metrics(returns_dict, cumulative_dict)
            # format and display dataframe...
            st.dataframe(metrics, use_container_width=True, height=350)
    # Tab 1
    with tabs[1]:
        st.subheader("Kumulative Performance & Drawdown")
        if returns_dict:
            plot_performance(cumulative_dict)
        else:
            st.warning("Keine Daten vorhanden.")
    # Tab 2
    with tabs[2]:
        st.subheader("Rolling Sharpe Ratio")
        if returns_dict:
            analyze_rolling_performance(returns_dict)
            st.subheader("Korrelation der Tagesrenditen")
            analyze_correlations(returns_dict)
        else:
            st.warning("Keine Daten vorhanden.")
    # Tab 3
    with tabs[3]:
        st.subheader("Monatliche Renditen")
        if returns_dict:
            monthly = pd.DataFrame({n: to_1d_series(r).resample('M').apply(lambda x: (1+x).prod()-1)
                                     for n,r in returns_dict.items()})
            if not monthly.empty:
                fig, ax = plt.subplots(figsize=(7, max(2.2, len(monthly.columns)*0.33)))
                sns.heatmap(monthly.T, annot=True, fmt='-.1%', cmap='RdYlGn', center=0,
                            linewidths=0.5, ax=ax, annot_kws={"size":4})
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Keine Monatsrenditen vorhanden.")
        else:
            st.warning("Keine Daten vorhanden.")
    # Tab 4: Composite Index
    with tabs[4]:
        st.subheader("🔀 Composite Index aus gewählten Assets")
        asset_names = list(returns_dict.keys())
        n = len(asset_names)
        if n < 2:
            st.info("Bitte mindestens zwei Assets laden.")
        else:
            st.markdown("**Gewichte einstellen (Summe=100%):**")
            returns_df = pd.DataFrame({k: to_1d_series(v) for k,v in returns_dict.items()}).dropna(how='any')

            # Optimierung
            def neg_sharpe(w):
                r = returns_df.mean(); C = returns_df.cov()
                port_ret = (r*w).sum()*252
                port_vol = np.sqrt(w.T @ (C*252) @ w)
                return -((port_ret - RISK_FREE_RATE)/port_vol) if port_vol>0 else np.inf
            cons = {'type':'eq','fun':lambda w: w.sum()-1}
            bounds = [(0,1)]*n; x0 = np.ones(n)/n
            res = opt.minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                opt_w = res.x; opt_w_pct = (opt_w*100).round(1)
            else:
                st.error("⚠️ Optimierung fehlgeschlagen, Gleichgewichtung genutzt.")
                opt_w = np.ones(n)/n; opt_w_pct = (opt_w*100).round(1)
            opt_map = dict(zip(asset_names, opt_w_pct))

                        # Button & Session State
            if st.button("Setze optimale Sharpe-Ratio-Gewichte"):
                # Setze alle Slider auf optimales Niveau
                for a, p in opt_map.items():
                    st.session_state[f"w_{a}"] = int(p)
                st.session_state['use_opt'] = True
                # Rerun, falls verfügbar, um Sliders sofort anzuwenden
                if hasattr(st, 'experimental_rerun'):
                    st.experimental_rerun()

            use_opt = st.session_state.get('use_opt', False)

            # Sliders starts here
            sliders=[]; rem=100; cols=st.columns(n)
            for i,a in enumerate(asset_names[:-1]):
                key=f"w_{a}"; val = st.session_state.get(key, int(100/n))
                val = min(val, rem); st.session_state[key]=val
                sliders.append(cols[i].slider(a, 0, rem, val, key=key)); rem-=sliders[-1]
            sliders.append(rem); cols[-1].number_input(asset_names[-1]+' (auto)',0,100,rem,disabled=True)

                        # Anzeige der optimalen Gewichte (Design)
            st.markdown("""
                <div style='margin-bottom:10px;font-size:1.1em;'>
                  <span style='font-size:1.3em;margin-right:8px;'>🔎</span>
                  <b>Optimale Gewichtung für maximales Sharpe Ratio:</b>
                </div>
            """, unsafe_allow_html=True)
            cols_cards = st.columns(min(4, n))
            for i, (asset, pct) in enumerate(opt_map.items()):
                with cols_cards[i % len(cols_cards)]:
                    st.markdown(f"""
                    <div style='
                        border-radius: 0.6em;
                        border: 1px solid #b4d5ee;
                        background: #f8fbfd;
                        padding: 0.7em 1em;
                        margin-bottom: 0.5em;
                        text-align: center;
                        box-shadow: 0 1px 4px #dbe9f4bb;
                    '>
                        <div style='font-size:1em;font-weight:600;'>{asset}</div>
                        <div style='font-size:1.3em;color:#146eb4;font-weight:700;'>{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Auswahl der Gewichte
            w_np = opt_w if use_opt else np.array(sliders)/100
            total = w_np.sum()*100
            st.markdown(f"**Summe der Gewichte:** {total:.1f}%")

            # Composite Index
            comp_ret = (returns_df * w_np).sum(axis=1)
            comp_cum = (1+comp_ret).cumprod()
            compare_cum = cumulative_dict.copy(); compare_cum['Composite Index']=comp_cum
            compare_ret = returns_dict.copy(); compare_ret['Composite Index']=comp_ret

            st.markdown("**Kumulative Performance:**"); plot_performance(compare_cum)
            st.markdown("**Risikokennzahlen:**")
            df_m = calculate_metrics(compare_ret, compare_cum)
            st.dataframe(df_m, use_container_width=True, height=350)

if __name__=="__main__":
    main()
