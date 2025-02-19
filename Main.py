import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cvxpy as cp
from datetime import datetime, timedelta

# ------------------------------
# Fonctions d'import / génération de données
# ------------------------------

@st.cache_data
def load_excel_data(uploaded_file):
    """
    Charge un fichier Excel et retourne un DataFrame avec l'index en datetime.
    Le fichier doit contenir une colonne de dates (ou être indexé par dates)
    et une colonne par crypto.
    """
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            st.error("L'index n'est pas au format date.")
    df.sort_index(inplace=True)
    return df

def generate_crypto_data(start_date, end_date, assets):
    """
    Génère des données simulées (random walk) pour une liste d'actifs.
    """
    dates = pd.date_range(start_date, end_date)
    data = pd.DataFrame(index=dates)
    np.random.seed(42)
    for asset in assets:
        # Simulation d'un random walk avec drift
        price = 100 + np.cumsum(np.random.normal(loc=0.1, scale=2, size=len(dates)))
        data[asset] = price
    return data

# ------------------------------
# Fonctionnalité 1 : Backtesting SMA Crossover
# ------------------------------

def backtest_sma(data, short_window=10, long_window=30):
    """Backtest simple d'une stratégie SMA crossover pour chaque crypto."""
    signals = {}
    returns = {}
    
    for asset in data.columns:
        df = pd.DataFrame()
        df["price"] = data[asset]
        df["SMA_short"] = df["price"].rolling(window=short_window).mean()
        df["SMA_long"] = df["price"].rolling(window=long_window).mean()
        df.dropna(inplace=True)
        df["signal"] = np.where(df["SMA_short"] > df["SMA_long"], 1, 0)
        df["position"] = df["signal"].shift(1).fillna(0)
        df["ret"] = df["price"].pct_change()
        df["strategy_ret"] = df["position"] * df["ret"]
        df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
        signals[asset] = df
        returns[asset] = df["strategy_ret"]
        
    return signals, returns

# ------------------------------
# Fonctionnalité 2 : Optimisation de Portefeuille (Markowitz)
# ------------------------------

def markowitz_optimization(returns, risk_free_rate=0.0):
    """
    Optimise la répartition du portefeuille en maximisant le ratio de Sharpe.
    """
    mu = returns.mean().values
    sigma = returns.cov().values
    n = len(mu)
    
    # Variables d'optimisation
    w = cp.Variable(n)
    port_return = mu @ w
    port_vol = cp.quad_form(w, sigma) ** 0.5
    sharpe = (port_return - risk_free_rate) / port_vol
    
    problem = cp.Problem(cp.Maximize(sharpe),
                         [cp.sum(w) == 1,
                          w >= 0])
    problem.solve()
    
    optimal_weights = w.value
    expected_sharpe = sharpe.value
    return dict(zip(returns.columns, optimal_weights)), expected_sharpe

# ------------------------------
# Fonctionnalité 3 : Visualisation et Analyse de Corrélation
# ------------------------------

def plot_correlation_heatmap(data):
    corr = data.pct_change().corr()
    fig = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Matrice de Corrélation des Rendements")
    return fig

# ------------------------------
# Fonctionnalité 4 : Simulation "What-If" et Scénarios de Marché
# ------------------------------

def simulate_scenarios(data, shock=-0.2, days=30):
    """
    Simule un scénario de marché en appliquant un choc aux prix et une reprise linéaire.
    """
    last_prices = data.iloc[-1]
    shocked_prices = last_prices * (1 + shock)
    
    sim_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=days)
    sim_data = pd.DataFrame(index=sim_dates, columns=data.columns)
    
    for asset in data.columns:
        sim_data[asset] = np.linspace(shocked_prices[asset], last_prices[asset], days)
    
    return sim_data

# ------------------------------
# Interface Streamlit
# ------------------------------

st.title("Portfolio Management Crypto")

st.sidebar.header("Configuration du Portfolio")

# Choix de la source des données
data_source = st.sidebar.radio("Source des données historiques", ("Données simulées", "Fichier Excel"))

if data_source == "Fichier Excel":
    uploaded_file = st.sidebar.file_uploader("Importer un fichier Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            data = load_excel_data(uploaded_file)
            st.success("Fichier importé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors de l'importation du fichier: {e}")
            st.stop()
    else:
        st.info("Veuillez importer un fichier Excel pour continuer.")
        st.stop()
else:
    # Saisie des cryptos souhaitées
    asset_input = st.sidebar.text_input("Liste des cryptos (séparées par des virgules)", "Bitcoin,Ethereum,Litecoin")
    assets = [asset.strip() for asset in asset_input.split(",") if asset.strip() != ""]
    # Période de simulation
    start_date = st.sidebar.date_input("Date de début", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("Date de fin", datetime.now())
    if start_date >= end_date:
        st.error("La date de début doit être antérieure à la date de fin.")
        st.stop()
    data = generate_crypto_data(start_date, end_date, assets)

st.header("1. Visualisation des Données Historiques")
st.line_chart(data)

# ------------------------------
# Backtesting SMA Crossover
# ------------------------------

st.header("2. Backtesting de la Stratégie SMA Crossover")
selected_asset = st.selectbox("Sélectionnez un actif pour le backtest", data.columns)
signals_dict, returns_dict = backtest_sma(data)
df_signal = signals_dict[selected_asset].reset_index()

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=df_signal["index"], y=df_signal["price"], mode='lines', name='Prix'))
fig_bt.add_trace(go.Scatter(x=df_signal["index"], y=df_signal["SMA_short"], mode='lines', name='SMA Short'))
fig_bt.add_trace(go.Scatter(x=df_signal["index"], y=df_signal["SMA_long"], mode='lines', name='SMA Long'))
st.plotly_chart(fig_bt, use_container_width=True)

st.subheader("Performance de la stratégie")
st.line_chart(df_signal.set_index("index")["cum_strategy"])

# ------------------------------
# Optimisation de Portefeuille (Markowitz)
# ------------------------------

st.header("3. Optimisation de Portefeuille (Markowitz)")
# Pour l'optimisation, on utilise les rendements quotidiens sur toutes les cryptos
returns_df = data.pct_change().dropna()
optimal_weights, exp_sharpe = markowitz_optimization(returns_df)

st.write("Poids optimaux recommandés :")
st.write(optimal_weights)
st.write(f"Ratio de Sharpe attendu : {exp_sharpe:.2f}")

# ------------------------------
# Analyse de Corrélation
# ------------------------------

st.header("4. Analyse de Corrélation et Diversification")
fig_corr = plot_correlation_heatmap(data)
st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------
# Simulation "What-If" et Scénarios de Marché
# ------------------------------

st.header("5. Simulations 'What-If' et Scénarios de Marché")
shock_val = st.slider("Intensité du choc (%)", min_value=-50, max_value=0, value=-30, step=1)
simulated_data = simulate_scenarios(data, shock=shock_val/100, days=30)
st.line_chart(simulated_data)

st.subheader("Impact sur le Portefeuille Optimisé")
# Valorisation du portefeuille optimisé avant le choc
last_prices = data.iloc[-1]
portfolio_value_before = sum(last_prices[asset] * optimal_weights.get(asset, 0) for asset in data.columns)
# Valorisation après le choc (dernier jour de la simulation)
sim_last_prices = simulated_data.iloc[-1]
portfolio_value_after = sum(sim_last_prices[asset] * optimal_weights.get(asset, 0) for asset in data.columns)
variation = ((portfolio_value_after - portfolio_value_before)/portfolio_value_before)*100

st.write(f"Valeur du portefeuille avant le choc : {portfolio_value_before:.2f}")
st.write(f"Valeur du portefeuille après simulation : {portfolio_value_after:.2f}")
st.write(f"Variation : {variation:.2f}%")

