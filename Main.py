import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI

# -------------------------------
# Fonctions d'accès aux données
# -------------------------------

@st.cache_data(ttl=3600)
def get_top_coins(n=200, vs_currency='usd'):
    """
    Récupère les top n cryptos par capitalisation via CoinGecko.
    Retourne une DataFrame avec les colonnes: id, symbol, name.
    """
    cg = CoinGeckoAPI()
    coins = cg.get_coins_markets(vs_currency=vs_currency, order='market_cap_desc', per_page=n, page=1)
    df = pd.DataFrame(coins)
    # On garde uniquement l'id, le symbole et le nom
    return df[['id', 'symbol', 'name']]

@st.cache_data(ttl=3600)
def fetch_crypto_history(coin_id, days=365, vs_currency='usd'):
    """
    Récupère l'historique des prix pour une crypto donnée via CoinGecko.
    Retourne un DataFrame indexé par la date, avec une colonne "price".
    """
    cg = CoinGeckoAPI()
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        prices = data['prices']  # liste de [timestamp, price]
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Erreur pour {coin_id}: {e}")
        return pd.DataFrame()

def get_current_price(coin_id, vs_currency='usd'):
    """
    Récupère le prix actuel d'une crypto.
    """
    cg = CoinGeckoAPI()
    try:
        data = cg.get_price(ids=coin_id, vs_currencies=vs_currency)
        return data[coin_id][vs_currency]
    except Exception as e:
        st.error(f"Erreur lors de la récupération du prix pour {coin_id}: {e}")
        return None

# -------------------------------
# Fonctions de simulation & forecast
# -------------------------------

def forecast_price_series(price_series, forecast_days=30, n_simulations=100):
    """
    Effectue une simulation Monte-Carlo pour forecast le cours futur d'une crypto.
    On utilise ici une modélisation par mouvement Brownien géométrique.
    Retourne un DataFrame moyen des trajectoires simulées et l'ensemble des simulations.
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = price_series.iloc[-1]

    dt = 1  # daily steps
    simulations = []

    for i in range(n_simulations):
        prices = [last_price]
        for d in range(forecast_days):
            # Génération d'un rendement aléatoire
            shock = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        simulations.append(prices)

    sim_df = pd.DataFrame(simulations).T
    forecast_mean = sim_df.mean(axis=1)
    forecast_mean.index = [price_series.index[-1] + timedelta(days=i) for i in range(forecast_days+1)]
    return forecast_mean, sim_df

def apply_fees(portfolio_value, management_fee, performance_fee, initial_value):
    """
    Applique des frais de gestion et de performance sur la variation du portefeuille.
    Pour simplifier, on déduit le management fee en proportion linéaire sur la période, et
    le performance fee sur le gain (la partie positive).
    """
    # Management fee déduit linéairement sur la variation
    mg_fee = portfolio_value * management_fee
    gain = max(portfolio_value - initial_value, 0)
    perf_fee = gain * performance_fee
    net_value = portfolio_value - mg_fee - perf_fee
    return net_value

# -------------------------------
# Fonctions de Stress Test & Risk Analysis
# -------------------------------

def stress_test(portfolio_value_series, shock_percent):
    """
    Simule un stress test en appliquant un choc instantané sur la valeur du portefeuille.
    Retourne une série avec la valeur instantanée après choc.
    """
    shock_value = portfolio_value_series.iloc[-1] * (1 + shock_percent/100)
    stressed = portfolio_value_series.copy()
    stressed.iloc[-1] = shock_value
    return stressed

def risk_metrics(price_series):
    """
    Calcule quelques indicateurs de risque simples : volatilité (std), max drawdown.
    """
    returns = price_series.pct_change().dropna()
    volatility = returns.std()
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    return volatility, max_drawdown

# -------------------------------
# Interface Streamlit
# -------------------------------

st.title("Portfolio Management & Forecasting pour Cryptos")

st.sidebar.header("Configuration du Portfolio")

# 1. Sélection parmi le Top 200
top_coins_df = get_top_coins(n=200)
coin_options = top_coins_df['name'] + " (" + top_coins_df['symbol'].str.upper() + ")"
coin_ids = top_coins_df['id'].tolist()
coin_dict = dict(zip(coin_options, coin_ids))

selected_coins = st.sidebar.multiselect("Sélectionnez les cryptos à inclure", options=coin_options,
                                          default=coin_options[:3])  # par défaut 3 cryptos

selected_ids = [coin_dict[name] for name in selected_coins]

# 2. Montant total du portefeuille et allocation
total_portfolio = st.sidebar.number_input("Montant total du portefeuille ($)", min_value=1000.0, value=100000.0, step=1000.0)

st.sidebar.markdown("### Allocation du portefeuille")
# Pour simplifier, on suppose une allocation égale entre les cryptos sélectionnées et le cash.
n_assets = len(selected_ids)
cash_allocation = st.sidebar.slider("Pourcentage d'allocation en cash", min_value=0, max_value=100, value=20, step=5)
crypto_allocation = 100 - cash_allocation

# Répartition égale entre l
