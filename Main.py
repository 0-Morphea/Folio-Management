import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI

st.set_page_config(layout="wide")

# -------------------------------
# Fonctions d'accès aux données
# -------------------------------

@st.cache_data(ttl=3600)
def get_top_coins(n=200, vs_currency='usd'):
    cg = CoinGeckoAPI()
    coins = cg.get_coins_markets(vs_currency=vs_currency, order='market_cap_desc', per_page=n, page=1)
    df = pd.DataFrame(coins)
    return df[['id', 'symbol', 'name']]

@st.cache_data(ttl=3600)
def fetch_crypto_history(coin_id, days=365, vs_currency='usd'):
    cg = CoinGeckoAPI()
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        prices = data['prices']  # liste de [timestamp, price]
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    except Exception as e:
        st.error(f"Erreur pour {coin_id}: {e}")
        return pd.DataFrame()

def get_current_price(coin_id, vs_currency='usd'):
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

def forecast_price_series(price_series, forecast_days=365, n_simulations=500):
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = price_series.iloc[-1]
    dt = 1  # daily step

    sims = np.zeros((n_simulations, forecast_days+1))
    sims[:, 0] = last_price

    for t in range(1, forecast_days+1):
        shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_simulations)
        sims[:, t] = sims[:, t-1] * np.exp(shocks)

    sim_df = pd.DataFrame(sims.T)
    sim_df.index = [price_series.index[-1] + timedelta(days=i) for i in range(forecast_days+1)]
    quantiles = sim_df.quantile([0.05, 0.50, 0.95]).T
    quantiles.columns = ['q05', 'q50', 'q95']
    return sim_df, quantiles

# -------------------------------
# Fonctions d'analyse de risque
# -------------------------------

def risk_metrics(price_series):
    returns = price_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(365)
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    return volatility, max_drawdown

def risk_label(volatility, max_drawdown):
    if volatility < 0.5 and max_drawdown > -0.15:
        return "Low Risk"
    elif volatility < 1.0 and max_drawdown > -0.30:
        return "Medium Risk"
    else:
        return "High Risk"

# -------------------------------
# Dashboard Agrégé
# -------------------------------

def portfolio_allocation_pie(allocation_dict, cash_value):
    labels = list(allocation_dict.keys()) + ['Cash']
    values = list(allocation_dict.values()) + [cash_value]
    fig = px.pie(names=labels, values=values, title="Répartition de l'Allocation du Capital")
    return fig

# -------------------------------
# Paramètres Utilisateur - Sidebar
# -------------------------------

st.sidebar.header("Configuration du Portfolio")

# Option pour échelle logarithmique
log_scale = st.sidebar.checkbox("Utiliser une échelle logarithmique pour les graphiques", value=False)

# 1. Sélection parmi le Top 200
top_coins_df = get_top_coins(n=200)
coin_options = top_coins_df['name'] + " (" + top_coins_df['symbol'].str.upper() + ")"
coin_dict = dict(zip(coin_options, top_coins_df['id']))

selected_coins = st.sidebar.multiselect("Sélectionnez les cryptos à inclure", options=coin_options,
                                          default=coin_options[:3])
selected_ids = [coin_dict[name] for name in selected_coins]

# 2. Montant total du portefeuille
total_portfolio = st.sidebar.number_input("Montant total du portefeuille ($)", min_value=1000.0, value=100000.0, step=1000.0)

# 3. Allocation entre cash et cryptos
cash_allocation = st.sidebar.slider("Pourcentage d'allocation en cash", min_value=0, max_value=100, value=20, step=5)
crypto_allocation_pct = 100 - cash_allocation

st.sidebar.markdown(f"Allocation globale en cryptos : **{crypto_allocation_pct}%**")

# 4. Pour chaque crypto, définir l'allocation en %
allocation_inputs = {}
if selected_ids:
    st.sidebar.markdown("### Allocation par Crypto")
    total_alloc = 0
    for coin_name in selected_coins:
        alloc = st.sidebar.number_input(f"Allocation pour {coin_name} (%)", min_value=0.0, max_value=100.0, 
                                          value=round(crypto_allocation_pct/len(selected_coins),1), step=0.5, key=coin_name)
        allocation_inputs[coin_name] = alloc
        total_alloc += alloc
    if total_alloc != 0:
        for coin in allocation_inputs:
            allocation_inputs[coin] = allocation_inputs[coin] / total_alloc * crypto_allocation_pct

# 5. Yield sur le cash
yield_cash = st.sidebar.number_input("Yield annuel sur cash (%)", min_value=0.0, value=5.0, step=0.5)

# 6. Fees
st.sidebar.markdown("### Fees")
management_fee = st.sidebar.number_input("Management Fee annuel (%)", min_value=0.0, value=2.0, step=0.1) / 100.0
performance_fee = st.sidebar.number_input("Performance Fee (%)", min_value=0.0, value=20.0, step=1.0) / 100.0

# 7. Période de forecast (en jours)
forecast_days = st.sidebar.number_input("Période de forecast (jours)", min_value=30, value=365, step=30)

# 8. Stress Test
stress_shock = st.sidebar.number_input("Choc de stress (variation en %)", min_value=-100, max_value=0, value=-30, step=5)

# Date d'investissement initiale (pour comparaison)
invest_date = st.sidebar.date_input("Date d'investissement initiale", value=datetime.now()-timedelta(days=365))

# -------------------------------
# Récupération des données
# -------------------------------

price_data = {}
current_prices = {}
for coin_id in selected_ids:
    df = fetch_crypto_history(coin_id, days=365)
    if not df.empty:
        price_data[coin_id] = df
        current_prices[coin_id] = df['price'].iloc[-1]

if not price_data:
    st.error("Aucune donnée historique récupérée pour les cryptos sélectionnées.")
    st.stop()

# Calcul des valeurs actuelles
crypto_values = {}
for coin_name, alloc_pct in allocation_inputs.items():
    coin_id = coin_dict[coin_name]
    alloc_amount = total_portfolio * (alloc_pct / 100)
    crypto_values[coin_name] = alloc_amount  # Supposé acquis au prix actuel

cash_value = total_portfolio * (cash_allocation/100)
portfolio_current = sum(crypto_values.values()) + cash_value

# -------------------------------
# Création des onglets
# -------------------------------

tab1, tab2 = st.tabs(["Investissements Individuels", "Analyse Globale du Portefeuille"])

# ===== Onglet 1 : Investissements Individuels =====
with tab1:
    st.markdown("### Historique & Projections par Actif")
    # Affichage de l'historique pour chaque crypto
    fig_hist = go.Figure()
    for coin_id, df in price_data.items():
        coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
        fig_hist.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name=coin_name))
    if log_scale:
        fig_hist.update_layout(yaxis_type="log")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Projection pour chaque crypto
    st.markdown("#### Projections Individuelles (12 mois)")
    for coin_id in selected_ids:
        coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
        series = price_data[coin_id]['price']
        sim_df, quantiles = forecast_price_series(series, forecast_days=forecast_days, n_simulations=500)
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=quantiles.index, y=quantiles['q05'], mode='lines',
                                      line=dict(color='red', dash='dash'), name='5e percentile'))
        fig_proj.add_trace(go.Scatter(x=quantiles.index, y=quantiles['q95'], mode='lines',
                                      line=dict(color='green', dash='dash'), name='95e percentile', fill='tonexty'))
        fig_proj.add_trace(go.Scatter(x=quantiles.index, y=quantiles['q50'], mode='lines',
                                      line=dict(color='blue'), name='Médiane'))
        fig_proj.update_layout(title=f"Projection de Prix pour {coin_name}",
                               xaxis_title="Date", yaxis_title="Prix ($)",
                               yaxis_type="log" if log_scale else "linear")
        st.plotly_chart(fig_proj, use_container_width=True)
    
    # Analyses complémentaires individuelles
    st.markdown("#### Analyses Complémentaires")
    # Scatter Plot Risque/Retour
    risk_return_data = []
    for coin_id, df in price_data.items():
        coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
        annual_return = df['price'].pct_change().dropna().mean() * 365
        volatility, _ = risk_metrics(df['price'])
        risk_return_data.append({'Crypto': coin_name, 'Annual Return': annual_return, 'Volatility': volatility})
    risk_df = pd.DataFrame(risk_return_data)
    fig_scatter = px.scatter(risk_df, x="Volatility", y="Annual Return", text="Crypto",
                             title="Scatter Plot Risque/Retour", labels={"Volatility": "Volatilité Annualisée", "Annual Return": "Retour Annualisé"})
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Heatmap de Corrélation
    price_df = pd.DataFrame()
    for coin_id, df in price_data.items():
        coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
        price_df[coin_name] = df['price']
    corr = price_df.pct_change().corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Matrice de Corrélation des Rendements")
    st.plotly_chart(fig_corr, use_container_width=True)

# ===== Onglet 2 : Analyse Globale du Portefeuille =====
with tab2:
    st.markdown("### Analyse Globale du Portefeuille")
    st.write(f"**Investissement Initial (au {invest_date.strftime('%Y-%m-%d')}) :** ${total_portfolio:,.2f}")
    st.write(f"**Valeur Actuelle du Portefeuille :** ${portfolio_current:,.2f}")
    global_return = (portfolio_current/total_portfolio - 1)*100
    st.write(f"**Rendement Global :** {global_return:.2f}%")
    
    # --- Forecast Global avec Enveloppe ---
    st.markdown("#### Projection Globale du Portefeuille sur 12 mois (Enveloppe)")
    n_simulations = 500
    # Initialiser un tableau pour les simulations agrégées
    agg_sim = np.zeros((n_simulations, forecast_days+1))
    # Pour chaque crypto, ajouter sa simulation pondérée
    for coin_name, alloc_pct in allocation_inputs.items():
        coin_id = coin_dict[coin_name]
        series = price_data[coin_id]['price']
        sim_df, _ = forecast_price_series(series, forecast_days=forecast_days, n_simulations=n_simulations)
        allocation_amount = total_portfolio * (alloc_pct / 100)
        # Nombre de coins acquis au prix actuel
        num_coins = allocation_amount / current_prices[coin_id]
        # Ajouter la valeur forecastée pour cette crypto
        agg_sim += sim_df.values * num_coins
    # Simulation du cash (déterministe)
    days_array = np.arange(forecast_days+1)
    cash_sim = cash_value * (1 + (yield_cash/100) * (days_array/365))
    # Ajouter le cash (même pour toutes les simulations)
    agg_sim += np.tile(cash_sim, (n_simulations, 1))
    
    # Conversion en DataFrame
    forecast_dates = [list(price_data.values())[0].index[-1] + timedelta(days=i) for i in range(forecast_days+1)]
    agg_sim_df = pd.DataFrame(agg_sim.T, index=forecast_dates)
    # Calcul des quantiles pour l'enveloppe
    agg_quantiles = agg_sim_df.quantile([0.05, 0.50, 0.95]).T
    agg_quantiles.columns = ['q05', 'q50', 'q95']
    
    # Graphique de l'enveloppe
    fig_envelope = go.Figure()
    fig_envelope.add_trace(go.Scatter(x=agg_quantiles.index, y=agg_quantiles['q05'], mode='lines',
                                      line=dict(color='red', dash='dash'), name='5e percentile'))
    fig_envelope.add_trace(go.Scatter(x=agg_quantiles.index, y=agg_quantiles['q95'], mode='lines',
                                      line=dict(color='green', dash='dash'), name='95e percentile', fill='tonexty'))
    fig_envelope.add_trace(go.Scatter(x=agg_quantiles.index, y=agg_quantiles['q50'], mode='lines',
                                      line=dict(color='blue'), name='Médiane'))
    fig_envelope.update_layout(title="Projection Globale du Portefeuille (Enveloppe Forecast)",
                               xaxis_title="Date", yaxis_title="Valeur du Portefeuille ($)",
                               yaxis_type="log" if log_scale else "linear")
    st.plotly_chart(fig_envelope, use_container_width=True)
    
    # --- Visualisation des Fees prévisionnels ---
    st.markdown("#### Projection des Fees Prévisionnels sur 12 mois")
    # Pour le management fee : fee quotidien = (management_fee/365) * valeur forecastée
    daily_mgmt_fee = agg_quantiles['q50'] * (management_fee / 365)
    cumulative_mgmt_fee = np.cumsum(daily_mgmt_fee)
    # Pour le performance fee : appliqué sur le gain par rapport à la valeur actuelle (si positif)
    daily_perf_fee = np.maximum(agg_quantiles['q50'] - portfolio_current, 0) * (performance_fee / forecast_days)
    cumulative_perf_fee = np.cumsum(daily_perf_fee)
    cumulative_total_fee = cumulative_mgmt_fee + cumulative_perf_fee
    
    fig_fees = go.Figure()
    fig_fees.add_trace(go.Scatter(x=agg_quantiles.index, y=cumulative_mgmt_fee, mode='lines',
                                  name='Cumulative Management Fee'))
    fig_fees.add_trace(go.Scatter(x=agg_quantiles.index, y=cumulative_perf_fee, mode='lines',
                                  name='Cumulative Performance Fee'))
    fig_fees.add_trace(go.Scatter(x=agg_quantiles.index, y=cumulative_total_fee, mode='lines',
                                  name='Cumulative Total Fee'))
    fig_fees.update_layout(title="Projection Cumulative des Fees sur 12 mois",
                           xaxis_title="Date", yaxis_title="Fees en $")
    st.plotly_chart(fig_fees, use_container_width=True)
    
    # --- Analyse de Risque Global ---
    portfolio_hist = pd.Series(dtype=float)
    for coin_name, alloc_pct in allocation_inputs.items():
        coin_id = coin_dict[coin_name]
        allocation_amount = total_portfolio * (alloc_pct / 100)
        series = price_data[coin_id]['price']
        val_series = series * (allocation_amount / series.iloc[0])
        if portfolio_hist.empty:
            portfolio_hist = val_series
        else:
            portfolio_hist = portfolio_hist.add(val_series, fill_value=0)
    portfolio_hist += cash_value

    vol_global, md_global = risk_metrics(portfolio_hist)
    st.write(f"**Volatilité Annualisée Globale :** {vol_global:.2%}")
    st.write(f"**Max Drawdown Global :** {md_global:.2%}")
    risk_lvl = risk_label(vol_global, md_global)
    st.write(f"**Niveau de Risque Global :** {risk_lvl}")
    
    st.markdown("#### Évolution Historique du Portefeuille")
    fig_hist_global = go.Figure()
    fig_hist_global.add_trace(go.Scatter(x=portfolio_hist.index, y=portfolio_hist, mode='lines', name='Historique Agrégé'))
    fig_hist_global.update_layout(xaxis_title="Date", yaxis_title="Valeur ($)",
                                  yaxis_type="log" if log_scale else "linear")
    st.plotly_chart(fig_hist_global, use_container_width=True)
