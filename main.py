import os
import requests
import pandas as pd
import numpy as np
import re
import json
import urllib.parse
import feedparser
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from google import genai
from google.genai import types
from datetime import datetime, timedelta

# SECURE ENVIRONMENT VARIABLES 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN")

app = FastAPI(title="MLFP Quant Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL CRASH HANDLER ---
# Forces FastAPI to send exact Python errors to the frontend WITH CORS headers,
# preventing the browser from throwing a generic "Failed to fetch" security block.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Backend Diagnostic: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": "*"}
    )

UPSTOX_KEYS = {
    'SBIN': 'NSE_EQ|INE062A01020',
    'RELIANCE': 'NSE_EQ|INE002A01018',
    'INFY': 'NSE_EQ|INE009A01021',
    'TCS': 'NSE_EQ|INE467B01029',
    'HDFCBANK': 'NSE_EQ|INE040A01034',
    'ICICIBANK': 'NSE_EQ|INE090A01021'
}

def fetch_upstox_data(symbol, years=2):
    instrument_key = UPSTOX_KEYS.get(symbol)
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    url = f'https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    
    try:
        # Increased timeout and catching all connection errors
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json().get('data', {}).get('candles', [])
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume', 'OI'])
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            return df.sort_values('Date').reset_index(drop=True)[['Date', 'ClosePrice', 'Volume']]
    except Exception:
        pass
    return pd.DataFrame()

def generate_hybrid_features(df):
    df = df.copy()
    df['Log_Returns'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Forward_Return'] = np.log(df['ClosePrice'].shift(-1) / df['ClosePrice'])
    
    sma_20_raw = df['ClosePrice'].rolling(window=20).mean()
    df['SMA_20_Dist'] = (df['ClosePrice'] - sma_20_raw) / sma_20_raw
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    delta = df['ClosePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))
    
    # Institutional MACD Momentum Indicators
    ema_12 = df['ClosePrice'].ewm(span=12, adjust=False).mean()
    ema_26 = df['ClosePrice'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    for col in ['Log_Returns', 'RSI_14']:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2)
        
    ar_preds, returns_series = [], df['Log_Returns'].values
    for i in range(len(df)):
        if i < 20: ar_preds.append(np.nan)
        else:
            window = returns_series[max(0, i-30):i]
            X_ar, y_ar = window[:-1].reshape(-1, 1), window[1:]
            if len(X_ar) > 5 and not np.isnan(X_ar).any() and not np.isnan(y_ar).any():
                reg = Ridge(alpha=1.0).fit(X_ar, y_ar)
                ar_preds.append(reg.predict(window[-1].reshape(1, -1))[0])
            else: ar_preds.append(0.0)
                
    df['AR1_Forecast'] = ar_preds
    df['Target_Direction'] = (df['Forward_Return'] > 0).astype(int)
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    return df.dropna().reset_index(drop=True)

@app.get("/")
def home():
    return {"status": "MLFP Quant Engine API is online."}

@app.get("/api/scanner")
def get_market_scanner():
    results = []
    for ticker in UPSTOX_KEYS.keys():
        df = fetch_upstox_data(ticker, years=1)
        if df.empty or len(df) < 30: continue
        
        df['Rolling_Vol'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0).rolling(10).std()
        sma20 = df['ClosePrice'].rolling(20).mean().iloc[-1]
        curr_close = df['ClosePrice'].iloc[-1]
        
        avg_vol = df['Rolling_Vol'].mean()
        curr_vol = df['Rolling_Vol'].iloc[-1]
        
        regime = "Low Volatility" if curr_vol < avg_vol else "High Volatility"
        signal = "BULLISH" if curr_close > sma20 else "BEARISH"
        
        if regime == "Low Volatility" and signal == "BULLISH": status = "green"
        elif regime == "Low Volatility" and signal == "BEARISH": status = "red"
        else: status = "yellow"
            
        results.append({
            "ticker": ticker,
            "regime": regime,
            "signal": signal,
            "price": round(curr_close, 2),
            "status": status
        })
    return {"scanner": results}

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str, friction: float = Query(0.0015), neutral_band: float = Query(0.05)):
    ticker = ticker.upper()
    if ticker not in UPSTOX_KEYS:
        raise HTTPException(status_code=400, detail="Ticker not supported.")
    
    raw_data = fetch_upstox_data(ticker)
    
    # Safe error catching: Changed from 500 to 400 to force CORS delivery
    if raw_data.empty: 
        raise HTTPException(status_code=400, detail=f"Upstox API failed to return data for {ticker}. The data source might be timing out.")
        
    master_df = generate_hybrid_features(raw_data)

    log_vol = np.log(master_df[['Rolling_Vol']] + 1e-8)
    scaled_vol = StandardScaler().fit_transform(log_vol)
    master_df['Volatility_Regime'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(scaled_vol)
    regime_means = master_df.groupby('Volatility_Regime')['Rolling_Vol'].mean()
    master_df['Regime_Label'] = master_df['Volatility_Regime'].apply(
        lambda x: 'Low Volatility' if x == regime_means.idxmin() else 'High Volatility'
    )

    feature_cols = ['Log_Returns', 'SMA_20_Dist', 'RSI_14', 'Relative_Volume', 'Log_Returns_Lag1', 'Log_Returns_Lag2', 'AR1_Forecast', 'MACD', 'MACD_Hist']
    X, y = master_df[feature_cols], master_df['Target_Direction']
    
    models = {
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
    }
    
    results = {name: {'auc': []} for name in models.keys()}
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            try: results[name]['auc'].append(roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
            except: results[name]['auc'].append(0.5)

    best_model_name = max(results, key=lambda k: np.mean(results[k]['auc']))
    
    oos_preds, oos_indices = [], []
    current_pos = 0  
    
    for train_idx, test_idx in tscv.split(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
        X_test_scaled = scaler.transform(X.iloc[test_idx])
        model = models[best_model_name].fit(X_train_scaled, y.iloc[train_idx])
        
        probs = model.predict_proba(X_test_scaled)[:, 1]
        for p in probs:
            if p > 0.55: current_pos = 1
            elif p < 0.45: current_pos = 0
            oos_preds.append(current_pos)
            
        oos_indices.extend(test_idx)

    res_df = master_df.iloc[oos_indices].copy()
    res_df['OOS_Pred'] = oos_preds
    
    res_df['Friction_Unfilt'] = (res_df['OOS_Pred'].diff().abs().fillna(0)) * friction
    res_df['Ret_Unfilt'] = (res_df['OOS_Pred'] * res_df['Forward_Return']) - res_df['Friction_Unfilt']
    
    res_df['Pos_Filt'] = np.where(res_df['Regime_Label'] == 'Low Volatility', res_df['OOS_Pred'], 0)
    res_df['Friction_Filt'] = (res_df['Pos_Filt'].diff().abs().fillna(0)) * friction
    res_df['Ret_Filt'] = (res_df['Pos_Filt'] * res_df['Forward_Return']) - res_df['Friction_Filt']

    clean_df = res_df.dropna(subset=['Ret_Unfilt', 'Ret_Filt', 'Forward_Return'])
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    live_model = models[best_model_name].fit(X_scaled, y)
    prob_up = live_model.predict_proba(final_scaler.transform(master_df[feature_cols].iloc[[-1]]))[0][1]

    upper_bound = 0.5 + neutral_band
    lower_bound = 0.5 - neutral_band
    if prob_up >= upper_bound: quant_signal = "BULLISH"
    elif prob_up <= lower_bound: quant_signal = "BEARISH"
    else: quant_signal = "NEUTRAL"

    ai_score, ai_summary = 0.0, "AI Sentiment Feed Unavailable"
    try:
        q = urllib.parse.quote(f"{ticker} stock news India")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        rss_resp = requests.get(rss_url, headers=headers, timeout=3)
        feed = feedparser.parse(rss_resp.content)
        
        headlines = "\n".join([f"- {h.title}" for h in feed.entries[:10]])
        if headlines.strip():
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"Analyze these recent news headlines for '{ticker}':\n{headlines}\nReturn ONLY a valid JSON: {{\"sentiment_score\": <float -1.0 to 1.0>, \"executive_summary\": \"<1 sentence>\"}}"
            
            resp = client.models.generate_content(
                model='gemini-3.5-flash', 
                contents=prompt, 
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            match = re.search(r'\{.*\}', resp.text, re.DOTALL)
            ai_json = json.loads(match.group(0)) if match else json.loads(resp.text)
            ai_score = ai_json.get('sentiment_score', 0)
            ai_summary = ai_json.get('executive_summary', "No summary provided.")
    except Exception as e:
        ai_summary = f"API Diagnostic: News parser failed. ({str(e)})"

    return {
        "ticker": ticker,
        "current_regime": master_df['Regime_Label'].iloc[-1],
        "best_model": best_model_name,
        "quant_signal": quant_signal,
        "quant_probability": round(prob_up * 100, 2),
        "ai_sentiment_score": ai_score,
        "ai_summary": ai_summary,
        "diagnostics": {
            "unfiltered_trades": int(clean_df['OOS_Pred'].diff().abs().sum()),
            "filtered_trades": int(clean_df['Pos_Filt'].diff().abs().sum()),
            "unfiltered_friction_pct": round(clean_df['Friction_Unfilt'].sum() * 100, 2),
            "filtered_friction_pct": round(clean_df['Friction_Filt'].sum() * 100, 2)
        },
        "performance": {
            "buy_and_hold_pct": round((np.exp(clean_df['Forward_Return'].sum()) - 1) * 100, 2),
            "unfiltered_strat_pct": round((np.exp(clean_df['Ret_Unfilt'].sum()) - 1) * 100, 2),
            "filtered_strat_pct": round((np.exp(clean_df['Ret_Filt'].sum()) - 1) * 100, 2)
        },
        "chart_data": {
            "labels": [str(d.date()) for d in clean_df['Date']],
            "buy_hold": [round(x, 4) for x in list(np.exp(clean_df['Forward_Return'].cumsum()) - 1)],
            "unfiltered": [round(x, 4) for x in list(np.exp(clean_df['Ret_Unfilt'].cumsum()) - 1)],
            "filtered": [round(x, 4) for x in list(np.exp(clean_df['Ret_Filt'].cumsum()) - 1)]
        }
    }
