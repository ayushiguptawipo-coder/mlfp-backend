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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from google import genai
from google.genai import types
from datetime import datetime, timedelta

# SECURE ENVIRONMENT VARIABLES
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN")

app = FastAPI(title="MLFP Quant Engine Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def fetch_live_quote(instrument_keys_list):
    if not instrument_keys_list: return {}
    # FIX: Encode individual keys but leave the comma intact so Upstox doesn't reject the request
    keys_param = ",".join([urllib.parse.quote(k) for k in instrument_keys_list])
    url = f'https://api.upstox.com/v2/market-quote/quotes?instrument_key={keys_param}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            return res.json().get('data', {})
    except Exception:
        pass
    return {}

def fetch_upstox_data_dynamic(instrument_key, years=3):
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    safe_key = urllib.parse.quote(instrument_key)
    url = f'https://api.upstox.com/v2/historical-candle/{safe_key}/day/{to_date}/{from_date}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    
    try:
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json().get('data', {}).get('candles', [])
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume', 'OI'])
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            return df.sort_values('Date').reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume']]
    except Exception:
        pass
    return pd.DataFrame()

def generate_hybrid_features(df):
    df = df.copy()
    df['Log_Returns'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Forward_Return'] = np.log(df['ClosePrice'].shift(-1) / df['ClosePrice'])
    
    sma_200 = df['ClosePrice'].rolling(window=200, min_periods=50).mean()
    df['Macro_Bull_Trend'] = (df['ClosePrice'] > sma_200).astype(int)
    
    sma_20 = df['ClosePrice'].rolling(window=20).mean()
    df['SMA_20_Dist'] = (df['ClosePrice'] - sma_20) / sma_20
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    delta = df['ClosePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))
    
    ema_12 = df['ClosePrice'].ewm(span=12, adjust=False).mean()
    ema_26 = df['ClosePrice'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['ClosePrice'].shift()).abs()
    low_close = (df['Low'] - df['ClosePrice'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean() / df['ClosePrice']
    
    for col in ['Log_Returns', 'RSI_14', 'MACD_Hist']:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2)
        
    ar_preds, returns_series = [], df['Log_Returns'].values
    for i in range(len(df)):
        if i < 20: ar_preds.append(np.nan)
        else:
            window = returns_series[max(0, i-30):i]
            X_ar, y_ar = window[:-1].reshape(-1, 1), window[1:]
            if len(X_ar) > 5 and not np.isnan(X_ar).any() and not np.isnan(y_ar).any():
                reg = Ridge(alpha=2.0).fit(X_ar, y_ar)
                ar_preds.append(reg.predict(window[-1].reshape(1, -1))[0])
            else: ar_preds.append(0.0)
                
    df['AR1_Forecast'] = ar_preds
    df['Target_Direction'] = (df['Forward_Return'] > 0).astype(int)
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    return df.dropna().reset_index(drop=True)

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "MLFP Quant Engine Pro API is online."}

@app.get("/api/search")
def search_stock(q: str):
    if not q or len(q) < 2: return {"results": []}
    url = f'https://api.upstox.com/v2/instruments/search?query={urllib.parse.quote(q)}&segments=EQ'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    try:
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json().get('data', [])
            results = []
            for item in data:
                if item.get('segment') in ['NSE_EQ', 'BSE_EQ']:
                    results.append({
                        "ticker": item.get('trading_symbol'),
                        "name": item.get('name'),
                        "instrument_key": item.get('instrument_key')
                    })
            unique_results = {}
            for r in results:
                if r['ticker'] not in unique_results:
                    unique_results[r['ticker']] = r
            return {"results": list(unique_results.values())[:8]}
    except Exception:
        pass
    return {"results": []}

@app.get("/api/scanner")
def get_market_scanner():
    keys = list(UPSTOX_KEYS.values())
    quotes = fetch_live_quote(keys)
    results = []
    
    for ticker, instr_key in UPSTOX_KEYS.items():
        df = fetch_upstox_data_dynamic(instr_key, years=1)
        if df.empty or len(df) < 30: continue
        
        df['Rolling_Vol'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0).rolling(10).std()
        sma20 = df['ClosePrice'].rolling(20).mean().iloc[-1]
        
        # FIX: Bullet-proof dictionary extraction. If the ISIN code matches anywhere, grab it!
        q_data = {}
        if quotes:
            for k, v in quotes.items():
                if instr_key.split('|')[-1] in k:
                    q_data = v
                    break
                    
        live_price = q_data.get('last_price', df['ClosePrice'].iloc[-1])
        change_pct = q_data.get('net_change', 0.0)
        
        avg_vol = df['Rolling_Vol'].mean()
        curr_vol = df['Rolling_Vol'].iloc[-1]
        
        regime = "Low Volatility" if curr_vol < avg_vol else "High Volatility"
        signal = "BULLISH" if live_price > sma20 else "BEARISH"
        
        if regime == "Low Volatility" and signal == "BULLISH": status = "green"
        elif regime == "Low Volatility" and signal == "BEARISH": status = "red"
        else: status = "yellow"
            
        results.append({
            "ticker": str(ticker),
            "regime": str(regime),
            "signal": str(signal),
            "price": float(round(live_price, 2)),
            "change_pct": float(round(change_pct, 2)),
            "status": str(status)
        })
    return {"scanner": results}

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str, instrument_key: str = Query(None), friction: float = Query(0.0015), neutral_band: float = Query(0.05)):
    ticker = ticker.upper()
    if not instrument_key:
        instrument_key = UPSTOX_KEYS.get(ticker)
        
    if not instrument_key:
        raise HTTPException(status_code=400, detail="Instrument Key missing. Please select from search.")
        
    raw_data = fetch_upstox_data_dynamic(instrument_key, years=3)
    if raw_data.empty: 
        raise HTTPException(status_code=400, detail=f"Data feed timed out or returned empty for {ticker}.")
        
    # FIX: Guarantee we get the data by just pulling the first value, bypassing any Upstox key formatting mismatch
    live_quotes = fetch_live_quote([instrument_key])
    quote_info = {}
    if live_quotes:
        quote_info = list(live_quotes.values())[0]
        
    current_live_price = float(quote_info.get('last_price', raw_data['ClosePrice'].iloc[-1]))
    current_day_change = float(quote_info.get('net_change', 0.0))
    current_day_high = float(quote_info.get('ohlc', {}).get('high', raw_data['High'].iloc[-1]))
    current_day_low = float(quote_info.get('ohlc', {}).get('low', raw_data['Low'].iloc[-1]))

    master_df = generate_hybrid_features(raw_data)

    log_vol = np.log(master_df[['Rolling_Vol']] + 1e-8)
    scaled_vol = StandardScaler().fit_transform(log_vol)
    master_df['Volatility_Regime'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(scaled_vol)
    regime_means = master_df.groupby('Volatility_Regime')['Rolling_Vol'].mean()
    master_df['Regime_Label'] = master_df['Volatility_Regime'].apply(
        lambda x: 'Low Volatility' if x == regime_means.idxmin() else 'High Volatility'
    )

    feature_cols = [
        'Log_Returns', 'SMA_20_Dist', 'RSI_14', 'Relative_Volume', 
        'Log_Returns_Lag1', 'Log_Returns_Lag2', 'AR1_Forecast', 
        'MACD', 'MACD_Hist', 'ATR_14'
    ]
    X, y = master_df[feature_cols], master_df['Target_Direction']
    
    models = {
        "CatBoost": CatBoostClassifier(depth=3, iterations=60, learning_rate=0.03, l2_leaf_reg=3.0, verbose=0, random_seed=42),
        "LightGBM": LGBMClassifier(max_depth=3, n_estimators=50, learning_rate=0.03, subsample=0.8, reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42),
        "XGBoost": XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, random_state=42, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=8, random_state=42),
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=500, random_state=42)
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
            except Exception: results[name]['auc'].append(0.5)

    best_model_name = max(results, key=lambda k: np.mean(results[k]['auc']))
    
    oos_positions, oos_indices = [], []
    for train_idx, test_idx in tscv.split(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
        X_test_scaled = scaler.transform(X.iloc[test_idx])
        model = models[best_model_name].fit(X_train_scaled, y.iloc[train_idx])
        
        probs = model.predict_proba(X_test_scaled)[:, 1]
        macro_trends = master_df['Macro_Bull_Trend'].iloc[test_idx].values
        
        for p, is_bull_trend in zip(probs, macro_trends):
            if p > (0.50 + neutral_band):
                conviction = (p - 0.50) / 0.50
                target_alloc = 0.30 if conviction < 0.25 else (0.65 if conviction < 0.50 else 1.00)
                oos_positions.append(target_alloc)
            elif p < (0.50 - neutral_band):
                if is_bull_trend == 1:
                    oos_positions.append(0.0)
                else:
                    conviction = (0.50 - p) / 0.50
                    target_alloc = 0.30 if conviction < 0.25 else (0.65 if conviction < 0.50 else 1.00)
                    oos_positions.append(-target_alloc)
            else:
                oos_positions.append(0.0)
        oos_indices.extend(test_idx)

    res_df = master_df.iloc[oos_indices].copy()
    res_df['Position_Unfilt'] = oos_positions
    res_df['Friction_Unfilt'] = (res_df['Position_Unfilt'].diff().abs().fillna(0)) * friction
    res_df['Ret_Unfilt'] = (res_df['Position_Unfilt'] * res_df['Forward_Return']) - res_df['Friction_Unfilt']
    
    res_df['Position_Filt'] = np.where(res_df['Regime_Label'] == 'Low Volatility', res_df['Position_Unfilt'], 0.0)
    res_df['Friction_Filt'] = (res_df['Position_Filt'].diff().abs().fillna(0)) * friction
    res_df['Ret_Filt'] = (res_df['Position_Filt'] * res_df['Forward_Return']) - res_df['Friction_Filt']

    clean_df = res_df.dropna(subset=['Ret_Unfilt', 'Ret_Filt', 'Forward_Return'])
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    live_model = models[best_model_name].fit(X_scaled, y)
    prob_up = live_model.predict_proba(final_scaler.transform(master_df[feature_cols].iloc[[-1]]))[0][1]

    upper_bound = 0.5 + neutral_band
    lower_bound = 0.5 - neutral_band
    is_macro_bull = master_df['Macro_Bull_Trend'].iloc[-1] == 1
    
    if prob_up >= upper_bound:
        quant_signal = "BULLISH"
    elif prob_up <= lower_bound:
        quant_signal = "NEUTRAL (Macro Bull Guard)" if is_macro_bull else "BEARISH"
    else:
        quant_signal = "NEUTRAL"

    # Gemini News Sentiment Engine
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
            prompt = f"Analyze these recent news headlines for '{ticker}':\n{headlines}\nReturn ONLY a valid JSON: {{\"sentiment_score\": <float -1.0 to 1.0>, \"executive_summary\": \"<1 sentence summary without any double quotes inside>\"}}"
            
            resp = client.models.generate_content(
                model='gemini-3.5-flash', 
                contents=prompt, 
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw_text = resp.text.strip()
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            clean_json = match.group(0) if match else raw_text
            
            try: ai_json = json.loads(clean_json)
            except Exception: ai_json = {"sentiment_score": 0.0, "executive_summary": clean_json.replace('"', "'").replace('\n', ' ')}
                
            ai_score = ai_json.get('sentiment_score', 0.0)
            ai_summary = ai_json.get('executive_summary', "No summary provided.")
    except Exception as e:
        ai_summary = f"News parser diagnostic: {str(e)}"

    # Candlestick Payload Construction
    recent_candles_df = raw_data.tail(100)
    candle_list = []
    for _, row in recent_candles_df.iterrows():
        candle_list.append({
            "time": str(row['Date'].date()),
            "open": float(round(row['Open'], 2)),
            "high": float(round(row['High'], 2)),
            "low": float(round(row['Low'], 2)),
            "close": float(round(row['ClosePrice'], 2)),
            "volume": float(row['Volume'])
        })

    # FIX 2: Correct Live Candle Stitching for Today
    today_str = datetime.now().strftime('%Y-%m-%d')
    if quote_info and candle_list[-1]["time"] != today_str:
        ohlc = quote_info.get('ohlc', {})
        candle_list.append({
            "time": today_str,
            "open": float(round(ohlc.get('open', current_live_price), 2)),
            "high": float(round(ohlc.get('high', current_live_price), 2)),
            "low": float(round(ohlc.get('low', current_live_price), 2)),
            "close": float(round(current_live_price, 2)),
            "volume": float(quote_info.get('volume', 0.0))
        })

    return {
        "ticker": str(ticker),
        "live_price": float(round(current_live_price, 2)),
        "day_change_pct": float(round(current_day_change, 2)),
        "day_high": float(round(current_day_high, 2)),
        "day_low": float(round(current_day_low, 2)),
        "current_regime": str(master_df['Regime_Label'].iloc[-1]),
        "best_model": str(best_model_name),
        "quant_signal": str(quant_signal),
        "quant_probability": float(round(float(prob_up) * 100, 2)),
        "ai_sentiment_score": float(ai_score),
        "ai_summary": str(ai_summary),
        "diagnostics": {
            "unfiltered_trades": int(clean_df['Position_Unfilt'].diff().abs().gt(0).sum()),
            "filtered_trades": int(clean_df['Position_Filt'].diff().abs().gt(0).sum()),
            "unfiltered_friction_pct": float(round(float(clean_df['Friction_Unfilt'].sum()) * 100, 2)),
            "filtered_friction_pct": float(round(float(clean_df['Friction_Filt'].sum()) * 100, 2))
        },
        "performance": {
            "buy_and_hold_pct": float(round((np.exp(float(clean_df['Forward_Return'].sum())) - 1) * 100, 2)),
            "unfiltered_strat_pct": float(round((np.exp(float(clean_df['Ret_Unfilt'].sum())) - 1) * 100, 2)),
            "filtered_strat_pct": float(round((np.exp(float(clean_df['Ret_Filt'].sum())) - 1) * 100, 2))
        },
        "chart_data": {
            "labels": [str(d.date()) for d in clean_df['Date']],
            "buy_hold": [float(round(x, 4)) for x in (np.exp(clean_df['Forward_Return'].cumsum()) - 1).tolist()],
            "unfiltered": [float(round(x, 4)) for x in (np.exp(clean_df['Ret_Unfilt'].cumsum()) - 1).tolist()],
            "filtered": [float(round(x, 4)) for x in (np.exp(clean_df['Ret_Filt'].cumsum()) - 1).tolist()]
        },
        "candles": candle_list
    }
