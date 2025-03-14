import os
import time
import json
import simpleaudio as sa
import hmac
import hashlib
import base64
import urllib.parse
import asyncio
import logging
import requests
import websockets
from skyfield.api import load
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import krakenex
import pytz
from pytz import timezone
import matplotlib.pyplot as plt

DEBUG = True

# ===============================
# PART 1: HISTORICAL MODEL TRAINING WITH ASTROLOGICAL DATA
# ===============================
MODEL_PATH = "ai_trade_model_with_astro.pkl"
SCALER_PATH = "scaler_crypto.pkl"
HIST_WINDOW_MINUTES = 1

TRADING_PAIRS = [
    "ETH/USD", "ADA/USD", "SOL/USD", "LTC/USD", "DOT/USD", "XLM/USD", "TAO/USD",
    "XRP/USD", "NANO/USD", "RUNE/USD", "SUI/USD", "XDT/USD", "NANO/USD", "XBT/USD","XDG/USD","KAITO/USD" ,"POPCAT/USD", "COTI/USD"
]

# Load planetary data
ephemeris = load('de421.bsp')
planets = {
    'Sun': ephemeris['sun'],
    'Moon': ephemeris['moon'],
    'Mercury': ephemeris['mercury'],
    'Venus': ephemeris['venus'],
    'Mars': ephemeris['mars'],
    'Jupiter': ephemeris['jupiter barycenter'],
    'Saturn': ephemeris['saturn barycenter'],
    'Uranus': ephemeris['uranus barycenter'],
    'Neptune': ephemeris['neptune barycenter'],
    'Pluto': ephemeris['pluto barycenter']
}
earth = ephemeris['earth']



def calculate_lunar_midpoints():
    """Calculates the lunar phase midpoint, Moon's zodiac midpoint, and encodes its RA with sine/cosine."""
    ts = load.timescale()
    now = datetime.utcnow()
    t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)

    ephemeris = load('de421.bsp')
    earth, moon, sun = ephemeris['earth'], ephemeris['moon'], ephemeris['sun']

    e = earth.at(t)
    moon_pos = e.observe(moon).apparent()
    sun_pos = e.observe(sun).apparent()

    # Calculate the moon's phase angle using the sun's ICRF position
    moon_phase_angle = moon_pos.phase_angle(sun)
    phase_degrees = moon_phase_angle.degrees

    # Map lunar phase to cycle with predetermined midpoints
    if phase_degrees < 90:
        lunar_cycle_stage = "First Half"
        midpoint = 90
    elif phase_degrees < 180:
        lunar_cycle_stage = "First Quarter Midpoint"
        midpoint = 135
    elif phase_degrees < 270:
        lunar_cycle_stage = "Second Half"
        midpoint = 270
    else:
        lunar_cycle_stage = "Last Quarter Midpoint"
        midpoint = 315

    # Calculate Moon's position in Zodiac (RA)
    ra, dec, distance = moon_pos.radec()
    moon_ra_hours = ra.hours
    moon_ra_degrees = (moon_ra_hours / 24) * 360
    moon_zodiac_degree = moon_ra_degrees % 30
    moon_sign_midpoint = 15 if moon_zodiac_degree < 15 else 30

    # Encode Moon's right ascension cyclically using sine and cosine
    moon_ra_radians = (moon_ra_hours / 24) * 2 * np.pi
    moon_ra_sin = np.sin(moon_ra_radians)
    moon_ra_cos = np.cos(moon_ra_radians)

    return {
        "lunar_phase_midpoint": midpoint,
        "moon_zodiac_midpoint": moon_sign_midpoint,
        "lunar_cycle_stage": lunar_cycle_stage,
        "moon_ra_sin": moon_ra_sin,
        "moon_ra_cos": moon_ra_cos
    }
 



#  --- NEW: Minute Cycle Encoding Function ---
def encode_minute_cycle(dt, period=60):
    """
    Encodes the current minute-of-day in a cyclical fashion.
    period: period in minutes (e.g. 5 for a 5‑minute cycle).
    """
    minute_of_day = dt.hour * 60 + dt.minute
    cycle_length = 1440 // period  # total number of cycles in a day
    cycle = (minute_of_day // period) % cycle_length
    radians = (cycle / cycle_length) * 2 * np.pi
    return np.sin(radians), np.cos(radians)

def encode_planetary_hour(hour_numeric):
    """Encodes a planetary hour (numeric 0-6) into cyclic sine and cosine components."""
    # We keep the original period based on 24 hours.
    hour_rad = (hour_numeric / 24) * 2 * np.pi
    return np.sin(hour_rad), np.cos(hour_rad)

def simulate_trade_durations(df, target_percent=0.00777, max_lookahead=60):
    df = df.sort_index()
    results = []
    for symbol, group in df.groupby('symbol'):
        group = group.sort_index()
        times = group.index.tolist()
        closes = group['close'].tolist() 
        highs = group['high'].tolist()
        lows = group['low'].tolist()
        exit_times = []
        durations = []
        quick_labels = []
        n = len(group)
        for i in range(n):
            entry_time = times[i]
            entry_price = closes[i]
            exit_time = None
            for j in range(i+1, n):
                if (times[j] - entry_time).total_seconds() / 60 > max_lookahead:
                    break
                if highs[j] >= entry_price * (1 + target_percent):
                    exit_time = times[j]
                    break
                elif lows[j] <= entry_price * (1 - target_percent):
                    exit_time = times[j]
                    break
            if exit_time is None:
                exit_times.append(pd.NaT)
                durations.append(np.nan)
                quick_labels.append(0)
            else:
                duration = (exit_time - entry_time).total_seconds() / 60.0
                exit_times.append(exit_time)
                durations.append(duration)
                quick_labels.append(1 if duration <= max_lookahead else 0)
        group['exit_time'] = exit_times
        group['trade_duration'] = durations
        group['quick_trade'] = quick_labels
        results.append(group)
    return pd.concat(results)

def fetch_kraken_historical_data(pair, interval, since):
    uri = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since}"
    try:
        response = requests.get(uri)
        response.raise_for_status()
        data = response.json()
        result = data.get("result")
        if not result:
            return pd.DataFrame()
        for key in result:
            if key != "last":
                ohlc_key = key
                break
        ohlc_data = result.get(ohlc_key, [])
        records = []
        for row in ohlc_data:
            timestamp = datetime.fromtimestamp(float(row[0]))
            records.append({
                'timestamp': timestamp,
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[6]),
                'symbol': pair
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching historical data for {pair}: {e}")
        return pd.DataFrame()

def fetch_astrological_data():
    ts = load.timescale()
    now = datetime.utcnow()
    t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
    positions = {}
    for planet_name, planet in planets.items():
        astrometric = earth.at(t).observe(planet).apparent()
        ra, dec, distance = astrometric.radec()
        ra_radians = (ra.hours / 24) * 2 * np.pi
        positions[planet_name] = {
            'ra_hours': ra.hours,
            'dec_degrees': dec.degrees,
            'distance_au': distance.au,
            'ra_sin': np.sin(ra_radians),
            'ra_cos': np.cos(ra_radians)
        }
    return positions

def get_current_planetary_hour_utc():
    """Returns the current planetary hour as a string (e.g., 'Sun')."""
    utc_now = datetime.now(pytz.utc)
    planetary_sequence = ['Saturn', 'Jupiter', 'Mars', 'Sun', 'Venus', 'Mercury', 'Moon']
    weekday_rulers = ['Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Sun']
    weekday = utc_now.weekday()  # Monday=0, Sunday=6
    first_hour_ruler = weekday_rulers[weekday]
    first_hour_index = planetary_sequence.index(first_hour_ruler)
    full_sequence = [planetary_sequence[(first_hour_index + i) % 7] for i in range(24)]
    sunrise_utc = utc_now.replace(hour=6, minute=0, second=0, microsecond=0)
    sunset_utc = utc_now.replace(hour=18, minute=0, second=0, microsecond=0)
    if sunrise_utc <= utc_now < sunset_utc:
        planetary_hour_length = (sunset_utc - sunrise_utc).total_seconds() / 12
        elapsed_seconds = (utc_now - sunrise_utc).total_seconds()
        hour_number = int(elapsed_seconds // planetary_hour_length)
    else:
        if utc_now >= sunset_utc:
            next_sunrise_utc = sunrise_utc + timedelta(days=1)
            planetary_hour_length = (next_sunrise_utc - sunset_utc).total_seconds() / 12
            elapsed_seconds = (utc_now - sunset_utc).total_seconds()
            hour_number = int(elapsed_seconds // planetary_hour_length) + 12
        else:  # before sunrise
            previous_sunset_utc = sunset_utc - timedelta(days=1)
            planetary_hour_length = (sunrise_utc - previous_sunset_utc).total_seconds() / 12
            elapsed_seconds = (utc_now - previous_sunset_utc).total_seconds()
            hour_number = int(elapsed_seconds // planetary_hour_length) + 12
    return full_sequence[hour_number % 24]

def apply_indicators(df):
    df['3_EMA'] = df['close'].ewm(span=3, adjust=False).mean()
    df['5_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['8_EMA'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_comparison'] = (df['3_EMA'] > df['8_EMA']).astype(int)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=50, min_periods=50).mean()
    avg_loss = loss.rolling(window=50, min_periods=50).mean().replace(0, 1e-8)
    df['rsi_5'] = 100 - (100 / (1 + avg_gain / avg_loss))
    df['ATR'] = (df['high'] - df['low']).rolling(window=50, min_periods=50).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['volume_ma'] = df['volume'].rolling(window=50).mean()
    df['volume_spike'] = (df['volume'] > df['volume_ma']).astype(int)
    df['stoch_rsi'] = (df['rsi_5'] - df['rsi_5'].rolling(window=50).min()) / (
    df['rsi_5'].rolling(window=50).max() - df['rsi_5'].rolling(window=50).min())
    df['close_minus_open'] = df['close'] - df['open']
    df['high_minus_low'] = df['high'] - df['low']
    df['close_pct_change'] = df['close'].pct_change().fillna(0)
    df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
    df['obv'] = ((df['close'] - df['close'].shift(1)) * df['volume']).cumsum()
    df['obv_momentum_shift'] = df['obv'].diff().fillna(0)
    df['heikin_ashi_trend'] = ((df['close'] + df['open']) / 2).diff().fillna(0)
    df['100_EMA'] = df['close'].ewm(span=50, adjust=False).mean()
    df['trend_filter'] = (df['close'] > df['100_EMA']).astype(int)
    df['SMA_100'] = df['close'].rolling(window=50).mean()
    df['SMA_15'] = df['close'].rolling(window=5).mean()
    return df.fillna(0)

def get_planetary_hour_from_timestamp(ts):
    utc_now = ts.replace(tzinfo=pytz.utc)
    planetary_sequence = ['Saturn', 'Jupiter', 'Mars', 'Sun', 'Venus', 'Mercury', 'Moon']
    weekday_rulers = ['Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Sun']
    weekday = utc_now.weekday()
    first_hour_ruler = weekday_rulers[weekday]
    first_hour_index = planetary_sequence.index(first_hour_ruler)
    full_sequence = [planetary_sequence[(first_hour_index + i) % 7] for i in range(24)]
    sunrise_utc = utc_now.replace(hour=6, minute=0, second=0, microsecond=0)
    sunset_utc = utc_now.replace(hour=18, minute=0, second=0, microsecond=0)
    if sunrise_utc <= utc_now < sunset_utc:
        planetary_hour_length = (sunset_utc - sunrise_utc).total_seconds() / 12
        elapsed_seconds = (utc_now - sunrise_utc).total_seconds()
        hour_number = int(elapsed_seconds // planetary_hour_length)
    else:
        if utc_now >= sunset_utc:
            next_sunrise_utc = sunrise_utc + timedelta(days=1)
            planetary_hour_length = (next_sunrise_utc - sunset_utc).total_seconds() / 12
            elapsed_seconds = (utc_now - sunset_utc).total_seconds()
            hour_number = int(elapsed_seconds // planetary_hour_length) + 12
        else:
            previous_sunset_utc = sunset_utc - timedelta(days=1)
            planetary_hour_length = (sunrise_utc - previous_sunset_utc).total_seconds() / 12
            elapsed_seconds = (utc_now - previous_sunset_utc).total_seconds()
            hour_number = int(elapsed_seconds // planetary_hour_length) + 12
    return full_sequence[hour_number % 24]

def convert_planetary_hour_to_numeric(hour_str):
    mapping = {'Saturn': 0, 'Jupiter': 1, 'Mars': 2, 'Sun': 3, 'Venus': 4, 'Mercury': 5, 'Moon': 6}
    return mapping.get(hour_str, -1)

def integrate_astro_data(df):
    astro_data = fetch_astrological_data()
    for planet, values in astro_data.items():
        df[f"{planet}_ra_hours"] = values["ra_hours"]
        df[f"{planet}_dec_degrees"] = values["dec_degrees"]
        df[f"{planet}_distance_au"] = values["distance_au"]
        df[f"{planet}_ra_sin"] = values["ra_sin"]
        df[f"{planet}_ra_cos"] = values["ra_cos"]
    # Compute the current planetary hour and encode it cyclically
    planet_hour_str = get_current_planetary_hour_utc()
    planet_hour = convert_planetary_hour_to_numeric(planet_hour_str)
    df['planetary_hour_sin'], df['planetary_hour_cos'] = encode_planetary_hour(planet_hour)
    return df

def load_training_data(pairs, interval=1):
    dfs = []
    since = int((datetime.now().replace(tzinfo=pytz.utc) - timedelta(hours=HIST_WINDOW_MINUTES)).timestamp())
    for pair in pairs:
        df_pair = fetch_kraken_historical_data(pair, interval, since)
        if not df_pair.empty:
            df_pair = apply_indicators(df_pair)
            df_pair = integrate_astro_data(df_pair)
            # Also add a minute-cycle feature (with a 5-minute period) to capture scalping cycles
            df_pair['minute_cycle_sin'], df_pair['minute_cycle_cos'] = zip(*df_pair['timestamp'].apply(lambda ts: encode_minute_cycle(ts, period=5)))
            # Also compute the traditional planetary hour numeric for combined model use
            df_pair['planetary_hour'] = df_pair['timestamp'].apply(lambda ts: convert_planetary_hour_to_numeric(get_planetary_hour_from_timestamp(ts)))
            dfs.append(df_pair)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.index = pd.to_datetime(df_all.index)
        df_all = simulate_trade_durations(df_all, target_percent=0.0076, max_lookahead=60)
        return df_all
    else:
        return pd.DataFrame()

# --- New separate model training functions for comparison ---

def train_astro_model_on_pairs(pairs):
    """Train a model using only astro features."""
    df = load_training_data(pairs, interval=1)
    if df.empty:
        return None
    df.set_index("timestamp", inplace=True)
    astro_features = [
        "Sun_ra_hours", "Sun_dec_degrees", "Sun_distance_au", "Sun_ra_sin", "Sun_ra_cos",
        "Moon_ra_hours", "Moon_dec_degrees", "Moon_distance_au", "Moon_ra_sin", "Moon_ra_cos",
        "Mercury_ra_hours", "Mercury_ra_sin", "Mercury_ra_cos",
        "Venus_ra_hours", "Venus_ra_sin", "Venus_ra_cos",
        "Mars_ra_hours", "Mars_ra_sin", "Mars_ra_cos",
        "Jupiter_ra_hours", "Jupiter_ra_sin", "Jupiter_ra_cos",
        "Saturn_ra_hours", "Saturn_ra_sin", "Saturn_ra_cos",
        "Uranus_ra_hours", "Uranus_ra_sin", "Uranus_ra_cos",
        "Neptune_ra_hours", "Neptune_ra_sin", "Neptune_ra_cos",
        "Pluto_ra_hours", "Pluto_ra_sin", "Pluto_ra_cos",
        "planetary_hour_sin", "planetary_hour_cos",
        "minute_cycle_sin", "minute_cycle_cos"
    ]
    X = df[astro_features]
    y = df['quick_trade']
    X = X.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler_astro.pkl")
    smote = SMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.35, random_state=42)
    model = XGBClassifier(n_estimators=100, learning_rate=0.5, max_depth=6,
                          use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    joblib.dump(model, "astro_model.pkl")
    y_pred = model.predict(X_test)
    print("Astro-Only Model Performance:")
    print(classification_report(y_test, y_pred))
    return model

def train_tech_model_on_pairs(pairs):
    """Train a model using only technical indicators."""
    df = load_training_data(pairs, interval=1)
    if df.empty:
        return None
    df.set_index("timestamp", inplace=True)
    tech_features = [
        "open", "high", "low", "close", "volume", "vwap",
        "3_EMA", "5_EMA", "8_EMA", "rsi_5", "ATR", "ema_comparison",
        "volume_ma", "volume_spike", "trend_filter",
        
    ]
    X = df[tech_features]
    y = df['quick_trade']
    X = X.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler_tech.pkl")
    smote = SMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.35, random_state=42)
    model = XGBClassifier(n_estimators=100, learning_rate=0.5, max_depth=6,
                          use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    joblib.dump(model, "tech_model.pkl")
    y_pred = model.predict(X_test)
    print("Technical-Only Model Performance:")
    print(classification_report(y_test, y_pred))
    return model

def train_model_on_pairs(pairs):
    """Train a combined model using both technical and astro features (including minute cycle)."""
    df = load_training_data(pairs, interval=1)
    if df.empty:
        return None
    df.set_index("timestamp", inplace=True)
    features = [
        "open", "high", "low", "close", "volume", "vwap",
        "3_EMA", "5_EMA", "8_EMA", "rsi_5", "ATR", "ema_comparison",
        "volume_ma", "volume_spike", "trend_filter",
        "Sun_ra_hours", "Sun_dec_degrees", "Sun_distance_au", "Sun_ra_sin", "Sun_ra_cos",
        "Moon_ra_hours", "Moon_dec_degrees", "Moon_distance_au", "Moon_ra_sin", "Moon_ra_cos",
        "Mercury_ra_hours", "Mercury_ra_sin", "Mercury_ra_cos",
        "Venus_ra_hours", "Venus_ra_sin", "Venus_ra_cos",
        "Mars_ra_hours", "Mars_ra_sin", "Mars_ra_cos",
        "Jupiter_ra_hours", "Jupiter_ra_sin", "Jupiter_ra_cos",
        "Saturn_ra_hours", "Saturn_ra_sin", "Saturn_ra_cos",
        "Uranus_ra_hours", "Uranus_ra_sin", "Uranus_ra_cos",
        "Neptune_ra_hours", "Neptune_ra_sin", "Neptune_ra_cos",
        "Pluto_ra_hours", "Pluto_ra_sin", "Pluto_ra_cos",
        "planetary_hour_sin", "planetary_hour_cos",
        "minute_cycle_sin", "minute_cycle_cos"
    ]
    X = df[features]
    y = df['quick_trade']
    X = X.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    smote = SMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.35, random_state=42)
    model = XGBClassifier(n_estimators=100, learning_rate=0.5, max_depth=6,
                          use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X_test)
    print("Combined Model Performance:")
    print(classification_report(y_test, y_pred))
    return model

print("Training Combined Model...")
combined_model = train_model_on_pairs(TRADING_PAIRS)
print("Training Astro-Only Model...")
astro_model = train_astro_model_on_pairs(TRADING_PAIRS)
print("Training Technical-Only Model...")
tech_model = train_tech_model_on_pairs(TRADING_PAIRS)



# ===============================
# PART 2: TRADING BOT
# ===============================
class InstitutionalFlowBotKrakenFutures:
    def __init__(self, starting_balance=131.80, max_trades=10,retraining_frequency=5):
        self.ai_model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        print("✅ AI model and scaler loaded successfully!")
        self.balance = starting_balance
        self.portfolio = {}
        self.trade_log = []
        self.live_prices = {}
        self.price_history = {}
        self.volume_history = {}
        self.retraining_frequency = retraining_frequency  # Editable frequency for retraining
        self.cycle_count = 0  # To keep track of cycles
        
        self.kraken_client = krakenex.API()
        self.kraken_client.key = os.getenv("API_KEY") or "ROdbPGCRNEyu13wOn/uI4rV6GWLKMHenvBAnGuO7qG9Qozhoid3S6h54"
        self.kraken_client.secret = os.getenv("API_SECRET") or "ljrlynS77+2kKnKHBcKw6IraIeVYQ6VQijI4ezAhweauRIKZ2wcO42rTZgkARVgTB4TY5eZkuu8ekb4GLoTK5w=="

        self.ephemeris = load('de421.bsp')
        self.planets = {
            'Sun': self.ephemeris['sun'],
            'Moon': self.ephemeris['moon'],
            'Mercury': self.ephemeris['mercury'],
            'Venus': self.ephemeris['venus'],
            'Mars': self.ephemeris['mars'],
            'Jupiter': self.ephemeris['jupiter barycenter'],
            'Saturn': self.ephemeris['saturn barycenter'],
            'Uranus': self.ephemeris['uranus barycenter'],
            'Neptune': self.ephemeris['neptune barycenter'],  # <-- corrected line
            'Pluto': self.ephemeris['pluto barycenter']
        }
        self.earth = self.ephemeris['earth']

        #self.portfolio_file = "portfolio.json"
       # self.portfolio = self.load_portfolio()

        self.pair_mapping = {
            
            "ETH/USD": "ETHUSD",
            "ADA/USD": "ADAUSD",
            "SOL/USD": "SOLUSD",
            "LTC/USD": "LTCUSD",
            "DOT/USD": "DOTUSD",
            "XRP/USD": "XRPUSD",
            "XLM/USD": "XLMUSD",
            "NANO/USD": "NANOUSD",
            "XBT/USD": "XBTUSD",
            "RUNE/USD": "RUNEUSD",
            "SUI/USD":"SUIUSD",
           # "USDT/USD":"USDTUSD","KAITO/USD": "KAITOUSD",  "XDG/USD":"XDGUSD"    "XDG/USD":"XDGUSD",
         
            "TAO/USD":"TAOUSD",
        
            "XDG/USD":"XDGUSD",
            "KAITO/USD": "KAITOUSD",
            "POPCAT/USD": "POPCATUSD",
            "COTI/USD": "COTIUSD"
           
        }
        self.trading_pairs = list(self.pair_mapping.keys())
        self.websocket_url = "wss://ws.kraken.com"
        self.bootstrap_price_history()

        logging.basicConfig(
            filename="bot_activity.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        self.trade_count = 0
        self.max_trades = max_trades
        print(f"🚀 Kraken Trading Bot Initialized | Starting Balance: ${self.balance}")

    # def save_portfolio(self):
    #     with open(self.portfolio_file, "w") as f:
    #         json.dump(self.portfolio, f)

    # def load_portfolio(self):
    #     try:
    #         with open(self.portfolio_file, "r") as f:
    #             return json.load(f)
    #     except FileNotFoundError:
    #         return {}


    


    @staticmethod
    def get_current_planetary_hour_utc():
        utc_now = datetime.now(pytz.utc)
        planetary_sequence = ['Saturn', 'Jupiter', 'Mars', 'Sun', 'Venus', 'Mercury', 'Moon']
        weekday_rulers = ['Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Sun']
        weekday = utc_now.weekday()  # Monday=0, Sunday=6
        first_hour_ruler = weekday_rulers[weekday]
        first_hour_index = planetary_sequence.index(first_hour_ruler)
        full_sequence = [planetary_sequence[(first_hour_index + i) % 7] for i in range(24)]
        sunrise_utc = utc_now.replace(hour=6, minute=0, second=0, microsecond=0)
        sunset_utc = utc_now.replace(hour=18, minute=0, second=0, microsecond=0)
        if sunrise_utc <= utc_now < sunset_utc:
            planetary_hour_length = (sunset_utc - sunrise_utc).total_seconds() / 12
            elapsed_seconds = (utc_now - sunrise_utc).total_seconds()
            hour_number = int(elapsed_seconds // planetary_hour_length)
        else:
            if utc_now >= sunset_utc:
                next_sunrise_utc = sunrise_utc + timedelta(days=1)
                planetary_hour_length = (next_sunrise_utc - sunset_utc).total_seconds() / 12
                elapsed_seconds = (utc_now - sunset_utc).total_seconds()
                hour_number = int(elapsed_seconds // planetary_hour_length) + 12
            else:  # before sunrise
                previous_sunset_utc = sunset_utc - timedelta(days=1)
                planetary_hour_length = (sunrise_utc - previous_sunset_utc).total_seconds() / 12
                elapsed_seconds = (utc_now - previous_sunset_utc).total_seconds()
                hour_number = int(elapsed_seconds // planetary_hour_length) + 12
        return full_sequence[hour_number % 24]


    # def adjust_trade_parameters(self, trade_side, price):
    #     """
    #     Adjusts trade parameters based on lunar cycle:
    #     - Reduces TP if the market is at a lunar midpoint.
    #     - Avoids high-risk trades near Full/New Moon.
    #     """
    #     phase_midpoint = self.lunar_data['lunar_phase_midpoint']
    #     sign_midpoint = self.lunar_data['moon_zodiac_midpoint']

    #     tp_multiplier = 0.00777  # Default TP multiplier

    #     if phase_midpoint in [90, 270]:  # Quarter Moon Midpoints (Reversal Zones)
    #         tp_multiplier *= 0.9  # Reduce TP slightly but stay profitable
        
    #     elif phase_midpoint == 180:  # Full Moon (Potential Exhaustion)
    #         if trade_side == "buy":
    #             print("⚠️ Avoiding Long Trades Near Full Moon")
    #             return None  # Skip trade
    #         tp_multiplier *= 1.2  # Increase TP for short trades
        
    #     if sign_midpoint == 15:  # Lunar Sign Midpoint
    #         tp_multiplier *= 0.8  # Reduce TP but stay above fee level

    #     adjusted_tp = round(price * (1 + tp_multiplier), 5)
    #     return adjusted_tp
    
    def bootstrap_price_history(self):
        for symbol in self.trading_pairs:
            kraken_symbol = self.pair_mapping.get(symbol, symbol)
            params = {"pair": kraken_symbol, "interval": 1}
            response = self.kraken_client.query_public("OHLC", params)
            if response.get("error"):
                print(f"❌ Error bootstrapping history for {symbol}: {response['error']}")
                continue
            result_key = list(response["result"].keys())[0]
            ohlc_data = response["result"][result_key]
            if len(ohlc_data) >= 50:
                closes = [float(record[4]) for record in ohlc_data][-200:]
                volumes = [float(record[6]) for record in ohlc_data][-200:]
                self.price_history[symbol] = closes
                self.volume_history[symbol] = volumes
                print(f"🔄 Bootstrapped history for {symbol}")
            else:
                print(f"⚠️ Not enough historical data for {symbol} during bootstrap.")

    async def websocket_handler(self):
        async with websockets.connect(self.websocket_url) as websocket:
            subscribe_msg = {
                "event": "subscribe",
                "pair": self.trading_pairs,
                "subscription": {"name": "ticker"}
            }
            await websocket.send(json.dumps(subscribe_msg))
            print(f"📡 Subscribed to Kraken WebSocket Market Data for {self.trading_pairs}")
            while True:
                try:
                    data = await websocket.recv()
                    msg = json.loads(data)
                    if isinstance(msg, list) and len(msg) > 1:
                        symbol = msg[-1]
                        price = float(msg[1]['c'][0])
                        self.live_prices[symbol] = price
                except Exception as e:
                    print(f"❌ WebSocket error: {e}")
                    break

    def get_min_order_size(self, symbol):
        try:
            response = self.kraken_client.query_public("AssetPairs")
            if response.get("error"):
                print(f"❌ Error fetching min order size: {response['error']}")
                return None
            kraken_symbol = self.pair_mapping.get(symbol, symbol)
            for pair_key, pair_data in response["result"].items():
                if pair_data.get("altname") == kraken_symbol and "ordermin" in pair_data:
                    return float(pair_data["ordermin"])
            return None
        except Exception as e:
            print(f"❌ Exception fetching min order size: {e}")
            return None

    def get_asset_price_precision(self, symbol):
        response = self.kraken_client.query_public("AssetPairs")
        if response.get("error"):
            logging.error(f"❌ Failed to fetch asset pairs: {response['error']}")
            return 2
        for pair, details in response["result"].items():
            if details["altname"] == symbol:
                return details.get("pair_decimals", 2)
        return 2

    def build_ohlc_df(self, symbol, window=200):
        prices = self.price_history.get(symbol, [])
        volumes = self.volume_history.get(symbol, [])
        if len(prices) < window or len(volumes) < window:
            return None
        bars = []
        for i in range(len(prices) - window + 1):
            chunk_prices = prices[i : i + window]
            chunk_volumes = volumes[i : i + window]
            bar = {
                "time": datetime.now(),  # approximate
                "open": chunk_prices[0],
                "high": max(chunk_prices),
                "low": min(chunk_prices),
                "close": chunk_prices[-1],
                "volume": sum(chunk_volumes)
            }
            bars.append(bar)
        return pd.DataFrame(bars)
    



   


    @staticmethod
    def apply_indicators(df):
        """Apply trading indicators to the DataFrame to match expected model features."""
        df['3_EMA'] = df['close'].ewm(span=3, adjust=False).mean()
        df['5_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
        df['8_EMA'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_comparison'] = (df['3_EMA'] > df['8_EMA']).astype(int)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=50, min_periods=50).mean()
        avg_loss = loss.rolling(window=50, min_periods=50).mean().replace(0, 1e-8)
        df['rsi_5'] = 100 - (100 / (1 + avg_gain / avg_loss))
        df['ATR'] = (df['high'] - df['low']).rolling(window=50, min_periods=50).mean()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['volume_ma'] = df['volume'].rolling(window=50).mean()
        df['volume_spike'] = (df['volume'] > df['volume_ma']).astype(int)
        df['stoch_rsi'] = (df['rsi_5'] - df['rsi_5'].rolling(window=50).min()) / (
        df['rsi_5'].rolling(window=50).max() - df['rsi_5'].rolling(window=50).min())
        df['close_minus_open'] = df['close'] - df['open']
        df['high_minus_low'] = df['high'] - df['low']
        df['close_pct_change'] = df['close'].pct_change().fillna(0)
        df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
        df['obv'] = ((df['close'] - df['close'].shift(1)) * df['volume']).cumsum()
        df['obv_momentum_shift'] = df['obv'].diff().fillna(0)
        df['heikin_ashi_trend'] = ((df['close'] + df['open']) / 2).diff().fillna(0)
        df['100_EMA'] = df['close'].ewm(span=50, adjust=False).mean()
        df['trend_filter'] = (df['close'] > df['100_EMA']).astype(int)
        df['SMA_100'] = df['close'].rolling(window=50).mean()
        df['SMA_15'] = df['close'].rolling(window=5).mean()
        return df.fillna(0)



    
    def plot_indicators(self, df, symbol):
        """Plot key indicators for visual debugging."""
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['close'], label='Close Price', color='black')
        plt.plot(df.index, df['SMA_100'], label='SMA_100', color='blue')
        plt.plot(df.index, df['SMA_15'], label='SMA_15', color='green')
        plt.plot(df.index, df['3_EMA'], label='3_EMA', linestyle='--', color='red')
        plt.plot(df.index, df['5_EMA'], label='5_EMA', linestyle='--', color='purple')
        ax2 = plt.gca().twinx()
        ax2.plot(df.index, df['rsi_5'], label='RSI_5', color='orange', alpha=0.7)
        ax2.set_ylabel('RSI', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        plt.title(f'Indicators for {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



    def get_refined_astrological_data(self):
        ts = load.timescale()
        now = datetime.utcnow()
        t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)

        positions = {}
        for planet_name, planet in self.planets.items():
            astrometric = self.earth.at(t).observe(planet).apparent()
            ra, dec, distance = astrometric.radec()
            positions[planet_name] = {
                'ra_hours': ra.hours,  # Right ascension in hours
                'dec_degrees': dec.degrees,  # Declination in degrees
                'distance_au': distance.au
            }
        return positions

    
    def calculate_planetary_aspects(self, positions):
        aspects = {}
        planets = list(positions.keys())
        for i in range(len(planets)):
            for j in range(i+1, len(planets)):
                planet_a = planets[i]
                planet_b = planets[j]
                ra_a = positions[planet_a]['ra_hours']
                ra_b = positions[planet_b]['ra_hours']

                angle = abs(ra_a - ra_b) * 15  # Convert RA hours difference to degrees
                angle = angle % 360
                aspects[f"{planet_a}-{planet_b}"] = angle
        return aspects
    

    def interpret_aspects_to_signal(self, aspects):
        volatility_signal = "Neutral"
        for aspect, angle in aspects.items():
            if angle <= 10 or angle >= 350:  # conjunction
                volatility_signal = "High"
                break
            elif 80 <= angle <= 100 or 260 <= angle <= 280:  # square
                volatility_signal = "Moderate"
            elif 115 <= angle <= 125 or 235 <= angle <= 245:  # trine
                volatility_signal = "Smooth"
        return volatility_signal
    
    def user_override_prompt(self, message):
        response = input(f"{message} [y/n]: ").strip().lower()
        return response == 'y'


    # New method for minimal retraining on each cycle
    async def retrain_model_cycle(self):
        """
        Retrain the model minimally using recent data.
        This uses the existing 'train_model_on_pairs' function.
        """
        print("🔄 Starting minimal retraining cycle...")
        # Run training in a separate thread so it doesn't block the event loop
        new_model = await asyncio.to_thread(train_model_on_pairs, self.trading_pairs)
        if new_model is not None:
            self.ai_model = new_model
            # Reload the scaler (it was dumped during training)
            self.scaler = joblib.load(SCALER_PATH)
            print("✅ Retraining complete. Model and scaler updated for next cycle.")
        else:
            print("⚠️ Retraining failed (insufficient data or error). Keeping previous model.")




  

    def play_shopify_sound(self):
        try:
            sound_file = "KrakenBot/kraken-futures-bot/shopify_sale_sound.wav"  # Ensure the file path is correct
            wave_obj = sa.WaveObject.from_wave_file(sound_file)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait for the sound to finish playing
        except Exception as e:
            print(f"⚠️ Error playing sound: {e}")


    async def place_trade(self, symbol, side, price):
        """Place a market order, simulate immediate trade closure, and update local balance."""
        await asyncio.sleep(0.5)  # simulate network delay
        combined_fee = 0.007  # 0.35% taker fee + 0.35% maker rebate

        min_order_size = self.get_min_order_size(symbol)
        if not min_order_size:
            logging.warning(f"⚠️ Could not fetch min order size for {symbol}, skipping trade.")
            return

        # Query the current balance from Kraken
        balance_response = self.kraken_client.query_private("Balance")
        available_balance = float(balance_response["result"].get("ZUSD", 0))
        trade_value = available_balance * 0.01  # Using 2% of available balance per trade
        trade_size = max(round(trade_value / price, 5), min_order_size)
        if available_balance < trade_value:
            logging.warning(f"⚠️ Insufficient balance. Skipping {symbol}.")
            return

        precision = self.get_asset_price_precision(symbol)
        tp_multiplier = 0.0075  # default take-profit multiplier

        # Adjust take-profit price and calculate net profit based on trade side
        if side == "buy":
            tp_price = round(price * (1 + tp_multiplier), precision)
            net_profit = (tp_price - price) * trade_size - (price * trade_size * combined_fee)
        elif side == "sell":
            tp_price = round(price * (1 - tp_multiplier), precision)
            net_profit = (price - tp_price) * trade_size - (price * trade_size * combined_fee)
        else:
            logging.error("Invalid trade side specified.")
            return

        print(f"[DEBUG] {symbol}: Calculated tp_price={tp_price}, net_profit={net_profit:.6f}")
        
        if net_profit <= 0:
            print(f"⚠️ Skipping {symbol}: Net profit {net_profit:.6f} is not positive.")
            return

        order_payload = {
            "ordertype": "market",
            "type": side,
            "volume": str(trade_size),
            "pair": symbol,
            "close[ordertype]": "take-profit",
            "close[price]": str(tp_price),
        }
        response = self.kraken_client.query_private("AddOrder", order_payload)
        if response.get("error"):
            logging.error(f"❌ Trade failed for {symbol}: {response.get('error', 'Unknown error')}")
            return response.get("error")

        order_txid = response["result"]["txid"][0]
        
        print(f"✅ Placed {side.upper()} order for {symbol} at {price} (TP={tp_price}), TXID={order_txid}")
        logging.info(f"✅ Placed {side.upper()} order for {symbol} at {price} (TP={tp_price})")

        # **📢 Play Shopify sound when trade is placed**
        self.play_shopify_sound()

        self.portfolio[symbol] = {"type": side, "price": price, "txid": order_txid}
        self.trade_count += 1

        self.balance += net_profit  # Update local balance
        print(f"💰 Updated local balance: {self.balance:.2f}")
        return order_txid



    def shutdown(self):
       #  self.save_portfolio()
        print("Bot shutting down. Portfolio saved.")

    @staticmethod
    def filter_active_pairs(trading_pairs, price_history, window=100, volume_history=None, volume_threshold=100):
        active_pairs = []
        for symbol in trading_pairs:
            prices = price_history.get(symbol, [])
            if len(prices) < window:
                continue
            sma = sum(prices[-window:]) / window
            current_price = prices[-1]
            if current_price <= sma:
                continue
            if volume_history:
                volumes = volume_history.get(symbol, [])
                if len(volumes) < window:
                    continue
                avg_volume = sum(volumes[-window:]) / window
                if avg_volume < volume_threshold:
                    continue
            active_pairs.append(symbol)
        return active_pairs

    def sort_trading_pairs_by_movement(self):
        ranked = []
        for symbol in self.trading_pairs:
            closes = self.price_history.get(symbol, [])
            if len(closes) < 2:
                ranked.append((symbol, 0.0))
                continue
            first_close = closes[0]
            last_close = closes[-1]
            if first_close == 0:
                change_pct = 0.0
            else:
                change_pct = (last_close - first_close) / abs(first_close)
            ranked.append((symbol, change_pct))
        ranked.sort(key=lambda x: x[1], reverse=True)
        self.trading_pairs = [pair for (pair, pct) in ranked]
        print("🔀 Reordered trading pairs by movement:", self.trading_pairs)
    
    
    async def execute_strategy(self):
        """Main trading strategy loop, now with lunar awareness."""
        MIN_CONFIDENCE = 0.95  # Minimum confidence threshold for trades

        asyncio.create_task(self.websocket_handler())  # Start WebSocket for live price updates
        self.sort_trading_pairs_by_movement()  # Sort pairs based on price movement

        favorable_planets = {'Mercury', 'Jupiter', 'Venus', 'Sun', 'Moon'}  # Best planets for trading

        while True:
            if self.trade_count >= self.max_trades:
                print(f"✅ Max trades ({self.max_trades}) reached. Stopping the bot.")
                break

            # 🌙 Fetch latest lunar cycle data
            self.lunar_data = calculate_lunar_midpoints()
            phase_midpoint = self.lunar_data['lunar_phase_midpoint']
            sign_midpoint = self.lunar_data['moon_zodiac_midpoint']
            print(f"🌙 Lunar Cycle: {self.lunar_data['lunar_cycle_stage']} | Midpoint: {phase_midpoint}° | Sign Midpoint: {sign_midpoint}°")

            # ☀️ Fetch planetary data
            current_planet = self.get_current_planetary_hour_utc()
            is_astrologically_favorable = current_planet in favorable_planets
            positions = self.get_refined_astrological_data()
            aspects = self.calculate_planetary_aspects(positions)
            volatility_signal = self.interpret_aspects_to_signal(aspects)

            print(f"🌟 Planetary Hour: {current_planet} ({'Favorable' if is_astrologically_favorable else 'Not Favorable'}) | Astro Volatility: {volatility_signal}")
            await asyncio.sleep(1)

            if not is_astrologically_favorable or volatility_signal == "Neutral":
                print("🛑 Astrological signals are not favorable, but overriding to continue trading...")

            for symbol in self.trading_pairs:
                if self.trade_count >= self.max_trades:
                    break  # Stop if max trades reached

                price = self.live_prices.get(symbol)
                if not price or price <= 0:
                    continue  # Skip if price data is invalid

                df = self.build_ohlc_df(symbol, window=60)
                if df is None or len(df) < 60:
                    continue  # Skip if insufficient data

                df = self.apply_indicators(df)  # Apply technical indicators

                # 🪐 Add planetary data
                astro_data = self.get_refined_astrological_data()
                for planet, values in astro_data.items():
                    df[f"{planet}_ra_hours"] = values["ra_hours"]
                    df[f"{planet}_dec_degrees"] = values["dec_degrees"]
                    df[f"{planet}_distance_au"] = values["distance_au"]
                    df[f"{planet}_ra_sin"] = np.sin((values["ra_hours"] / 24) * 2 * np.pi)
                    df[f"{planet}_ra_cos"] = np.cos((values["ra_hours"] / 24) * 2 * np.pi)

                # 🕐 Encode planetary hour
                current_ph = self.get_current_planetary_hour_utc()
                ph_numeric = convert_planetary_hour_to_numeric(current_ph)
                ph_sin, ph_cos = encode_planetary_hour(ph_numeric)
                df['planetary_hour_sin'] = ph_sin
                df['planetary_hour_cos'] = ph_cos

                # ⏳ Encode minute cycle (for micro scalping)
                minute_sin, minute_cos = encode_minute_cycle(datetime.now(), period=5)
                df['minute_cycle_sin'] = minute_sin
                df['minute_cycle_cos'] = minute_cos

                # 🎯 Select feature set
                features = [
                    "open", "high", "low", "close", "volume", "vwap",
                    "3_EMA", "5_EMA", "8_EMA", "rsi_5", "ATR", "ema_comparison",
                    "volume_ma", "volume_spike", "trend_filter",
                    "Sun_ra_hours", "Sun_dec_degrees", "Sun_distance_au", "Sun_ra_sin", "Sun_ra_cos",
                    "Moon_ra_hours", "Moon_dec_degrees", "Moon_distance_au", "Moon_ra_sin", "Moon_ra_cos",
                    "Mercury_ra_hours", "Mercury_ra_sin", "Mercury_ra_cos",
                    "Venus_ra_hours", "Venus_ra_sin", "Venus_ra_cos",
                    "Mars_ra_hours", "Mars_ra_sin", "Mars_ra_cos",
                    "Jupiter_ra_hours", "Jupiter_ra_sin", "Jupiter_ra_cos",
                    "Saturn_ra_hours", "Saturn_ra_sin", "Saturn_ra_cos",
                    "Uranus_ra_hours", "Uranus_ra_sin", "Uranus_ra_cos",
                    "Neptune_ra_hours", "Neptune_ra_sin", "Neptune_ra_cos",
                    "Pluto_ra_hours", "Pluto_ra_sin", "Pluto_ra_cos",
                    "planetary_hour_sin", "planetary_hour_cos",
                    "minute_cycle_sin", "minute_cycle_cos"
                ]

                X_live = df[features].iloc[[-1]]
                X_live = X_live.replace([np.inf, -np.inf], np.nan).bfill().ffill()
                X_live_scaled = self.scaler.transform(X_live)

                # 🎯 Get model prediction
                prediction_raw = self.ai_model.predict_proba(X_live_scaled)[0][1]
                confidence = int(prediction_raw * 100)

                # 📈 Determine trade action
                trade_action = "buy" if prediction_raw > 0.5 else "sell"

                # 🚀 Adjust trade execution with lunar midpoints
                #adjusted_tp = self.adjust_trade_parameters(trade_action, price)

                if confidence >= MIN_CONFIDENCE * 100:
                    order_txid = await self.place_trade(self.pair_mapping.get(symbol, symbol), trade_action, price)
                    if order_txid:
                        print(f"✅ Trade placed for {symbol}, TXID: {order_txid}. Confidence: {confidence}%")
                    else:
                        print(f"❌ Trade placement failed for {symbol}.")
                else:
                    print(f"😴 {symbol} - Low confidence ({confidence}%) or lunar constraint. Skipping.")

                await asyncio.sleep(.5)  # Sleep to prevent rate limiting

            # 🔄 Retrain model every few cycles
            self.cycle_count += 1
            if self.cycle_count % self.retraining_frequency == 0:
                await self.retrain_model_cycle()




async def main():
    bot = InstitutionalFlowBotKrakenFutures(starting_balance=20, max_trades=20  ,retraining_frequency=1)
    try:
        await bot.execute_strategy()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually.")
    finally:
        bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
