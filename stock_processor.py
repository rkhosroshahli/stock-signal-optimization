import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import sma_indicator, ema_indicator



def stock_processor(ticker):
    fillna = False
    msft = yf.Ticker(ticker)
    df_ticker = msft.history(period="max")

    sma_days = 9
    df_ticker[f'trend_sma_{sma_days}'] = sma_indicator(close=df_ticker['Close'], window=sma_days, fillna=fillna)
    sma_days = 20
    df_ticker[f'trend_sma_{sma_days}'] = sma_indicator(close=df_ticker['Close'], window=sma_days, fillna=fillna)
    sma_days = 50
    df_ticker[f'trend_sma_{sma_days}'] = sma_indicator(close=df_ticker['Close'], window=sma_days, fillna=fillna)
    sma_days = 200
    df_ticker[f'trend_sma_{sma_days}'] = sma_indicator(close=df_ticker['Close'], window=sma_days, fillna=fillna)

    ema_days = 9
    df_ticker[f'trend_ema_{ema_days}'] = ema_indicator(close=df_ticker['Close'], window=ema_days, fillna=fillna)
    ema_days = 20
    df_ticker[f'trend_ema_{ema_days}'] = ema_indicator(close=df_ticker['Close'], window=ema_days, fillna=fillna)
    ema_days = 50
    df_ticker[f'trend_ema_{ema_days}'] = ema_indicator(close=df_ticker['Close'], window=ema_days, fillna=fillna)
    ema_days = 200
    df_ticker[f'trend_ema_{ema_days}'] = ema_indicator(close=df_ticker['Close'], window=ema_days, fillna=fillna)

    df_ticker = df_ticker.loc['2022':'2023']

    return df_ticker


