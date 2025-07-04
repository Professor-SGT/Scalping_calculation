#!/usr/bin/env python
# coding: utf-8

# In[ ]:

print('モジュールをインポートしています...')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import numpy as np
import seaborn as sns
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
# ややインストールに手間がかかる
import talib
import os
import time

# In[ ]:




def vr(df, window=26, type=1):
    """
    FORMULA
    Volume Ratio (VR)
    Formula:
    VR[A] = SUM(av + cv/2, n) / SUM(bv + cv/2, n) * 100
    VR[B] = SUM(av + cv/2, n) / SUM(av + bv + cv, n) * 100
    Wako VR = SUM(av - bv - cv, n) / SUM(av + bv + cv, n) * 100
        av = volume if close > pre_close else 0
        bv = volume if close < pre_close else 0
        cv = volume if close = pre_close else 0
    """
    df['av'] = np.where(df['Close'].diff() > 0, df['Volume'], 0)
    avs = df['av'].rolling(window=window, center=False).sum()
    df['bv'] = np.where(df['Close'].diff() < 0, df['Volume'], 0)
    bvs = df['bv'].rolling(window=window, center=False).sum()
    df['cv'] = np.where(df['Close'].diff() == 0, df['Volume'], 0)
    cvs = df['cv'].rolling(window=window, center=False).sum()
    df.drop(['av', 'bv', 'cv'], inplace=True, axis=1)
    
    if type == 1: # VR[A]
        vr = (avs + cvs / 2) / (bvs + cvs / 2) * 100  
    elif type == 2: # VR[B]
        vr = (avs + cvs / 2) / (avs + bvs + cvs) * 100
    else: # Wako VR
        vr = (avs - bvs - cvs) / (avs + bvs + cvs) * 100
    return vr



# In[ ]:




# テクニカル指標
def Technical_Marking(df, folder):
    close, volume = df['Close'], df['Volume']

    high, low, open = df['High'], df['Low'], df['Open']

    # Simple Moving Average
    
    
    df['sma5'] = talib.SMA(close, timeperiod=5)
    df['sma15'] = talib.SMA(close, timeperiod=15)
    df['sma25'] = talib.SMA(close, timeperiod=25)
    df['sma50'] = talib.SMA(close, timeperiod=50)

    
    # Bollinger Bands
    df['upper1'], df['middle'], df['lower1'] = talib.BBANDS(close, timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
    df['upper2'], middle, df['lower2'] = talib.BBANDS(close, timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)

    
    # MACD - Moving Average Convergence/Divergence
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    df['ema5volume'] = talib.SMA(volume, timeperiod=5)
    df['ema25volume'] = talib.SMA(volume, timeperiod=25)
    df['ema50volume'] = talib.SMA(volume, timeperiod=50)

    
    # 'Monthly' Volume Ratio
    df['vr'] = vr(df) 
    df['vr14'] = vr(df, window=14)
    
    # RSI - Relative Strength Index
    df['rsi9'] = talib.RSI(close, timeperiod=9)
    df['rsi14'] = talib.RSI(close, timeperiod=14)

    


    



    return df



# In[ ]:
def yfUpdate(data):
    print(data.head())
    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    data = data.set_index('Date')
    return data


def main(symbol='^N225', start=None, end=None):
    
    if symbol == 'temp':
        tickers = pd.read_csv('Code/temp.csv'.format(symbol))['Code']    
    elif not (symbol == 'JPX400'):
        tickers = pd.read_csv('Code/{}codelist.csv'.format(symbol))['Code']
    else:
        tickers = pd.read_csv('Code/JPNK400.csv'.format(symbol))['0'].values
    data = pd.DataFrame()
    if symbol in ['TOPIX', 'Nikkei225',]:
        extension = '.T'
    else:
        extension = ''
    
    
    folder = 'Price/{}/'.format(symbol.replace('.T', '').replace("'", ""))
    if not os.path.isdir(folder):
        print(folder)
        os.mkdir(folder)
    print(tickers)
    for ticker in tickers:
        print(ticker, start, end)
        # ticker = extension
        time.sleep(0.1)
        df = yf.download(str(ticker).replace("'", "")+extension, start=start, end=end, interval='1d').reset_index(drop=False)
        

        

        try:
            df = yfUpdate(df)
            df = Technical_Marking(df, folder)
        except:
            continue

        df.to_csv(folder+'{}.csv'.format(ticker))
    
    return
 
 
if __name__ == '__main__':
    
    
    print('実行中です...')
    end = dt.today().date()
    start=(end-relativedelta(months=2)).replace(day=1)
    symbol = input("What ticker(for yfinance) do you set? If you do press enter-key simply, it sets '日経225指数' automatic. Nikkei, TOPIX, NYdow, S&P500, NASDAQ, JPX400?")
    if symbol == '':
        symbol = 'Nikkei225'
    main(symbol=symbol,
         start=start,
    end=end)
      
 


# In[ ]:




