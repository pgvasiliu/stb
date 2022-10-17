#!/usr/bin/env python3

import argparse
from distutils.command.build_py import build_py
import json
import math
import os

import sys
#sys.path.insert(0, './util')

import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import datetime as dt


import requests
import matplotlib.dates as mpl_dates

# pip3 install tradingview_ta
from tradingview_ta import TA_Handler, Interval, Exchange


#####  PANDAS  #####
import yfinance as yf
import pandas as pd

import numpy as np

#######################
#####  FUNCTIONS  #####
#######################
def taJson(product, exch, myinterval):
    p = product.replace('-', '')
    screener = "america"

    ta = TA_Handler(
        symbol=p,
        screener=screener,
        exchange=exch,
        interval=myinterval
    )
    try:
        analysis = ta.get_analysis()
        return analysis
    except Exception as e:
        print(f'{SIGNAL_NAME}Exception:')
        print(e)
        sys.exit(1)

#
# Method to send message on Discord. You need to create a Discord webhook for your channel and paste it here
#
def send_discord_message (discord, ticker, title, description):
    # curl -i -H "Accept: application/json" -H "Content-Type:application/json" -X POST --data '{"content": "Posted Via Command line"}' https://discord.com/api/webhooks/102...../YPhlm.......

    my_time         = time.strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "username" : "Python Trading Bot",
        "content" : "\n-----\n" + ticker + ' ' + my_time
    }
    data["embeds"] = [
        {
            "title" : title,
            "description" : description
        }
    ]
    result = requests.post(discord, json = data)
    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    #else:
    #    print("DISCORD Payload delivered successfully, code {}.".format(result.status_code))


def decimals ( number ):
    f = '{0:.2f}'.format(number)
    return float(f)

def ro(num):
    if num is None:
        return 0

    return round(num, 2)

#def is_close(a, b, tol=1e-8):
#    return np.abs(a - b) < tol

def is_close(a, b):
    relative = 1e-09
    absolute = 0.0
    return abs(a - b) <= max(relative * max(abs(a), abs(b)), absolute)

##############################
#####  CANDLE  PATTERNS  #####
##############################
# https://github.com/nathanmcmillan/napa-bot/blob/b69a43bc994942be2b633295abcb7a954e26fcc4/python/patterns.py #
#def hammer(o,c,h,l):
#    body = abs(o - c)
#    wick = abs(min(o, c ) - l)
#    if wick > body * 2.0:
#        if is_close(c, h):
#            return 'green'
#        elif is_close(o, h):
#            return 'red'
#    return 'NA'

    
def hammer(ohlc_df):    
    """returns dataframe with hammer candle column"""
    df = ohlc_df.copy()
    df["hammer"] = (((df["High"] - df["Low"])>3*(df["Open"] - df["Close"])) & \
                   ((df["Close"] - df["Low"])/(.001 + df["High"] - df["Low"]) > 0.6) & \
                   ((df["Open"] - df["Low"])/(.001 + df["High"] - df["Low"]) > 0.6)) & \
                   (abs(df["Close"] - df["Open"]) > 0.1* (df["High"] - df["Low"]))
    return df

def shooting_star(candle):
    body = abs(candle.open - candle.closing)
    wick = abs(max(candle.open, candle.closing) - candle.high)
    if wick > body * 2.0:
        if is_close(candle.open, candle.low):
            return 'green'
        elif is_close(candle.closing, candle.low):
            return 'red'
    return ''

def marubozu(candle):
    if is_close(candle.open, candle.low) and is_close(candle.closing, candle.high):
        return 'green'
    if is_close(candle.open, candle.high) and is_close(candle.closing, candle.low):
        return 'red'
    return ''


def trend(candles, start, end):
    if candles[end].closing > candles[start].closing:
        return 'green'
    return 'red'


def change(candles, start, end):
    return abs(candles[end].closing - candles[start].closing) / candles[start].closing


def volume_trend(candles, start, end):
    if candles[end].volume > candles[start].volume:
        return 'green'
    return 'red'


def color(candle):
    if candle.closing > candle.open:
        return 'green'
    return 'red'






#####  PANDAS  data  download  #####
def download_pandas ( symbol ):
    # Yahoo fix for '.'
    symbol = symbol.replace ( '.', '-' )
    ticker = yf.Ticker( symbol )
    hist = ticker.history(period='2y', interval='1d')
    #return pd.DataFrame(hist)
    dataframe = pd.DataFrame(hist).drop(columns=['Stock Splits', 'Dividends'])
    #df = df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume','Dividends':'dividends' , 'Stock Splits':'splits'})
    return pd.DataFrame(dataframe)


#####  PANDAS crossover & crossunder  #####
def pandas_crossover(a, b):
    return a[-2] < b[-2] and a[-1] > b[-1]

def pandas_crossunder(a, b):
    return a[-2] > b[-2] and a[-1] < b[-1]



########################
#####  PANDAS SMA  #####
########################
#def SMA ( close, t ):
#    import talib
#    return talib.SMA( close, t)
# https://github.com/Priyanshu154/Backtest/blob/511e2e8525b23a14ecdf5a48c28399c7fd41eb14/Backtest/Backtest/Indicator.py
def __SMA(close, t):
    mas = []
    for i in range(t - 1):
        mas.append(-1)
    for i in range(len(close) - t + 1):
        summ = 0
        for j in range(i, t + i):
            summ = summ + close[j]
        meann = summ / t
        mas.append(meann)
    return mas
#SMA Ends here

#def __SMA(data, t):
#    sma = data.rolling(t).mean()
#    return sma



#############################################
#####  PANDAS  Commodity Channel Index  #####
#############################################
#def CCI ( data, t=20):
#  import talib
#  return talib.CCI ( data['High'], data['Low'], data['Close'], timeperiod=t)
def __CCI(df, ndays = 20):
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['sma'] = df['TP'].rolling(ndays).mean()
    df['mad'] = df['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad())
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
    return df

#########################
#####  PANDAS TEMA  #####
#########################
#def TEMA ( close, t):
#    import talib
#    return talib.TEMA(close, timeperiod=t)
def __TEMA ( df, t=30 ):
    ema1 = df['Close'].ewm(span = t ,adjust = False).mean()
    ema2 = ema1.ewm(span = t ,adjust = False).mean()
    ema3 = ema2.ewm(span = t ,adjust = False).mean()
    return (3*ema1)-(3*ema2) + ema3


#####  EMA  #####
#def EMA(close, t):
#    import numpy as np
#    import talib
#    return talib.EMA ( np.array(close), t)
def __EMA ( df, t=9 ):
    #def EMA(close, t):
    #    import numpy as np
    #    return talib.EMA ( np.array(close), t)
    ema = df['Close'].ewm(span = t ,adjust = False).mean()
    return ( ema )


##################################################
#####  PANDAS  Weighted Moving Average(WMA)  #####
##################################################
#def wma(src, length):
#    import talib
#    return talib.WMA(src, length)
#
# https://github.com/Priyanshu154/Backtest/blob/511e2e8525b23a14ecdf5a48c28399c7fd41eb14/Backtest/Backtest/Indicator.py
# Reference for code is taken from tradingview
def __WMA(close, t):
    wma = []
    for i in range(t - 1):
        wma.append(-1)
    for i in range(t-1, len(close)):
        norm = 0.0
        summ = 0.0
        for j in range(0, t):
            weight = (t-j)*t
            norm = norm + weight
            summ = summ + (close[i-j]*weight)
        wma.append(summ/norm)
    return wma
# WMA Ends Here


###########################
#####  PANDAS WILL%R  #####
###########################
#def WILLR( data, t=14):
#    import talib
#    return talib.WILLR ( data['High'], data['Low'], data['Close'], timeperiod=t)
def __WILLR (high, low, close, t):
    import pandas as pd
    import numpy as np
    import math

    highh = high.rolling(t).max()
    lowl = low.rolling(t).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr


#########################
#####  PANDAS  KDJ  #####
#########################
def __KDJ (df, n=9, ph='High', pl='Low', pc='Close'):
    df['max'] = df[ph].rolling(window=n).max()
    df['min'] = df[pl].rolling(window=n).min()
    df['rsv'] = (df[pc] - df['min']) / (df['max'] - df['min']) * 100

    # df.dropna(inplace=True)

    df['K'] = df['rsv'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    df.drop(['max', 'min', 'rsv'], axis=1, inplace=True)

    return df

# MACD
def __MACD (df, m=12, n=26, p=9, pc='Close'):

    df = df.copy()
    df['EMA_s'] = df[pc].ewm(span=m, adjust=False).mean()
    df['EMA_l'] = df[pc].ewm(span=n, adjust=False).mean()

    df['MACD']  = df['EMA_s'] - df['EMA_l']
    #df["MACD"] = df.apply(lambda x: (x["EMA_s"]-x["EMA_l"]), axis=1)
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=p, adjust=False).mean()
    df['MACD_HIST']   = (df['MACD'] - df['MACD_SIGNAL'])

    df.drop(['EMA_s', 'EMA_l'], axis=1, inplace=True)

    return df

def calculate_MACD(self,short_window, long_window, triger_line_days):
        
    df = copy.deepcopy(self.data)
        
    ewm_short = df['close'].ewm(span=short_window, adjust=False, min_periods=short_window).mean()
    ewm_long = df['close'].ewm(span=long_window, adjust=False, min_periods=long_window).mean()
        
    macd = ewm_short - ewm_long  #MACD line
    macd_s = macd.ewm(span=triger_line_days, adjust=False, min_periods=triger_line_days).mean() #MACD signal
    macd_d = macd - macd_s #Difference
        
    run_name = '%s_%s_%s' %(short_window,long_window,triger_line_days)
        
    df['macd_' + run_name] = df.index.map(macd)
    df['macd_s'+ run_name] = df.index.map(macd_s)
    df['macd_d'+ run_name] = df.index.map(macd_d)
    df = df[['macd_' + run_name,'macd_d'+ run_name,'macd_s'+ run_name]].round(4)
        

#################
#####  RSI  #####
#################
# https://github.com/Priyanshu154/Backtest/blob/511e2e8525b23a14ecdf5a48c28399c7fd41eb14/Backtest/Backtest/Indicator.py
#def RSI ( close, t ):
#    import talib
#    return talib.RSI ( close, timeperiod=t)
def __RSI(close, t):
    n = len(close)
    rsi = []
    Ups = 0.0
    Downs = 0.0
    for j in range(t-1):
        rsi.append(-1)
    #Ye sabse pehla avgU/avgD find karne ke liye simple average vala step
    for i in range(1,t):
        diff = close[i] - close[i-1]
        if(diff > 0):
            Ups += diff
        else:
            Downs += (-diff)

    preU = Ups/t
    preD = Downs/t
    #simple average mil gaya to hamara pehla rsi bi mil gaya
    rs = preU/preD
    rsi.append( (100 - (100/(1+rs))) )
    #yaha se prev_avgUp vala loop
    Ups = 0.0
    Downs = 0.0
    for i in range(t,n):
        diff = close[i] - close[i-1]
        if(diff > 0):
            Ups = diff
            Downs = 0.0
        else:
            Downs = (-diff)
            Ups = 0.0
        u = (1/t)*Ups + ((t-1)/t)*preU
        d = (1/t)*Downs + ((t-1)/t)*preD
        preU = u    #Update previous-Up and previous-Down
        preD = d
        rs = u/d
        rsi.append( (100 - (100/(1+rs))) )   #RSI for a particular date
    return rsi
#RSI Ends Here

"""
data["RSI_144"] = RSI ( _close, 14 )
def RSI (data, time_window):
    # Function to compute the RSI or Relative Strength Index for a stock. 
    # Attempts to give a person an indication if a particular stock is over- or under-sold

    diff = data.diff(1).dropna()
    # diff in one field(one day)
    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

    rsi = 100 - 100 / (1 + abs(up_chg_avg/down_chg_avg))

    return rsi
"""

def stoch_rsi(rsi, d_window=3, k_window=3, window=14):
    """
    Computes the stochastic RSI. Default values are d=3, k=3, window=14.
    """
    minrsi = rsi.rolling(window=window, center=False).min()
    maxrsi = rsi.rolling(window=window, center=False).max()
    stoch = ((rsi - minrsi) / (maxrsi - minrsi)) * 100
    K = stoch.rolling(window=k_window, center=False).mean()
    D = K.rolling(window=d_window, center=False).mean()
    return K, D


##################################################
#####  PANDAS  Rolling Moving Average (RMA)  #####
##################################################
def __RMA(close, t):
    rma = []
    sma = SMA(close, t)
    for i in range(t):
        rma.append(sma[i])
    for i in range(t, len(close)):
        rma.append( (rma[i-1]*(t-1) + close[i])/t )
    return rma
# RMA Ends here


########################################
##### PANDAS  Rate Of Change(ROC)  #####
########################################
#def __ROC ( data, t=10):
#  return talib.ROC ( data['Close'], timeperiod=t)
def __ROC(close, t):
    roc = []
    for i in range(t-1):
        roc.append(-1)
    for i in range(t-1, len(close)):
        sum = 100*(close[i]-close[i-t])/close[i-t]
        roc.append(sum)
    return roc
# ROC Ends here

########################
#####  PANDAS ATR  #####
########################
#def __ATR(df_func):
#    import numpy as np
#    # Calculating ATR - Average True Range
#    high_low   = df_func['High'] - df_func['Low']
#    high_close = np.abs(df_func['High'] - df_func['Close'].shift())
#    low_close  = np.abs(df_func['Low'] - df_func['Close'].shift())
#
#    ranges     = pd.concat([high_low, high_close, low_close], axis=1)
#    true_range = np.max(ranges, axis=1)
#
#    df_func['ATR_14'] = true_range.rolling(14).sum()/14
#
#    return df_func

# https://stackoverflow.com/questions/40256338/calculating-average-true-range-atr-on-ohlc-data-with-python
def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

# https://stackoverflow.com/questions/40256338/calculating-average-true-range-atr-on-ohlc-data-with-python
def __ATR (df, n=14):
    data = df.copy()
    high = data['High']
    low = data['Low']
    close = data['Close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr


##############################
#####  PANDAS ATR BANDS  #####
##############################
def __ATR_bands ( data, t=14 ):
    _open  = data['Open']
    _close = data['Close']
    _high  = data['High']
    _low   = data['Low']

    atr = __ATR( data, t)

    atr_multiplicator = 2.0
    atr_basis = __EMA ( data, 20)

    
    atr_band_upper  = atr_basis + atr_multiplicator * atr
    atr_band_lower  = atr_basis - atr_multiplicator * atr
    atr_band_middle = atr_basis

    return atr_band_lower[-1], atr_band_upper[-1], atr_band_middle[-1]


########################
#####  PANDAS MOM  #####
########################
def MOM ( data, n):
    return data['Close'] / data['Close'].shift(n) - 1



#################
#####  FIB  #####
#################
def fib_retracement(p1, p2):
    list =[0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 3.618, 4.236]
    dict = {}
    dist = p2 - p1
    for val in list:
        dict[str(val) ] =  "%.2f" % (p2 - dist*val)
    return dict
#Fibonacci Retracement ends here

def fib (data):
    #Calculate the max and min close price
    maximum_price = data['Close'].max()
    minimum_price = data['Close'].min()
    return fib_retracement ( maximum_price, minimum_price )

# https://github.com/sanampatel/python-pivot-levels/blob/master/pivot_levels.py #
def classic(opens, high, low, close):
    pivot = dict()
    ran = ro(high - low)
    pivot["PP"] = ro((high + low + close) / 3)

    pivot["R1"] = ro((pivot["PP"] * 2) - low)
    pivot["R2"] = ro(pivot["PP"] + ran)
    pivot["R3"] = ro(pivot["R2"] + ran)  # pivot["PP"] + (ran * 2)
    pivot["R4"] = ro(pivot["R3"] + ran)   # pivot["PP"] + (ran * 3)

    pivot["S1"] = ro((pivot["PP"] * 2) - high)
    pivot["S2"] = ro(pivot["PP"] - ran)
    pivot["S3"] = ro(pivot["S2"] - ran)
    pivot["S4"] = ro(pivot["S3"] - ran)

    return pivot


def fibonacci(opens, high, low, close):
    pivot = dict()
    pivot["PP"] = ro((high + low + close) / 3)

    pivot["R1"] = ro(pivot["PP"] + ((high - low) * .382))
    pivot["R2"] = ro(pivot["PP"] + ((high - low) * .618))
    pivot["R3"] = ro(pivot["PP"] + ((high - low) * 1.000))

    pivot["S1"] = ro(pivot["PP"] - ((high - low) * .382))
    pivot["S2"] = ro(pivot["PP"] - ((high - low) * .618))
    pivot["S3"] = ro(pivot["PP"] - ((high - low) * 1.000))

    return pivot


def woodie(opens, high, low, close):
    pivot = dict()
    ran = ro(high - low)
    pivot["PP"] = ro((high + low + close) / 3)

    pivot["PP"] = ro((high + low + (opens * 2)) / 4)

    pivot["R1"] = ro((pivot["PP"] * 2) - low)
    pivot["R2"] = ro(pivot["PP"] + ran)
    pivot["R3"] = ro(pivot["R1"] + ran)
    pivot["R4"] = ro(pivot["R3"] + ran)

    pivot["S1"] = ro((pivot["PP"] * 2) - high)
    pivot["S2"] = ro(pivot["PP"] - ran)
    pivot["S3"] = ro(pivot["S1"] - ran)
    pivot["S4"] = ro(pivot["S3"] - ran)

    return pivot

def camarilla(opens, high, low, close):
    pivot = dict()
    ran = ro(high - low)

    pivot["PP"] = ro((high + low + close) / 3)

    pivot["R1"] = ro(close + (ran * 1.1/12))
    pivot["R2"] = ro(close + (ran * 1.1/6))
    pivot["R3"] = ro(close + (ran * 1.1/4))
    pivot["R4"] = ro(close + (ran * 1.1/2))

    pivot["S1"] = ro(close - (ran * 1.1/12))
    pivot["S2"] = ro(close - (ran * 1.1/6))
    pivot["S3"] = ro(close - (ran * 1.1/4))
    pivot["S4"] = ro(close - (ran * 1.1/2))

    return pivot

def demark(opens, high, low, close):
    pivot = dict()
    factor = 0

    if close < opens:
        factor = (high + (low * 2) + close)

    if close > opens:
        factor = ((high * 2) + low + close)

    if close == opens:
        factor = (high + low + (close * 2))

    pivot["PP"] = ro(factor / 4)
    pivot["R1"] = ro(factor / 2 - low)
    pivot["S1"] = ro(factor / 2 - high)

    return pivot


# https://github.com/sanampatel/python-pivot-levels/blob/master/pivot_levels.py #
def pivot_levels(opens, high, low, close):
    pivot = dict()
    pivot["pivot"] = dict()

    #pivot["input"] = {"open": opens, "high": high, "low": low, "close": close}
    #pivot["pivot"]["classic"] = classic(opens, high, low, close)
    #pivot["pivot"]["fibonacci"] = fibonacci(opens, high, low, close)
    #pivot["pivot"]["woodie"] = woodie(opens, high, low, close)
    pivot["pivot"]["camarilla"] = camarilla(opens, high, low, close)
    #pivot["pivot"]["demark"] = demark(opens, high, low, close)

    return pivot



###########################
#####  SUPPORT & RES  #####
###########################
def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support

def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
    return resistance

def sr ( df ):
    mylist = []

    def isFarFromLevel(l):
        return np.sum([abs(l-x) < s  for x in levels]) == 0

    #import datetime as dt
    start = (dt.datetime.now()-dt.timedelta(days=365)).strftime('%Y-%m-%d')

    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

    s =  np.mean(df['High'] - df['Low'])

    levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):
            l = df['Low'][i]

            if isFarFromLevel(l):
                levels.append((i,l))

        elif isResistance(df,i):
            l = df['High'][i]

            if isFarFromLevel(l):
                levels.append((i,l))

    for a, b in levels:
        b = "%.2f" % b
        mylist.append ( b )
    sorted_float = sorted( mylist, key = lambda x:float(x))

    return sorted_float

"""
###################################
#####  Implement TTM SQUEEZE  #####
###################################
def ttm_squeeze(df):
    df['20sma'] = df['Close'].rolling(window=20).mean()
    df['stddev'] = df['Close'].rolling(window=20).std()
    df['lower_band'] = df['20sma'] - (2 * df['stddev'])
    df['upper_band'] = df['20sma'] + (2 * df['stddev'])

    df['TR1'] = abs(df['High'] - df['Low'])
    df['ATR'] = df['TR1'].rolling(window=20).mean()

    df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
    df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

    def in_squeeze(df):
        return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

    df['squeeze_on'] = df.apply(in_squeeze, axis=1)

    if df.iloc[-3]['squeeze_on'] and df.iloc[-1]['squeeze_on']:
        return True, False
    elif df.iloc[-3]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
        return True, True
    else:
        return False, False

def vix(close, low, pd=23, bbl=23, mult=1.9, lb=88, ph=0.85, pl=1.01):
    hst = highest(close, pd)
    wvf = (hst - low) / hst * 100
    s_dev = mult * stdev(wvf, bbl)
    mid_line = sma(wvf, bbl)
    lower_band = mid_line - s_dev
    upper_band = mid_line + s_dev

    range_high = (highest(wvf, lb)) * ph
    range_low = (lowest(wvf, lb)) * pl

    green_hist = [wvf[-i] >= upper_band[-i] or wvf[-i] >= range_high[-i] for i in range(8)][::-1]
    red_hist = [wvf[-i] <= lower_band[-i] or wvf[-i] <= range_low[-i] for i in range(8)][::-1]

    return green_hist, red_hist
"""

####################################
#####  PANDAS BOLLINGER BANDS  #####
####################################
def __BB (data, lookback):
    std = data.rolling(lookback).std()
    upper_bb  = __SMA(data, lookback) + std * 2
    lower_bb  = __SMA(data, lookback) - std * 2
    middle_bb = __SMA(data, lookback)
    return upper_bb, lower_bb, middle_bb

#
#def add_ema(data, tspan=[12, 26, 21, 50, 34, 55, 99, 200]):
#    """
#    Adds Exponential Moving Averages (EMA) to the dataframe. The default timeframes are 12,26,20,50,34 and 55.
#    """
#    for t in tspan:
#        data[f'ema{t}'] = data[CLOSE].ewm(span=t).mean()
#        data[f'dist_ema{t}'] = data[CLOSE] - data[f'ema{t}']
#    return data


#######################################################
#####  PANDAS  STOCHASTIC OSCILLATOR CALCULATION  #####
#######################################################

def __STOCHASTIC (df, k, d):
#     """
#     Fast stochastic calculation
#     %K = (Current Close - Lowest Low)/
#     (Highest High - Lowest Low) * 100
#     %D = 3-day SMA of %K
#
#     Slow stochastic calculation
#     %K = %D of fast stochastic
#     %D = 3-day SMA of %K
#
#     When %K crosses above %D, buy signal
#     When the %K crosses below %D, sell signal
#     """
     temp_df = df.copy()
     # Set minimum low and maximum high of the k stoch
     low_min = temp_df["Low"].rolling(window=k).min()
     high_max = temp_df["High"].rolling(window=k).max()

     # Fast Stochastic
     temp_df['k_fast'] = 100 * (temp_df["Close"] - low_min)/(high_max - low_min)
     temp_df['d_fast'] = temp_df['k_fast'].rolling(window=d).mean()

     # Slow Stochastic
     temp_df['k_slow'] = temp_df["d_fast"]
     temp_df['d_slow'] = temp_df['k_slow'].rolling(window=d).mean()
     return temp_df

def vol_spike(DF, n=20):
    df = DF.copy()
    df['MA_Vol'] = df["volume"].ewm(span=n,min_periods=n).mean()
    df['vol_spike'] = np.where(df['volume']>2.4*df['MA_Vol'],1,0)
    return df



###########################
#####  MAIN  PROGRAM  #####
###########################
settings = {
    "enable_debug": 0,
    "enable_discord": 1
}

#Discord Webhook URL
discord_env = {
    "PROD": 'https://discord.com/api/webhooks/1024457008390340660/V5MaHfnUCBwjQw5zlW4DFTj_wEDZsMXGLpy3dxLUv969oz86N0KITLKZds3ju9lC7tvt',
    "DEV": 'https://discord.com/api/webhooks/1026327480900014162/YPhlm0QMkHoOZXmpL2IC1BPwIIWBBm3MEzW02RYHu4yNyYMVOfRXDI8sfkV5HXCjuITG'
}

#discord_url = discord_env['PROD']
discord_url = discord_env['DEV']

#TRADE_SIGNALS = {'buy': ('buy', -1), 'hold': ('hold', 0), 'sell': ('sell', 1)}


# this is a list of tickers and  match for what to buy / sell
strategy = []

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+', help='Load json file  --> format of file is:   { "ticker1": "exchange", "ticker2": "exchange" }')

args=parser.parse_args()

ticker_files = sorted ( set (args.files) )
for myfile in ticker_files:
    print ("\n\n")
    print('--------------------------------------------------------------------')

    # load the tickers json config file
    with open(myfile) as f:
        data = json.load(f)
        #ticker_exchange = dict(sorted(data.items()))
        ticker_exchange = dict(data.items())
        # loop through tickers
        for symbol, exchange in ticker_exchange.items():


            # Pull 1 day data and 1 week data
            json_analysis_1d   = taJson(symbol, exchange, Interval.INTERVAL_1_DAY)
            #json_analysis_1w   = taJson(symbol, exchange, Interval.INTERVAL_1_WEEK)


            # overall rating, oscillator and moving averages recommendation
            recommendation      = json_analysis_1d.summary['RECOMMENDATION']
            osc_recommendation  = json_analysis_1d.oscillators['RECOMMENDATION']
            mave_recommendation = json_analysis_1d.moving_averages['RECOMMENDATION']


            # indicator names + values
            ind_1d  = json_analysis_1d.indicators
            #ind_1w  = json_analysis_1w.indicators

            # oscilators names + values
            osc_1d = json_analysis_1d.oscillators



            ##################
            #####  DATA  #####
            ##################


            # current price
            price        = ind_1d['close']
            price_string = str ( price )

            _low    = ind_1d['low']
            _high   = ind_1d['high']
            _open   = ind_1d['open']
            _close  = ind_1d['close']
            _vol    = ind_1d['volume']
            _change = ind_1d['change']

            _change_2dec = f'{_change:.2f}'

            buyVol = _vol * (_close - _low) / (_high - _low)
            sellVol = _vol * (_high - _close) / (_high - _low)
            buySellRatio = 100 * (buyVol) / (buyVol + sellVol)
            totVol = round(buyVol, 0) + round(sellVol, 0) ;
            buyPercent  = ( round(buyVol, 0)  / totVol ) * 100
            sellPercent = ( round(sellVol, 0) / totVol ) * 100

            ########################
            #####  indicators  #####
            ########################
            # Tradingview indicators start with '_'
            _rsi   = ind_1d["RSI"]
            _rsi_1 = ind_1d['RSI[1]']   # RSI previous day

            _stock_k  = ind_1d['Stoch.K']
            _stock_d  = ind_1d['Stoch.D']

            _stock_k_1 = ind_1d['Stoch.K[1]']  # stock k Previous day
            _stock_d_1 = ind_1d['Stoch.D[1]']  # stock k Previous day

            #print ( "Stoch:Last day K1 %s, D1 %s,  CUR: K %s, D %s " % ( _stock_k_1, _stock_d_1,  _stock_k,  _stock_d )    )

            _stock_rsi_k = float(ind_1d['Stoch.RSI.K'])

            _ema10  = ind_1d['EMA10']
            _ema20  = ind_1d['EMA20']
            _ema30  = ind_1d['EMA30']
            _ema50  = ind_1d['EMA50']
            _ema100 = ind_1d['EMA100']
            _ema200 = ind_1d['EMA200']

            _cci20  = ind_1d['CCI20']
            _cci20_1 = ind_1d['CCI20[1]']

            _wr     = ind_1d['W.R']

            ###################################
            #####  download PANDAS  data  #####
            ###################################
            symbol = symbol.replace ( '.', '-' )
            df     = download_pandas ( symbol )

            #_vol  = ind_1d['volume']
            vol_1  = df['Volume'][-2]
            #print ("vol = %s, vol_1 = %s " % (_vol, vol_1 ))

            # Previous day data
            low_1    = df['Low'][-2]
            high_1   = df['High'][-2]
            open_1   = df['Open'][-2]
            close_1  = df['Close'][-2]

            #analysis_text = ' | hammer {}'.format( hammer( _open, _close, _high, _low ))
            #analysis_text += ' | macd {:.2f}'.format(macd.current)
            #analysis_text += ' | flow {:.2f}'.format(money_flow_index.current)
            #analysis_text += ' | obv {:.2f}'.format(balance_volume.current)
            #analysis_text += ' | rsi {:.2f}'.format(relative_strength_index.current)
            #analysis_text += ' | hammer {}'.format(patterns.hammer(candles[-1]))
            #analysis_text += ' | star {}'.format(patterns.shooting_star(candles[-1]))
            #analysis_text += ' | marubozu {}'.format(patterns.marubozu(candles[-1]))
            #print ( analysis_text )
            
            
            # if enable_discord  = 1 ( see setting above )
            #if ( settings['enable_discord']):
            #    discord_message = "TICKER: %s   Current: o=%.2f c=%.2f l=%.2f h=%.2f , Previous day: o=%.2f c=%.2f l=%.2f h=%.2f" % ( symbol, _open, _close, _low, _high, open_1, close_1, low_1, high_1)
            #    send_discord_message (discord_url, symbol, symbol, discord_message)


            df['Volume_60D_Mean'] = df['Volume'].rolling(60, min_periods=1).mean()
            #df['Volume_60D_Mean'] = df['Volume'].rolling(window = 5).mean().shift(1)
            df['Market Cap']      = df['Open'] * df['Volume']



            #####  ADD SMA 5, 8 to pandas data  #####
            df['SMA_5']   = __SMA ( df['Close'], 5 )
            sma_5         = decimals ( df['SMA_5'][-1] )
            sma_5_1       = decimals ( df['SMA_5'][-2] )
            #sma_5        = ind_1d['SMA5']

            df['SMA_8']   = __SMA ( df['Close'], 8 )
            sma_8         = decimals ( df['SMA_8'][-1] )
            sma_8_1       = decimals ( df['SMA_8'][-2] )

            #print ("SMA5_1 = %s, SMA5 = %s " % (sma_5_1, sma_5 ))
            #print ("SMA8_1 = %s, SMA8 = %s " % (sma_8_1, sma_8 ))

            ######################
            #####  calc RSI  #####
            ######################
            df['RSI_14']   = __RSI ( df['Close'], 14 )

            df['RSI_14']   = __RSI(df['Close'], 14)
            #df['Stoch_RSI_K'], df['Stoch_RSI_D'] = stoch_rsi(df['RSI_14'], 3, 3, 14)


            #sq1, sq2 = ttm_squeeze(df)


            ###################################
            #####  calc DATA :: CCI, ATR  #####
            ###################################
            df = __CCI(df, 20)
            #df = __ATR(df_func=df)


            df = hammer ( df )


            #################################
            #####  calc DATA :: ATR 14  #####
            #################################
            df['ATR_14']  = __ATR ( df, 14 )
            atr_14        = df['ATR_14'][-1]
            atr_14_1      = df['ATR_14'][-2]


            #################################
            ##### calc DATA :: TEMA 30  #####
            #################################
            df['TEMA_30'] = __TEMA ( df, 30 )
            tema_30       = df['TEMA_30'][-1]
            tema_30_1     = df['TEMA_30'][-2]

            df['TEMA_9'] = __TEMA ( df, 9 )


            ###############################
            #####  calc DATA :: EMA 9 #####
            ###############################
            df['EMA_9']   = __EMA ( df, 9 )
            ema_9         = df['EMA_9'][-1]
            ema_9_1       = df['EMA_9'][-2]

            df['EMA_21']  = __EMA ( df, 21 )
            ema_21        = df['EMA_21'][-1]
            ema_21_1      = df['EMA_21'][-2]


            ##############################
            #####  calc DATA :: W%R  #####
            ##############################
            # Oversold area is <= -80, Overbought area is >= -20
            df['WILLR_14'] = __WILLR ( df['High'], df['Low'], df['Close'], 14 )
            willr_14       = df['WILLR_14'][-1]
            willr_14_1     = df['WILLR_14'][-2]

            df['WILLR_20'] = __WILLR ( df['High'], df['Low'], df['Close'], 20 )
             
            
            #######################
            #####  calc MACD  #####
            #######################
            df       = __MACD (df)
            _macd         = float(ind_1d["MACD.macd"])
            #_macd     = df['MACD'][-1]
            _macd_1   = df['MACD'][-2]

            _macd_hist   = df['MACD_HIST'][-1]
            _macd_hist_1 = df['MACD_HIST'][-2]

            _macd_signal   = float(ind_1d["MACD.signal"]) 
            #_macd_signal  = df['MACD_SIGNAL'][-1]
            _macd_signal_1 = df['MACD_SIGNAL'][-2]


            ######################
            #####  calc KDJ  #####
            ######################
            df = __KDJ (df)


            #############################
            #####  calc BBOL bands  #####
            #############################
            df['BB_up'], df['BB_low'], df['BB_middle'] = __BB ( df['Close'], 20 )
            bb_up     = df['BB_up'][-1]
            bb_up_1   = df['BB_up'][-2]

            bb_low     = df['BB_low'][-1]
            bb_low_1   = df['BB_low'][-2]

            bb_middle     = df['BB_middle'][-1]
            bb_middle_1   = df['BB_middle'][-2]

            ####################################
            #####  calc TV STOCHASTIC RSI  #####
            ####################################
            # TradingView's Stoch RSI is different than regular Stoch RSI 
            # https://github.com/freqtrade/freqtrade/issues/2961
            period = 14
            smoothD = 3
            SmoothK = 3
            stochrsi  = (df['RSI_14'] - df['RSI_14'].rolling(period).min()) / (df['RSI_14'].rolling(period).max() - df['RSI_14'].rolling(period).min())
            df['TV_SRSI_k'] = stochrsi.rolling(SmoothK).mean() * 100
            df['TV_SRSI_d'] = df['TV_SRSI_k'].rolling(smoothD).mean()





            ##########################################################
            #####  calc  STOCHASTIC 10, 6,6 not default 14, 3, 3 #####
            ##########################################################
            df = __STOCHASTIC (df, 14,3)




            ####################################
            #####  FIBONACCI  retracement  #####
            ####################################
            #fibonacci     = fib ( symbol )


            ####################################
            ######  CAMARILLA retracement  #####
            ####################################


            ########################################
            #####  calc ATR bands ( Keltner )  #####
            ########################################
            atr_band_lower, atr_band_higher, atr_band_middle = __ATR_bands ( df, 14 )


            #df['Golden Cross'] = df['Close'].rolling(50, min_periods=1).mean() - df['Close'].rolling(200, min_periods=1).mean()

            #######################################################
            #####  THIS IS WHERE YOU CAN SEE THE PANDAS DATA  #####
            if ( settings['enable_debug'] ):
                print ( df.tail() )
                #                  Open        High         Low       Close     Volume       SMA_5       SMA_8          TP         sma        mad         CCI    ATR_14     TEMA_30       EMA_9      EMA_21   WILLR_14
                #Date                                                                                                                                                                                                 
                #2022-09-30  361.799988  365.910004  357.040009  357.179993  153396100  363.638000  367.218754  360.043335  383.199253  11.917402 -129.535603  8.185571  363.844079  369.162755  380.485210 -99.684798
                #2022-10-03  361.079987  368.549988  359.209991  366.609985   89756500  364.097998  365.871250  364.789988  381.783992  12.484594  -90.746533  8.413030  362.566060  368.652201  379.223826 -74.506670
            #######################################################




            #############################
            #####  STRATEGIES  !!!  #####
            #############################

            # 9 spaces
            message = 9 * ' '

            # A list of  messages for a ticker
            advice = []


            #####  (1) (SMA 5,8): SMA 5, 8 crossover / crossunder  #####
            if ( sma_5 > sma_8 ) and ( _close > sma_5 ) and ( _close > sma_8 ) and ( _open < _close ):
                advice.append("  [BULLISH]  (SMA 5,8)  {SMA 5 > 8}")

                if ( sma_5_1 < sma_8_1 ):
                    advice.append ("  [BULLISH]  (SMA 5,8)  {SMA 5 crossover SMA 8}")

            if ( sma_5 < sma_8 ) and ( _close < sma_5 ) and ( _close < sma_8 ) and ( _open > _close ):
                advice.append("  [BEARISH]  (SMA 5,8)  {SMA 5 > 8}")

                if ( sma_5_1 > sma_8_1 ):
                    advice.append ("  [BEARISH]  (SMA 5,8)  {SMA 5 crossunder SMA 8}")


            if ( df['Close'][-2] <df['SMA_5'][-2] ) & ( df['Close'][-1] > df['SMA_5'][-1] ) & ( df['CCI'][-1] < -100 ):
                advice.append("  [BULLISH]  (SMA 5, CCI)")

            if ( df['Close'][-2] > df['SMA_5'][-2] ) and ( df['Close'][-1] < df['SMA_5'][-1] ) and ( df['CCI'][-1] > 100 ):
                advice.append("  [BEARISH]  (SMA 5, CCI)")


            if df['Close'][-2] < df['SMA_5'][-2] and df['Close'][-1] > df['SMA_5'][-1]:
                advice.append("  [BULLISH]  (SMA 5)")

            if df['Close'][-2] > df['SMA_5'][-2] and df['Close'][-1] < df['SMA_5'][-1]:
                advice.append("  [BEARISH]  (SMA 5)")




            #####  (2) EMA  #####

            #data['ema50'] = data['close'] / data['close'].ewm(50).mean()
            #data['ema21'] = data['close'] / data['close'].ewm(21).mean()
            #data['ema15'] = data['close'] / data['close'].ewm(14).mean()
            #data['ema5'] = data['close'] / data['close'].ewm(5).mean()

            #####  (3) TEMA 30  #####
            if ( _close > tema_30 ) and ( tema_30 > tema_30_1 ) and ( _close > close_1 ) and ( _close > ema_9 ) and ( ema_9 > ema_9_1) and ( ema_9 > tema_30):
                advice.append("  [BULLISH]  (TEMA 30 1)")

            if ( _close < tema_30 ) and ( tema_30 < tema_30_1 ) and ( _close < close_1) and ( _close < ema_9 ) and ( ema_9 < ema_9_1):
                advice.append("  [BEARISH]  (TEMA 30 1)")



            #####  (4) ATR_14  & W%R  #####
            # To reduce the false signal, check the William %R value and should be on the oversold area and previously reach < -95
            if ( ema_9 - ( 2 * atr_14) > _open ) and ( _wr < -80) and ( willr_14_1 < -95 ) and ( _close > _open ):
                advice.append("  [BULLISH]  (ATR_14 ; W%R)")

            # To reduce the false signal, check the William %R value and should be on the overbought area and previously reach > -5
            if ( ema_9 + ( 2 * atr_14) < _close ) and ( _wr > -20 ) and ( willr_14_1 > -5 ):
                advice.append("  [BEARISH]  (ATR_14 ; W%R)")



            #####  (5) CCI signal buy - cciin range(-200,200) & cci > cci avf 4 days  #####
            #if ( _cci20 > -200 ) and ( _cci20 < 200 ) and ( _cci20 > df['CCI'].shift(1).rolling(4).mean()[-1] ):
            #    advice.append("  [BULLISH]  (CCI)")

            #if ( _cci20 > -200 ) and ( _cci20 < 200 ) and ( _cci20 < df['CCI'].shift(1).rolling(4).mean()[-1] ):
            #    advice.append("  [BEARISH]  (CCI)")

            # if cci[i-1] > lower_band and cci[i] < lower_band:
            # if cci[i-1] < upper_band and cci[i] > upper_band:
            
            #if cci <= -200:  ACTIVELY_BUY
            #if cci >= 200:   ACTIVELY_SELL

            # if r[1]['cci(-2)'] < -100 and r[1]['cci(-1)'] > -100:  BUY
            # if r[1]['cci(-2)'] < 100 and r[1]['cci(-1)'] > 100:

            if ( _cci20 > -100 ) & (_cci20_1 < -100):
                advice.append("  [BULLISH]  (CCI_20 CROSSOVER)")

            if ( _cci20 < 100 ) & ( _cci20_1 > 100):
                advice.append("  [BEARISH]  (CCI_20 CROSSUNDER)")



            #####  (6) MACD  #####
            # https://github.com/yaoyao-wang/MACD/tree/bbd120436282ff34ffb092918fd661088820dc81
            # If the MACD line crosses above 0, it is highly likely to be a bull market. And when it crosses below zero, the market tends to be bearish.
            # Therefore, when the histogram is positive which means the MACD line is higher than the signal line, the indicator is considered bullish. 
            # However, when the histogram is negative which means that the MACD line is larger than the signal line, the indicator is considered bearish.
            # d['macd']>=d['signal']) & (d['macd_prev']<d['signal_prev'])  ==> BUY
            # d['macd']<d['signal']) & (d['macd_prev']>=d['signal_prev'])  ==> SELL

            # CROSS_OVER from below 0
            if ( _macd < 0 ) and ( _macd_signal < 0) and ( _macd_hist_1 < 0 ) and (_macd_hist > 0 ) and ( _macd > _macd_1) and ( _macd_signal > _macd_signal_1 ) and ( _macd > _macd_signal ):
                advice.append("  [BULLISH]  (MACD_CROSS_OVER)")

            # CROSS_UNDER from above 0 
            if ( _macd > 0) and ( _macd_signal > 0 ) and ( _macd_hist_1 > 0 ) and ( _macd_hist < 0 ):
                advice.append("  [BEARISH]  (MACD_CROSS_UNDER)")

            if ( _macd < 0 ) and ( _macd_signal < 0 ) and ( _macd > _macd_signal ) and ( _macd_hist > _macd_hist_1):
                advice.append("  [BULLISH]  (MACD_GOOD_BUY)")

            if ( _macd > 0 ) and ( _macd_signal > 0 ) and ( _macd > _macd_signal ) and ( _macd_hist < _macd_hist_1):
                advice.append("  [BULLISH]  (MACD_SELL)")

            #if k[i] < 30 and d[i] < 30 and macd[i] < -2 and macd_signal[i] < -2:
            #if k[i] > 70 and d[i] > 70 and macd[i] > 2 and macd_signal[i] > 2:



            #####  (7)  Stoch RSI  #####
            # https://github.com/angelassets/Binance-Trading-Bot/blob/Binance-Trading-Bot/modules/rsi_stoch_signalmod_djcommie.py
            # Stoch.RSI (25 - 52) & Stoch.RSI.K > Stoch.RSI.D, RSI (49-67), EMA10 > EMA20 > EMA100, Stoch.RSI = BUY, RSI = BUY, EMA10 = EMA20 = BUY
            if (_rsi - _rsi_1 >= 2.5) and ( _rsi >= 49 and _rsi <= 67) and ( _stock_rsi_k >= 25 and _stock_rsi_k <= 58) and '''(EMA10 > EMA20 and EMA20 > EMA100)''' and ( _stock_k - _stock_d >= 4.5):
                advice.append("  [BULLISH]  (StochRSI)")

            # Stoch RSI crossover
            if ( df['TV_SRSI_k'][-1] < 20 ) and ( df['TV_SRSI_d'][-1] < 20 ) and ( df['TV_SRSI_k'][-1] > df['TV_SRSI_k'][-2] ) and ( df['TV_SRSI_d'][-2] >= df['TV_SRSI_k'][-2] ) and ( df['TV_SRSI_k'][-1] > df['TV_SRSI_d'][-1]):
                advice.append("  [BULLISH]  (STOCH_RSI CROSSOVER from below)")

            if ( df['TV_SRSI_k'][-1] > 80 ) and ( df['TV_SRSI_d'][-1] < 80 ) and ( df['TV_SRSI_k'][-1] < df['TV_SRSI_k'][-2] ) and ( df['TV_SRSI_d'][-2] < df['TV_SRSI_k'][-2] ) and ( df['TV_SRSI_k'][-1] < df['TV_SRSI_d'][-1]):
                advice.append("  [BULLISH]  (STOCH_RSI CROSSUNDER from above)")            

            # if k[i] < lower_band and d[i] < lower_band and k[i] < d[i]:
            # if k[i] > upper_band and d[i] > upper_band and k[i] > d[i]:


            #####  (8)  Stochastic  #####
            #df['Enter Short'] = ((df['%K'] < df['%D']) & ( df['%K'].shift(1) > df['%D'].shift(1)) ) & ( df['%D']>80 ) & (df['Close'] < df['Rolling Mean'])
            #df['Enter Long'] = ((df['%K']>df['%D']) & (df['%K'].shift(1) < df['%D'].shift(1))) & (df['%D'] < 20) & (df['Close'] > df['Rolling Mean'])

            #data["BUY-SIGNAL"] = (data["STOCHASTIC %D LINE"] < 20) & (data["RSI"] < 50)
            #data["SELL-SIGNAL"] = (data["RSI"] > 70) & (data["STOCHASTIC %D LINE"] > 80)

            # if k[i-1] > 30 and d[i-1] > 30 and k[i] < 30 and d[i] < 30 and prices[i] < lower_bb[i]:
            # if k[i-1] < 70 and d[i-1] < 70 and k[i] > 70 and d[i] > 70 and prices[i] > upper_bb[i]:

            if df['k_slow'][-2] > 30 and df['d_slow'][-2] > 30 and df['k_slow'][-1] < 30 and df['d_slow'][-1] < 30 and _close < bb_low:
                advice.append("  [BEARISH]  (STOCH)")
            if df['k_slow'][-2] < 70 and df['d_slow'][-2] < 70 and df['k_slow'][-1] > 70 and df['d_slow'][-1] > 70 and _close > bb_up:   
                advice.append("  [BEARISH]  (STOCH)")


            # A buy signal is given when the oscillator falls below 20 and then rises above 20.
            if df['k_slow'][-2] < 20 < df['k_slow'][-1] or df['d_slow'][-2] < 20 < df['d_slow'][-1]:
                advice.append("  [BULLISH]  (Stochastic BUY  crossing 20)")
            # A sell signal is given when the oscillator rises above the 80 and then falls below 80.
            if df['k_slow'][-2] > 80 > df['k_slow'][-1] or df['d_slow'][-2] > 80 > df['d_slow'][-1]:
                advice.append("  [BEARISH]  (Stochastic SELL  crossing 80)")
 

 
            # A buy signal occurs when an increasing %K line crosses above the %D line in the  oversold region (%K < 20)
            if df['k_slow'][-2] - df['k_slow'][-1] < 0 < df['k_slow'][-1] - df['d_slow'][-1] and df['k_slow'][-1] < 20:
                advice.append("  [BULLISH]  (Stochastic BUY)")
            # A sell signal occurs when a decreasing %K line crosses below the %D line in the overbought region (%K > 80)
            if df['k_slow'][-2] - df['k_slow'][-1] > 0 > df['k_slow'][-1] - df['d_slow'][-1] and df['k_slow'][-1] > 80:
                advice.append("  [BEARISH]  (Stochastic SELL)")



            if ( df['k_slow'][-1]  < 20 ) & ( df['k_slow'][-1] > df['d_slow'][-1]):
                advice.append("  [BULLISH]  (Stochastic BUY 2)")

            if ( df['k_slow'][-1]  > 80 ) & ( df['k_slow'][-1] < df['d_slow'][-1]):
                advice.append("  [BEARISH]  (Stochastic SELL 2)")


            #####  (9)  BOL Bands  #####
            if ( _close <= bb_low ):
                # oversold
                advice.append("  [BULLISH]  (BB 1)")

            if ( _close >= bb_up ):
                # oversold
                advice.append("  [BEARISH]  (BB 1)")


            if ( ( df['Close'][-1] <  df['BB_low'][-1] ) | (df['Open'][-1] < df['BB_low'][-1]) ):
                advice.append("  [BULLISH]  (BB 2)")

            if ( df['Close'][-1] > df['BB_low'][-1] )    | ( df['Open'][-1] > df['BB_low'][-1] ):
                advice.append("  [BEARISH]  (BB 2)")


            #####  (10)  KDJ  #####
            # You get a buy signal from the KDJ indicator when the three curves converge. 
            # The blue K line crosses the D line from bottom to top and then moves above the yellow J line. The purple D line is at the bottom.
            # The signal is even stronger when the golden form appears under the 20 line, that is in the oversold area.

            #A sell signal is received when the lines converge in a way that the blue line K crosses the line D from top to bottom. The blue line continues below the yellow and the purple one runs above the others.
            #The signal is stronger when the dead fork of the KDJ oscillator occurs in the overbought zone that is above the line of 80 value.

            # KDJ CROSS  https://github.com/pgvasiliu/futu_algo/blob/master/strategies/KDJ_Cross.py
            if ( 20 > df['D'][-1] > df['D'][-2] > df['K'][-2] )  &  ( df['K'][-1] > df['K'][-2] )  &  ( df['K'][-1] > df['D'][-1] ):
                advice.append("  [BULLISH]  (KDJ CROSSOVER)")

            if ( 80 < df['D'][-1] < df['D'][-2] < df['K'][-2] ) and ( df['K'][-1] < df['K'][-2] ) and ( df['K'][-1] < df['D'][-1] ):
                advice.append("  [BEARISH]  (KDJ CROSSUNDER)")



            #####  (11) W%R  #####
            #if wr[i-1] > -80 and wr[i] < -80:   BUY
            #if wr[i-1] < -20 and wr[i] > -20:   SELL

            if df['WILLR_20'][-2] > -50 and df['WILLR_20'][-1] < -50 and _macd > _macd_signal:
                advice.append("  [BULLISH]  (WILLR_20 MACD)")

            if df['WILLR_20'][-2] < -50 and df['WILLR_20'][-1] > -50 and _macd < _macd_signal:
                advice.append("  [BEARISH]  (WILLR_20 MACD)")


            if ( df['WILLR_20'][-2] < -80 ) and ( df['WILLR_20'][-1] > -80 ):
                advice.append("  [BULLISH]  (WILLR_20 CROSSOVER)")

            if ( df['WILLR_20'][-2] < -20 ) and ( df['WILLR_20'][-1] > -20 ):
                advice.append("  [BEARISH]  (WILLR_20 CROSSUNDER)")



            #####  (12)  RSI BBOL TEMA  #####
            # https://github.com/superduong/ALIN/blob/main/freqtrade/templates/sample_.py
            # RSI crosses above 30, tema below BB middle, tema is raising, Volume is not 0
            if ( _rsi >= 30 ) and ( _rsi_1 < 30 ) and ( df['TEMA_9'][-1] <= df['BB_middle'][-1] ) and ( df['TEMA_9'][-1] > df['TEMA_9'][-2] ) and ( _vol > 0):
                advice.append("  [BULLISH]  (RSI_BBOL_TEMA 1)")
            
            if ( _rsi >=70 ) and ( _rsi_1 < 70 ) and ( df['TEMA_9'][-1] > df['BB_middle'][-1] )  and ( df['TEMA_9'][-1] < df['TEMA_9'][-2] ) and ( _vol > 0):
                 advice.append("  [BEARISH]  (RSI_BBOL_TEMA 1)")
            


            #####  (13) SMA 5; W%R  #####
            if df['Close'][-2] < df['SMA_5'][-2] and df['Close'][-1] > df['SMA_5'][-1] and df['WILLR_20'][-1] < -80:
                advice.append("  [BULLISH]  (SMA 5 ; W%R)")
            
            if df['Close'][-2] > df['SMA_5'][-2] and df['Close'][-1] < df['SMA_5'][-1] and df['WILLR_20'][-1] > -20:
                advice.append("  [BEARISH]  (SMA 5 ; W%R)")



            #####  (14) RSI  #####
            #RSI signal buy - RSI in range(25-75) & RSI>RSI avg last 4 days
            if ( df['RSI_14'][-1] > 25 ) &  ( df['RSI_14'][-1] < 75) & ( df['RSI_14'][-1] > df['RSI_14'].shift(1).rolling(window=5).mean()[-2] ):
                advice.append("  [BULLISH]  (RSI_14 1)")

            if ( df['RSI_14'][-1] > 25 ) & ( df['RSI_14'][-1] < 75 ) & ( df['RSI_14'][-1] < df['RSI_14'].shift(1).rolling(window=5).mean()[-2] ):
                advice.append("  [BEARISH]  (RSI_14 1)")



            #####  (13) CUSTOM  #####

            # --------------   BUY ------------- #
            discord_message = ''

            if ( df['WILLR_20'][-2] < -80 ) and ( df['WILLR_20'][-1] > -80 ):
                if ( _cci20 > -100 ) & (_cci20_1 < -100):
                    if df['k_slow'][-2] < 20 < df['k_slow'][-1] or df['d_slow'][-2] < 20 < df['d_slow'][-1]:
                        advice.append("  [BULLISH]  (CUSTOM WILLR_20, CCI_20, STOCH ===> CROSSOVER)")
                        if ( settings['enable_discord']):
                            discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "  [CUSTOM BULLISH]  (CUSTOM WILLR_20, CCI_20, STOCH ===> CROSSOVER)" )
                            #send_discord_message (discord_url, symbol, symbol, discord_message)

            if df['k_slow'][-2] < 20 < df['k_slow'][-1] or df['d_slow'][-2] < 20 < df['d_slow'][-1]:
                if ( settings['enable_discord']):
                    discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "  [CUSTOM BULLISH]  (Stochastic BUY  crossing 20)" )
                    #send_discord_message (discord_url, symbol, symbol, discord_message)


            if ( sma_5 > sma_8 ) and ( _close > sma_5 ) and ( _close > sma_8 ) and ( _open < _close ) and ( sma_5_1 < sma_8_1 ):
                if ( settings['enable_discord']):
                    discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "  [CUSTOM BULLISH]  (SMA 5,8) CROSSOVER" )
                    send_discord_message (discord_url, symbol, symbol, discord_message)


            if ( _close > tema_30 ) and ( _close > close_1 ) and ( _close > ema_9 ) and ( ema_9 > ema_9_1) and ( ema_9 > tema_30) & ( tema_30 > tema_30_1):
                if ( settings['enable_discord']):
                    discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "  [CUSTOM BULLISH]  (TEMA 30 1)" )
                    send_discord_message (discord_url, symbol, symbol, discord_message)


            if ( len ( discord_message) > 10):
                send_discord_message (discord_url, symbol, symbol, discord_message)

            # --------------   SELL ------------- #
            discord_message = ''

            if ( df['WILLR_20'][-2] < -20 ) and ( df['WILLR_20'][-1] < -20 ):
                if ( _cci20 < 100 ) & ( _cci20_1 > 100):
                        if df['k_slow'][-2] > 80 > df['k_slow'][-1] or df['d_slow'][-2] > 80 > df['d_slow'][-1]:
                            advice.append("  [BEARISH]  (WILLR_20, CCI_20, STOCH ===> CROSSUNDER)")
                            if ( settings['enable_discord']):
                                discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "  [CUSTOM BEARISH]  (WILLR_20, CCI_20, STOCH ===> CROSSUNDER)" )
                                #send_discord_message (discord_url, symbol, symbol, discord_message)

            # A sell signal is given when the oscillator rises above the 80 and then falls below 80.
            if df['k_slow'][-2] > 80 > df['k_slow'][-1] or df['d_slow'][-2] > 80 > df['d_slow'][-1]:
                if ( settings['enable_discord']):
                    discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "\n  [CUSTOM BEARISH]  (Stochastic SELL  crossing 80)" )
                    #send_discord_message (discord_url, symbol, symbol, discord_message)

            if ( sma_5 < sma_8 ) and ( _close < sma_5 ) and ( _close < sma_8 ) and ( _open > _close ) and ( sma_5_1 > sma_8_1 ):
                if ( settings['enable_discord']):
                    discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "\n  [CUSTOM BEARISH]  (SMA 5,8) CROSSUNDER" )
                    #send_discord_message (discord_url, symbol, symbol, discord_message)

            if ( _close < tema_30 ) and ( close_1 > tema_30_1 ):
                if ( settings['enable_discord']):
                    discord_message += '  {:8s}  {:10s}   {:8s}%  {:30s} '.format ( symbol, price_string , _change_2dec, "\n  [CUSTOM BEARISH]  (price CROSSUNDER TEMA_30)" )
                    #send_discord_message (discord_url, symbol, symbol, discord_message)

            if ( len ( discord_message) > 10):
                send_discord_message (discord_url, symbol, symbol, discord_message)

                    
            #################################################################################################################################################
            print ('  {:8s}  {:10s}   {:8s}%  {:25s}   {:20s} '.format (symbol, price_string , _change_2dec, recommendation, 'mAVE:' + mave_recommendation ))


            # if enable_discord  = 1 ( see setting above )
            #if ( settings['enable_discord']):
            #    discord_message = '  {:8s}  {:10s}   {:8s}%  {:25s}   {:20s} '.format ( symbol, price_string , _change_2dec, recommendation, 'mAVE:' + mave_recommendation )
            #    send_discord_message (discord_url, symbol, symbol, discord_message)


            if ( len ( advice ) > 0):
                message += symbol + ' # ' + ", ".join( advice )
                strategy.append ( symbol + ' # ' + ", ".join( advice ) )
                print ( message )


            print ( "         [%s] FIBs CAM ---> %s" % ( symbol,  pivot_levels(_open, _high, _low, _close) ) )
            print ( "         [%s] ATR_band ---> (LOW %.2f, %.2f%% away )   CUR %s   (MAX %.2f, %.2f%% away)" % ( symbol, atr_band_lower, 100 - ( atr_band_lower * 100 / price ), price_string, atr_band_higher, 100 - ( price * 100 / atr_band_higher  ) ) )
            print ( "         [%s] SupRes   ---> %s" % ( symbol, sr ( df ) ) )
            print ( "         [%s] FIBs     ---> %s" % ( symbol, fib (df) ) )

            print('--------------------------------------------------------------------')
            time.sleep(2)

    #####  BUY / SELL  list  #####
    if ( len ( strategy ) > 0):
        print ("\n\n results:\n" )
        for line in strategy:
            print ( line )

