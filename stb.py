#!/usr/bin/env python3

import argparse
import json
import math
import os

import sys
sys.path.insert(0, './util')

import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import datetime as dt


import requests


# pip3 install tradingview_ta
from tradingview_ta import TA_Handler, Interval, Exchange


#####  PANDAS  #####
import yfinance as yf
import pandas as pd


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
    # curl -i -H "Accept: application/json" -H "Content-Type:application/json" -X POST --data '{"content": "Posted Via Command line"}' https://discord.com/api/webhooks/1026327480900014162/YPhlm0QMkHoOZXmpL2IC1BPwIIWBBm3MEzW02RYHu4yNyYMVOfRXDI8sfkV5HXCjuITG

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


#####  PANDAS  data  download  #####
def download_pandas ( symbol ):
    # Yahoo fix for '.'
    symbol = symbol.replace ( '.', '-' )
    ticker = yf.Ticker( symbol )
    hist = ticker.history(period='2y', interval='1d')
    #return pd.DataFrame(hist)  
    dataframe = pd.DataFrame(hist).drop(columns=['Stock Splits', 'Dividends'])
    return pd.DataFrame(dataframe)


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


########################
#####  PANDAS ATR  #####
########################
def __ATR(df_func):
    import numpy as np
    # Calculating ATR - Average True Range
    high_low   = df_func['High'] - df_func['Low']
    high_close = np.abs(df_func['High'] - df_func['Close'].shift())
    low_close  = np.abs(df_func['Low'] - df_func['Close'].shift())

    ranges     = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    df_func['ATR_14'] = true_range.rolling(14).sum()/14
    
    return df_func


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


##############################
#####  PANDAS ATR BANDS  #####
##############################
def __ATR_bands ( data, t=14 ):
    _open  = data['Open']
    _close = data['Close']
    _high  = data['High']
    _low   = data['Low']

    atr = __ATR( data)

    atr_multiplicator = 2.0
    atr_basis = __EMA ( data, 20)

    atr_band_upper = atr_basis + atr_multiplicator * atr
    atr_band_lower = atr_basis - atr_multiplicator * atr
    
    return atr_band_lower[-1], atr_band_upper[-1]


###########################
#####  MAIN  PROGRAM  #####
###########################


#Discord Webhook URL for testing: https://gist.github.com/Bilka2/5dd2ca2b6e9f3573e0c2defe5d3031b2
discord_url   = 'https://discord.com/api/webhooks/1026327480900014162/YPhlm0QMkHoOZXmpL2IC1BPwIIWBBm3MEzW02RYHu4yNyYMVOfRXDI8sfkV5HXCjuITG'
#discord_url = 'https://discord.com/api/webhooks/1024457008390340660/V5MaHfnUCBwjQw5zlW4DFTj_wEDZsMXGLpy3dxLUv969oz86N0KITLKZds3ju9lC7tvt'

# this is a list of tickers and strategy match for what to buy / sell
strategy = []

# Set debugging to 1 will print additional stuff
debug = 0

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
        ticker_exchange = dict(sorted(data.items()))
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

            _macd_orange   = float(ind_1d["MACD.macd"])    # MACD blue   line
            _macd_blue     = float(ind_1d["MACD.signal"])  # MACD orange line

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

            discord_message = "TICKER: %s   Current: o=%.2f c=%.2f l=%.2f h=%.2f , Previous day: o=%.2f c=%.2f l=%.2f h=%.2f" % ( symbol, _open, _close, _low, _high, open_1, close_1, low_1, high_1)
            send_discord_message (discord_url, symbol, symbol, discord_message)

            #####  ADD SMA 5, 8 to pandas data  #####
            df['SMA_5']   = __SMA ( df['Close'], 5 )
            sma_5         = decimals ( df['SMA_5'][-1] )
            sma_5_1       = decimals ( df['SMA_5'][-2] )

            df['SMA_8']   = __SMA ( df['Close'], 8 )
            sma_8         = decimals ( df['SMA_8'][-1] )
            sma_8_1       = decimals ( df['SMA_8'][-2] )

            #print ("SMA5_1 = %s, SMA5 = %s " % (sma_5_1, sma_5 ))
            #print ("SMA8_1 = %s, SMA8 = %s " % (sma_8_1, sma_8 ))


            ###################################
            #####  calc DATA :: CCI, ATR  #####
            ###################################
            df = __CCI(df, 20) 
            df = __ATR(df_func=df)

            #################################
            #####  calc DATA :: ATR 14  #####
            #################################
            #df['ATR_14']  = ATR ( df, 14 )
            #atr_14        = df['ATR_14'][-1]
            #atr_141       = df['ATR_14'][-2]


            #################################
            ##### calc DATA :: TEMA 30  #####
            #################################
            df['TEMA_30'] = __TEMA ( df, 30 )
            tema_30       = df['TEMA_30'][-1]
            tema_30_1     = df['TEMA_30'][-2]


            ###############################
            #####  calc DATA :: EMA 9 #####
            ###############################
            df['EMA_9']   = __EMA ( df, 9 )
            ema_9         = df['EMA_9'][-1]
            ema_9_1       = df['EMA_9'][-2]

            df['EMA_21']   = __EMA ( df, 21 )
            ema_9          = df['EMA_21'][-1]
            ema_21_1       = df['EMA_21'][-2]


            ##############################
            #####  calc DATA :: W%R  #####
            ##############################
            df['WILLR_14'] = __WILLR ( df['High'], df['Low'], df['Close'], 14 )
            willr_14       = df['WILLR_14'][-1]
            willr_14_1     = df['WILLR_14'][-2]


            ####################################
            #####  FIBONACCI  retracement  #####
            ####################################
            #fibonacci     = fib ( symbol )


            ###################################
            #####  ATR bands ( Keltner )  #####
            ###################################
            #atr_band_lower, atr_band_higher = __ATR_bands ( df, 14 )



            #######################################################
            #####  THIS IS WHERE YOU CAN SEE THE PANDAS DATA  #####
            if ( debug ):
                print ( df.tail() )
            #######################################################




            #############################
            #####  STRATEGIES  !!!  #####
            #############################

            # 9 spaces
            message = 9 * ' '

            # A list of strategy messages
            advice = []


            #####  1. (SMA 5,8): SMA 5, 8 crossover / crossunder  #####
            if ( sma_5 > sma_8 ) and ( _close > sma_5 ) and ( _close > sma_8 ) and ( _open < _close ):
                advice.append("STRATEGY  [BULLISH]  (SMA 5,8)  {SMA 5 > 8}")

                if ( sma_5_1 < sma_8_1 ):
                    advice.append ("STRATEGY  [BULLISH]  (SMA 5,8)  {SMA 5 crossover SMA 8 today}")


            #####  // TEMA 30 strategy // #####
            #if ( price > tema_30 ) and ( price > ema_9 ) and ( ema_9 > tema_30):
            #    print ( "TEMA , EMA 9 BUY" )

            # Buy If price currently lower than MA substracts by ATR (with some multiplier)
            # To reduce the false signal, check the William %R value and should be on the oversold area and previously reach < -95
            #if ( ema_9 - ( 2 * atr_14) > _open ) and ( _wr < -80) and ( willr_14_1 < -95 ) and ( _close > _open ):
            #    print ( "TV BUY" )

            # Sell If price currently higher than MA add by ATR (with some multiplier)
            # To reduce the false signal, check the William %R value and should be on the overbought area and previously reach > -5
            #if ( ema_9 + ( 2 * atr_14) < _close ) and ( _wr > -20 ) and ( willr_14_1 > -5 ):
            #    print ( "TV SELL" )

            print ('  {:8s}  {:10s}   {:8s}%  {:25s}   {:20s} '.format (symbol, price_string , _change_2dec, recommendation, 'mAVE:' + mave_recommendation ))

            #discord_message = '  {:8s}  {:10s}   {:8s}%  {:25s}   {:20s} '.format ( symbol, price_string , _change_2dec, recommendation, 'mAVE:' + mave_recommendation )
            #send_discord_message (discord_url, symbol, symbol, discord_message)

            
            if ( len ( advice ) > 0):
                message += symbol + ' # ' + ", ".join( advice )
                strategy.append ( symbol + ' # ' + ", ".join( advice ) )
                print ( message )

            #print ( "             SupRes   [%s] ---> %s" % ( symbol, sr ( symbol ) ) )
            #print ( "             Fibona   [%s] ---> %s" % ( symbol, fibonacci ) )
            #print ( "             ATR_band [%s] ---> (LOW %.2f, %.2f%% away )   CUR %s   (MAX %.2f, %.2f%% away)" % ( symbol, atr_band_lower, 100 - ( atr_band_lower * 100 / price ), price_string, atr_band_higher, 100 - ( price * 100 / atr_band_higher  ) ) )

            print('--------------------------------------------------------------------')
            time.sleep(2)
            
    #####  BUY / SELL  list  #####
    if ( len ( strategy ) > 0):
        print ("\n\nSTRATEGY results:\n" )
        for line in strategy:
            print ( line )



