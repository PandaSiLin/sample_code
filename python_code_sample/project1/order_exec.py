import MetaTrader5 as mt5
from datetime import datetime
import keras
import pandas as pd 
import numpy as np

def open_trade(action, symbol, lot, sl_points, tp_points, deviation):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
    '''
    # prepare the buy request structure
    symbol_info = mt5.symbol_info(symbol)

    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
        slval = round(price - sl_points * 0.0001,5)
        tpval = round(price + tp_points * 0.0001,5)
    elif action =='sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
        slval = round(price + sl_points * 0.0001,5)
        tpval = round(price - tp_points * 0.0001,5)
    point = mt5.symbol_info(symbol).point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": slval,
        "tp": tpval,
        "deviation": deviation,
        "magic": ordernum,
        "comment": "sent by python",
        "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    # send a trading request
    result = mt5.order_send(request)  

    return result, request

# set up connection
mt5.initialize()
mt5.login(login = 35146240, password = "xt7lixru")

symbol = "EURUSD" 
positions=mt5.positions_get(symbol = symbol)

if not positions:

    # load trained model
    model = keras.models.load_model("LSTM_eurusd.model")

    # set parameters
    lot = 0.1
    tp_points = 20
    sl_points = 20
    deviation = 5
    ordernum = 10001

    # Get quote
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    rates = pd.DataFrame(rates)
    rates['time']=pd.to_datetime(rates['time'], unit='s')

    # Forecast
    X = rates.loc[-1:0,['close']].values
    Xpred = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    Yhat = model.predict(Xpred)
    xhat = X[0,0]
    yhat = Yhat[0,0]


    # open order only if there is no open position
    if positions == None:

        # open order only if it can have more than tp
        if abs(yhat - xhat) >= tp_points * 0.0001:

            if yhat >= xhat:
                action = "buy"
            else:
                action = "sell"
            
            open_trade(action, symbol, lot, sl_points, tp_points, deviation)
        
        else:
            print("estimated closing is: ", yhat)

    print("yhat :", yhat)