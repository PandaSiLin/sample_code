import pandas as pd
import datetime
import re
import os
import datatable


def feature_engineering(data):
    # data = pd.read_csv("Data/Price Prediction/exchange-a-orderbook 2020-08-01.csv")

    data['receiveTs'] = pd.to_datetime(data['receiveTs'])

    # compute mid_price
    data = data.assign(
        mid_price=lambda x: (x.Pa_1 + x.Pa_2) / 2,
        hrs=lambda x: x.receiveTs.dt.hour,
        mins=lambda x: x.receiveTs.dt.minute
    )

    # prepare seconds level mid_price
    mid_price_in_secs = data.assign(date_in_secs=data.receiveTs.dt.strftime("%Y-%m-%d %H:%M:%S")) \
        .groupby(['date_in_secs']) \
        .agg({'mid_price': 'min'})

    # get lead 5s time as target;  get lag 1s-4s mid_prices as features
    data = data.assign(
        target_time=lambda x: x.receiveTs.apply(lambda y: y + datetime.timedelta(seconds=5)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
        lag_1s_time=lambda x: x.receiveTs.apply(lambda y: y - datetime.timedelta(seconds=1)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
        lag_2s_time=lambda x: x.receiveTs.apply(lambda y: y - datetime.timedelta(seconds=2)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
        lag_3s_time=lambda x: x.receiveTs.apply(lambda y: y - datetime.timedelta(seconds=3)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
        lag_4s_time=lambda x: x.receiveTs.apply(lambda y: y - datetime.timedelta(seconds=4)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
    ) \
        .join(mid_price_in_secs, how='left', on='target_time', rsuffix='_target') \
        .fillna(method='bfill') \
        .join(mid_price_in_secs, how='left', on='lag_1s_time', rsuffix='_lag1') \
        .join(mid_price_in_secs, how='left', on='lag_2s_time', rsuffix='_lag2') \
        .join(mid_price_in_secs, how='left', on='lag_3s_time', rsuffix='_lag3') \
        .join(mid_price_in_secs, how='left', on='lag_4s_time', rsuffix='_lag4') \
        .fillna(method='ffill')

    # change Py_x (y=[a,b], x=[2:11]) as delta: Py_x = Py_x - Py_1
    for feature in range(2, 11):
        pa_col = 'Pa_' + str(feature)
        data[pa_col] = data[pa_col] - data.Pa_1
        pb_col = 'Pb_' + str(feature)
        data[pb_col] = data[pb_col] - data.Pb_1

    # change lag mid_prices as delta: mid_price_lagi = mid_price - mid_price_lagi
    for lag in range(1,4):
        lag_col = 'mid_price_lag' + str(lag)
        data[lag_col] = data.mid_price - data[lag_col]

    r = re.compile("(P|V|mid)")
    final_features = list(filter(r.match, data.columns)) + ['hrs', 'mins']
    final_data = data[final_features]

    return final_data


if False:
    file_names = [i for i in os.listdir('Data/Price Prediction/') if re.match('exchange-a.*', i)]
    raw_data = pd.concat([datatable.fread('Data/Price Prediction/' + f).to_pandas() for f in file_names])
    raw_data.to_csv("Data/Price Prediction/combined_exchange_a.csv")

