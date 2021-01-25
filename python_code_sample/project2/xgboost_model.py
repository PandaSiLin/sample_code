import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import preprocess


raw_data = pd.read_csv("Data/Price Prediction/combined_exchange_a.csv")
raw_df = preprocess.feature_engineering(raw_data).dropna()
raw_df.to_csv("output/raw_df.csv")

X = raw_df.drop(columns='mid_price_target', axis=1)
y = raw_df.mid_price_target
train_X, test_X, train_y, test_y = train_test_split(X.values, y, random_state=15, test_size=0.2)

# train_X = xgb.DMatrix(train_X.values)
# test_X = xgb.DMatrix(test_X.values)
print('start time: {}'.format(datetime.now()))


def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3],
        'subsample': [0.5, 0.8],
        'colsample_bytree': [0.5, 0.8],
        'n_estimators': [100, 200, 500, 1000],
        'objective': ['reg:squarederror']
    }

    xgb_model = xgb.XGBRegressor()

    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_tuning,
                           # scoring = 'neg_mean_absolute_error', #MAE
                           # scoring = 'neg_mean_squared_error',  #MSE
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

    gsearch.fit(X_train, y_train)

    return gsearch


best_tune = hyperParameterTuning(train_X, train_y)
#joblib.dump(best_tune, 'output/xgboost_tune.pkl')
print('end time: {}'.format(datetime.now()))

xgb_model = xgb.XGBRegressor(
    gamma=1,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=1000,
    subsample=0.8,
    random_state=15,
    objective='reg:squarederror'
)
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(train_X, train_y, verbose=False)

#xgb_model.save_model('output/xgb.model')


# Model tuning
# Grid of hyperparameters to search over
