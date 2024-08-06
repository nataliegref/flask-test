from flask import Flask, jsonify, request
from flask_cors import CORS 

import pandas as pd 
from scipy.stats import t
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/model": {"origins": "http://localhost:3000"}}) # add address for production

def insert_missing_days(df):
    full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')

    df_full = df.reindex(full_time_index)

    return df_full

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['minutes'] = df.index.hour * 60 + df.index.minute
    return df

def add_lags(df):
    df = df.copy()
    target_map = df['flow'].to_dict()
    df['lag1d'] = (df.index - pd.Timedelta('1 days')).map(target_map)
    df['lag3d'] = (df.index - pd.Timedelta('3 days')).map(target_map)
    df['lag7d'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    return df

def process_raw_data(raw_data):
    df = pd.DataFrame(raw_data)
    df = df.set_index('time')
    df.flow = pd.to_numeric(df.flow)
    df.index = pd.to_datetime(df.index)
    df = insert_missing_days(df)
    df = create_features(df)
    df = add_lags(df)
    
    return df

def fill_nan(df):
    df = df.copy()
    bool_nulls = df.isnull().any()
    null_columns = bool_nulls[bool_nulls].index.tolist()
    for item in null_columns:
        df.loc[:, item] = df[item].ffill()
    for item in null_columns:
        df.loc[:, item] = df[item].bfill()
    return df

def prepare_ML_data(df):
    cutoff_date = df.index[-1] - pd.Timedelta('7 days')
    start_date = cutoff_date - pd.Timedelta('7 days')

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'minutes', 'lag1d', 'lag3d', 'lag7d']
    TARGET = 'flow'

    X_test_w_null = df.loc[df.index >= cutoff_date][FEATURES]
    X_predictions = fill_nan(X_test_w_null)

    df_wo_null = df.dropna()

    train = df_wo_null.loc[(df_wo_null.index >= start_date) & (df_wo_null.index < cutoff_date)]
    test = df_wo_null.loc[df_wo_null.index >= cutoff_date]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]


    return X_train, y_train, X_test, y_test, X_predictions

def format_time_series(series):
    dates = []
    values = []
    for index, value in series.items():
        dates.append(str(index))
        values.append(value)
    return [dates, values]

def LR_model(X_train, y_train, X_test, X_pred):

    model = LinearRegression()
    model.fit(X_train, y_train)
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
    else:
        y_pred = [0]
        
    y_pred2 = model.predict(X_pred)
    
    residuals = y_train - model.predict(X_train)
    
    return y_pred, y_pred2, residuals

def calculate_confidence(residuals, predictions, lengthX, confidence_level = 0.95):

    std_error = np.std(residuals)

    alpha = 1 - confidence_level
    dof = lengthX - 1  # degrees of freedom
    t_stat = t.ppf(1 - alpha/2, dof)
    # t_stat = 1.96 #approximation

    lower_bound = predictions - t_stat * std_error
    upper_bound = predictions + t_stat * std_error

    
    return list(lower_bound), list(upper_bound)

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/about')
def about():
    return 'ML Time series prediction API'

@app.route("/model", methods=['POST'])
def run_model():
    data = request.get_json()

    if not data or 'rawData' not in data:
        return jsonify({'error': 'Invalid input, "rawData" is required'}), 400

    raw_data = data.get('rawData')
    time_series_df = process_raw_data(raw_data)

    X_train, y_train, X_test, y_test, X_predictions = prepare_ML_data(time_series_df)
    test_results, prediction_results, residuals = LR_model(X_train, y_train, X_test, X_predictions)
    prediction_results = pd.Series(prediction_results, index=X_predictions.index) 
    lower_bound, upper_bound = calculate_confidence(residuals, prediction_results, len(X_train))

    mse = mean_squared_error(test_results, y_test)

    all_ys = pd.concat([y_train, y_test])
    time_series = format_time_series(all_ys)
    predictions = format_time_series(prediction_results)

    bounds = [predictions[0], lower_bound, upper_bound]

    return jsonify({'forecast': predictions, 'timeSeries': time_series, 'bounds':bounds, 'mse':round(mse, 2)})



if __name__ == "__main__":
    app.run(port=8000) 