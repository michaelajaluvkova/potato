import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from math import sqrt
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import time
from matplotlib import pyplot as plt 
from sklearn.model_selection import ParameterGrid
from prophet.diagnostics import cross_validation, performance_metrics
import gc
import plotly
import plotly.graph_objects as go
import plotly.express as px
from multiprocessing import Pool, cpu_count
import time
from typing import List, Optional, Type, Any
from sklearn.preprocessing import StandardScaler
import datetime as dt
from sklearn.linear_model import Ridge
import random
import boto3
import json
import psycopg2 as pg
import psycopg2.extras as pge
import pytz
import os

today = pd.Timestamp.today().normalize()
today_naive = today.to_pydatetime()
yesterday = today - pd.DateOffset(days=1)
three_days_ago = today - pd.DateOffset(days=3)
tomorrow = today + pd.DateOffset(days=1)

def choose_metrics(df):
    """
    This function takes a DataFrame and sorts it by 'date' in descending order.
    Then, it takes the unique values from the 'METRICS' column and selects the first XX unique METRICS.
    Finally, it filters the original DataFrame to only include rows with those XX unique metrics.
    It is useful when you have too big dataframe and want to make calculations faster.

    Parameters:
        df: The DataFrame to be filtered and sorted.

    Returns:
        filtered_df: A DataFrame containing only the rows with the chosen metrics.
    """
    df = df.sort_values('date', ascending=False)
    unique_metrics = df['METRICS'].unique()
    chosen_metrics = unique_metrics[0:80] # XX is now set on '80' but can be changed in no time
    filtered_df = df[df['METRICS'].isin(chosen_metrics)]
    return filtered_df

df2 = pd.read_csv('/opt/ml/processing/input/kpi_report_new.csv', usecols=['country', 'warehouse_id', 'year', 'puf', 'METRICS', 'VALUE', 'date', 'Recommendations', 'orders', 'day', 'moving_average', 'moving_sd'])
df2['date'] = pd.to_datetime(df2['date'])
df2['month'] = df2['date'].dt.month
df2 = df2.sort_values(by='date', ascending=False)
# df2 = df2[(df2['country'] == 'cz')]

# It is neccessary to filter the data so there are only combinations with minimum rows of 24 (because of shift func's purposes)
grouped = df2.groupby(['country', 'warehouse_id', 'METRICS']).transform('count')
df2 = df2[grouped['orders'] >= 24]  # 'orders' is just an arbitrary column to get the count from the transform


def reduce_mem_usage(df, verbose=True):
    """
    This function optimizes the memory usage of a DataFrame by downcasting the numeric types.
    It iterates through each numeric column and downcasts its datatype to the smallest
    possible numeric type that can fit the data without losing information.

    Parameters:
        df: The DataFrame whose memory usage is to be reduced.
        verbose (bool): If True, prints the reduction in memory usage. Defaults to True.

    Returns:
        df: A DataFrame with the same data, but using less memory.

    Example:
        Original memory usage might be 50 MB. By calling this function, the memory
        usage could be reduced to 30 MB, a reduction of 40%.
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2 # convert to MB

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

reduce_mem_usage(df2)

def lin_reg(df):
    """
Performs linear regression on data to predict values for each unique combination of warehouse_id, country and METRICS.

Parameters:
    df: The input DataFrame containing data. The DataFrame should contain columns like 'warehouse_id', 'country', 'METRICS', 'VALUE', and 'date'. The 'date' column should be in datetime format.

Returns:
    models (dict): A dictionary containing trained linear regression models for each unique combination of 'warehouse_id', 'country', and 'METRICS'. Each model is associated with the corresponding training data, RMSE, bounds, and date information.

    results_df: A DataFrame containing the results of the linear regression for each unique combination. It includes the following columns:
        - 'date': The date of the prediction.
        - 'warehouse_id': The warehouse ID.
        - 'country': The country where the warehouse is located.
        - 'METRICS': The specific metric being predicted.
        - 'predicted_value': The predicted value for the specified metrics.
        - 'real_value': The actual known value.
        - 'difference': The difference between the predicted and real values.
        - 'lower_bound': The lower bound of the confidence interval for the prediction.
        - 'upper_bound': The upper bound of the confidence interval for the prediction.
        - 'is_outlier': A boolean value indicating if the prediction is considered an outlier.
        - 'rmse': The Root Mean Square Error (RMSE) for the model. So far unused.
        - 'model_type': A string indicating the type of model used ('lin_reg' in this case).

Note:
    The function performs extensive preprocessing of the data, creates lag variables, and filters records based on certain conditions. It also prints warnings and information about the status of processing, including time estimates and outlier detection messages.

Example:
    models, results_df = lin_reg(df)
    """
    df['month'] = df['date'].dt.month
    np.random.seed(42) #setting random seed so the results stays the same
    models = {}
    results_list = []
    combinations = df[['warehouse_id', 'country', 'METRICS']].drop_duplicates().values
    today = pd.Timestamp.today().normalize()
    today_naive = today.to_pydatetime()
    yesterday = today - pd.DateOffset(days=1)
    three_days_ago = today - pd.DateOffset(days=3)
    tomorrow = today + pd.DateOffset(days=1)
    combination_sizes = df.groupby(['warehouse_id', 'country', 'METRICS']).size()
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df = df.dropna(subset=['VALUE']).astype({'VALUE': np.float32})
    print(combination_sizes)
    filtered_df = df.groupby(['warehouse_id', 'country', 'METRICS']).filter(lambda x: len(x) >= 24)
    unique_combinations = set(filtered_df[['warehouse_id', 'country', 'METRICS']].itertuples(index=False))
    
    # counts remaining time until finishing
    start_time = time.time()
    iterations_count = 0
    df['date'] = pd.to_datetime(df['date'])

    # important for loop where all importat stuff happens
    for warehouse_id, country, metrics in unique_combinations:
        combination_size = combination_sizes.loc[(warehouse_id, country, metrics)]
        data = df[(df['warehouse_id'] == warehouse_id) & (df['country'] == country) & (df['date'] < yesterday) & (df['METRICS'] == metrics) & (df['orders'].notnull())]
        # rest of preprocessing and data training
        if not data.empty:
            data = data.sort_values('date')
            data = data.dropna(subset=['VALUE'])

            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            number = len(data['date'].unique())
            data_new = data.tail(number)  
            data_new['lag_1'] = data_new['VALUE'].shift(1)
            data_new['lag_7'] = data_new['VALUE'].shift(7)
            data_new['lag_14'] = data_new['VALUE'].shift(14)
            data_new['lag_21'] = data_new['VALUE'].shift(21)
            data_new = data_new.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'lag_21', 'VALUE'])
            validators = ['month'] + [col for col in ['lag_1', 'lag_7', 'lag_14', 'lag_21', 'VALUE'] if not data_new[col].isnull().all()]
            X = data_new[validators].drop(columns=['VALUE', 'month'])  # We remove 'VALUE' as it's our target variable
            y = data_new[validators].drop(columns=['lag_1', 'lag_7', 'lag_14', 'lag_21', 'month'])  # We columns from X since it is our predictors
            X_days = data_new[['lag_1', 'lag_7', 'lag_14', 'lag_21']].tail(3)
            dates = data['date']
            
            if len(X) == 0 or len(y) == 0:
                print(f"Insufficient data for warehouse_id {warehouse_id}, country {country}, metrics {metrics}")
                continue

            model = LinearRegression().fit(X, y) # fitting the linreg model
            
            prediction = model.predict(X_days) # creating prediction
            residuals = y - model.predict(X) # computing residuals for CI and RMSE
            std_residuals = np.std(residuals)
            std_residuals_array = std_residuals.values

            # Ensure both arrays are 1D and have the same length
            min_length = min(len(std_residuals_array), len(prediction))
            std_residuals_array = std_residuals_array[:min_length]
            prediction = prediction[:min_length]

            lower_bounds = prediction - 1.96 * std_residuals_array
            upper_bounds = prediction + 1.96 * std_residuals_array

            y_array = y.iloc[-3:, 0].values # array with rela values

            # Ensure both arrays are 1D and have the same length
            min_length = min(len(y_array), len(prediction))
            y_array = y_array[:min_length]
            prediction = prediction[:min_length]
            diffs = y_array - prediction

            if np.any(lower_bounds == 0):
                print(f"Warning: The lower bounds for warehouse_id {warehouse_id}, country {country}, and metrics {metrics} hit zero. This could indicate an issue with the data.")

            # computing is_outlier column
            is_real_outlier = (y_array > upper_bounds) | (y_array < lower_bounds)
            is_pred_outlier = (prediction > upper_bounds) | (prediction < lower_bounds)
            is_outlier = np.where(np.isnan(y_array), is_pred_outlier, is_real_outlier)
            
            # computing rmse
            rmse = [np.sqrt(mean_squared_error(y_array[-3:], prediction[:, 0])) if date.date() < today.date() else np.nan for date in [today_naive - pd.DateOffset(days=1), today_naive]]
            rmse = rmse[0]
            real_value = [y_i if date.date() < today.date() else np.nan for y_i, date in zip(y[-3:].values, [today_naive - pd.DateOffset(days=1), today_naive])],

            # creating dataframe with all of the results and important columns
            result = pd.DataFrame({
            'date': [yesterday],
            'warehouse_id': warehouse_id,
            'country': country,
            'METRICS': metrics,
            'predicted_value': prediction.tolist(),
            'real_value': [y_i if date.date() < today.date() else np.nan for y_i, date in zip(y[-3:].values, [today_naive - pd.DateOffset(days=1)])],
            'difference': diffs.tolist(),
            'lower_bound': lower_bounds.tolist(),
            'upper_bound': upper_bounds.tolist(),
            'is_outlier': is_outlier.tolist(),
            'rmse': rmse,
            'model_type': 'lin_reg'
        })
            # commputing remaining time
            iterations_count += 1
            elapsed_time = time.time() - start_time
            average_time_per_iteration = elapsed_time / iterations_count
            remaining_iterations = len(unique_combinations) - iterations_count
            estimated_time_remaining = average_time_per_iteration * remaining_iterations
            minutes, seconds = divmod(estimated_time_remaining, 60)

            print(f"Estimated time remaining: {int(minutes)} minutes and {int(seconds)} seconds.")
            contains_outliers = result['is_outlier'].apply(any)
            if contains_outliers.any():
                print(f"For warehouse_id {warehouse_id}, country {country}, metrics {metrics}: outliers detected.")

            result.loc[result['date'] >= today, ['rmse', 'real_value', 'difference']] = np.nan
            result.sort_index(inplace=True)
            results_list.append(result)
            models[(warehouse_id, country, metrics)] = {"model": model, "X": X, "y": y, "rmse": rmse, "lower_bound": lower_bounds, "upper_bound": upper_bounds, "date": dates}
        else:
            print(f"df empty for warehouse_id {warehouse_id}, country {country}, metrics {metrics}")

    if not results_list:
        print("results_list is empty. No result DataFrame was created.")
        return models, None

    results_df = pd.concat(results_list)
    print("Number of rows in the linear regression result dataframe:", len(results_df))

    return models, results_df

# executing linear regression
_, results_linreg = lin_reg(df2)
if results_linreg is not None:
    print(results_linreg.head())
    results_linreg.to_csv('results_linreg.csv')
    print("Results for lin_reg have been saved to a CSV file.")
    del _, results_linreg
    gc.collect()
else:
    print("lin_reg did not return a valid DataFrame. No CSV file was created.")

def prophet_model(df):
    """
    Time-serie value forecasting  using a Ridge regression model with several lagged values as features.

    Parameters:
    df: A DataFrame containing the data to be forecasted. It should contain the following columns:
        - 'date': The date of the observation.
        - 'VALUE': The target variable.
        - 'warehouse_id': Identifier for the warehouse.
        - 'country': The country associated with the observation.
        - 'METRICS': The specific metric being observed.
        Will be taken from KEBOOLA Sniffer flow.
        
    Returns:
    models_prophet (dict): A dictionary containing the trained Ridge models for each unique combination of 'warehouse_id', 'country', and 'METRICS'.
    final_forecast (DataFrame): A DataFrame containing the forecasts, including predicted values, confidence intervals, and additional information such as outlier detection, rmse, etc.

    Notes:
    - Make sure to import all required libraries such as pandas, numpy, and scikit-learn (Ridge model).
    - The function filters out any group with fewer than 24 records.
    - Also make sure you have important SklearnModel class.
    - The function saves the final forecast to a CSV file if it is not empty.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'date': 'ds', 'VALUE': 'y'}, inplace=True)
    print(df.columns)
    
    # Pre-creating importat lists and dicts
    all_forecast = []
    models_prophet = {}
    forecast_df = []
    results_list = []
    combinations = df[['warehouse_id', 'country', 'METRICS']].drop_duplicates().values
    yesterday = today - pd.DateOffset(days=1)

    total_combinations = len(combinations)
    counter = 0
    start_time = time.time()
    dates = df['ds']
    
    # Filtering combinations with lesser than 24 rows.
    filtered_df = df.groupby(['warehouse_id', 'country', 'METRICS']).filter(lambda x: len(x) >= 24) 
    combination_sizes = filtered_df.groupby(['warehouse_id', 'country', 'METRICS']).size()
    unique_combinations = set(filtered_df[['warehouse_id', 'country', 'METRICS']].itertuples(index=False))
    print("Unique Combinations:", unique_combinations)
    
    # important for loop where all important stuff happens
    for warehouse_id, country, METRICS in unique_combinations:
        print(df.columns) # pre-problem checking
        data = filtered_df[(filtered_df['warehouse_id'] == warehouse_id) & (filtered_df['country'] == country) & (filtered_df['ds'] <= yesterday) & (filtered_df['METRICS'] == METRICS) & (filtered_df['orders'].notnull())] # because of weekends
        if not data.empty:
            # Creating shift functions for 1 day before, 7 days before, 14 days before and 21 days before
            # it is something like moving average, the final lag columns contains value of that concrete combination of X days before
            data = data.sort_values('ds')
            data['lag_1'] = data['y'].shift(1)
            data['lag_7'] = data['y'].shift(7)
            data['lag_14'] = data['y'].shift(14)
            data['lag_21'] = data['y'].shift(21)
            data = data.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'lag_21', 'y'])
            X = data[['lag_1', 'lag_7', 'lag_14', 'lag_21', 'ds', 'y']]
            y = data['y']

            counter += 1 # for computing remaining time until finishing
            forecast = pd.DataFrame()
            
            # defining SklearnModel
            model = SklearnModel(
                    model=Ridge,
                    features=['lag_1', 'lag_7', 'lag_14', 'lag_21'],
                    yearly_seasonality_order=5,
                    yearly_seasonality_min_history=10,
                    weekly_seasonality_order=5,
                    weekly_seasonality_min_history=10,
                    trend=True,
                    normalize=True,
                    exponential_sample_weight=0.994, #0.994
                    params={'alpha': 0.05}
                )

            model.train(X) # model training
            prediction = model.predict(X) # model prediction
            residuals = y.values - prediction['yhat'] # computing residuals for CI and RMSE purposes
            std_residuals = np.std(residuals)
            
            # computing CI (confidence intervals)
            lower_bounds = prediction['yhat'] - 1.96 * std_residuals
            upper_bounds = prediction['yhat'] + 1.96 * std_residuals
            diffs = y[-3:] - prediction['yhat'] # computing difference between real value and predicted value

            yhat_array = prediction['yhat'].values
            yhat_array = prediction['yhat'].values[-3:] # predicted value
            
            upper_bounds_array = np.array(upper_bounds[-3:])
            lower_bounds_array = np.array(lower_bounds[-3:])
            print('lower_bounds:', lower_bounds)
            
            # Predicting whether the value is outlier or is not
            is_real_outlier = (y[-3:].values > upper_bounds[-3:]) | (y[-3:].values < lower_bounds[-3:])
            is_pred_outlier = (yhat_array > upper_bounds_array) | (yhat_array < lower_bounds_array)
            is_outlier = np.where(np.isnan(y[-3:]), is_pred_outlier, is_real_outlier)

            rmse = [np.sqrt(mean_squared_error(y[-3:].values, prediction['yhat'].values[-3:])) if date < today else np.nan for date in [today_naive - pd.DateOffset(days=1)]]

            prediction_filtered = prediction[prediction['ds'].dt.date == yesterday] # filtering only for yesterday's date
            prediction_dates = pd.date_range(start=yesterday, periods=0)
                        
            if not prediction_filtered.empty:
                yhat_value = prediction_filtered['yhat'].values[0]
            else:
                yhat_value = np.nan
                
            forecast = prediction_filtered
            
            # Now, assign the predictions to the corresponding dates
            for i in range(len(prediction)):
                forecast['yhat_lower'] = lower_bounds
                forecast['yhat_upper'] = upper_bounds

            data.rename(columns={'y': 'VALUE'}, inplace=True)
            if len(data) > 0:
                forecast['warehouse_id'] = warehouse_id
                forecast['country'] = country
                forecast['METRICS'] = METRICS
                forecast.rename(columns={'y': 'real_value'}, inplace=True)
                
                models_prophet[(warehouse_id, country, METRICS)] = model
                forecast['yhat'] = forecast['yhat']
                forecast['difference'] = forecast['yhat'] - forecast['real_value']
                forecast['value_for_outlier_detection'] = np.where(forecast['real_value'].isnull(), forecast['yhat'], forecast['real_value'])
                forecast['is_outlier'] = (forecast['value_for_outlier_detection'] < forecast['yhat_lower']) | (forecast['value_for_outlier_detection'] > forecast['yhat_upper'])

                forecast_filtered = forecast.dropna(subset=['real_value', 'yhat'])
                forecast['rmse'] = [rmse] * len(forecast)
                forecast['model_type'] = 'prophet'
                elapsed_time = time.time() - start_time
                avg_time_per_combination = elapsed_time / counter
                remaining_time = (total_combinations - counter) * avg_time_per_combination
                print(f"Processed combination: warehouse_id {warehouse_id}, country {country}, metrics {METRICS}. Progress: {counter/total_combinations*100:.2f}%. Estimated remaining time: {remaining_time/60:.2f} minutes.")
                all_forecast.append(forecast)
            else:
                print(f"df empty for warehouse_id {warehouse_id}, country {country}, metrics {METRICS}")
                continue
    if not forecast.empty:
        final_forecast = pd.concat(all_forecast)
        final_forecast.rename(columns={'ds': 'date', 'yhat': 'predicted_value', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound', 'METRICS': 'METRICS'}, inplace=True)
        print(final_forecast.columns)
        return models_prophet, final_forecast
    else:
        print("nothing here mate")
        
models, results_prophet = prophet_model(df2)
if results_prophet is not None:
    results_prophet.to_csv('results_prophet.csv')
    print("Results for prophet have been saved to a CSV file.")
    del models, results_prophet
    gc.collect()
else:
    print("prophet did not return a valid DataFrame. No CSV file was created.")
    
class ModelResultsProcessor:
    def __init__(self):
        self.today = pd.Timestamp.now().normalize().tz_localize(None)
        self.final_df_list = []
        with open('/opt/ml/processing/input/config/sniffer_json.json', 'r') as f:
            self.config = json.load(f)

    def print_unique_values(self, df, columns):
        """ Prints unique values of a column. Useful for continuous error checking."""
        for col in columns:
            print(f"{col} unique values: {df[col].unique()}")

    def filter_and_print(self, condition, message):
        filtered_len = len(self.filtered_df[condition])
        print(f"{message}: {filtered_len}")
    
    def load_kpi_report(self):
        """ Loads the main dataframe and select only important columns. """
        kpi_source = self.config['dataframes']['kpi_report']['source']
        usecols = self.config['dataframes']['kpi_report']['usecols']

        return pd.read_csv(kpi_source, usecols=usecols)

    def load_and_concat(self):
        """ Concatenates results from statistical prediction models. """
        linreg_source = self.config['dataframes']['results_linreg']['source']
        prophet_source = self.config['dataframes']['results_prophet']['source']
        
        self.results_linreg = pd.read_csv(linreg_source)
        self.results_prophet = pd.read_csv(prophet_source)
        self.results_df = pd.concat([self.results_linreg, self.results_prophet])
        
    def drop_unnecessary_columns(self):
        """ Drops unnecessary columns. """
        columns_to_drop = self.config['columns_to_drop']
        self.results_df.drop(columns=columns_to_drop, inplace=True)

    def preprocess_data(self):
        self.results_df['date'] = pd.to_datetime(self.results_df['date']).dt.tz_localize(None)
        self.results_df['date'] = pd.to_datetime(self.results_df['date'])

        self.results_df = self.results_df[self.results_df['date'] >= self.today - pd.Timedelta(days=1)]
        
        # Code to convert types and deal with NaN values        
        cols_to_float = self.config['cols_to_float']
        # cols_to_float = ['predicted_value', 'real_value', 'difference', 'lower_bound', 'upper_bound']
        for col in cols_to_float:
            self.results_df[col] = self.results_df[col].astype(str).replace('[^\d.]', '', regex=True)
            self.results_df[col] = pd.to_numeric(self.results_df[col], errors='coerce')
        self.results_df['is_outlier'] = self.results_df['is_outlier'].apply(lambda x: str(x).replace('[', '').replace(']', '').strip() if isinstance(x, str) else str(x))
    
    def postprocess_data(self):
        # Filtering and rounding
        self.results_df = self.results_df[self.results_df['lower_bound'] >= 0]
        self.results_df[['predicted_value', 'difference', 'lower_bound', 'upper_bound', 'rmse']] = self.results_df[['predicted_value', 'difference', 'lower_bound', 'upper_bound', 'rmse']].round(3)
        self.results_df['is_outlier'] = self.results_df['is_outlier'].map({'True': True, 'False': False}).astype(bool)
    
    def averaging(self):
        """ Takes results from both models, averages them and create new model type 'averaged'. """
        grouped_results = self.results_df.groupby(['warehouse_id', 'country', 'METRICS', 'model_type'])
        
        for (warehouse_id, country, METRICS, model_type), group in grouped_results:
            group_copy = group.copy()
            
            avg_predicted_value = group_copy['predicted_value'].mean()
            avg_lower_bound = group_copy['lower_bound'].mean()
            avg_upper_bound = group_copy['upper_bound'].mean()
            real_value = group_copy['real_value'].iloc[0]
            difference = group_copy['difference'].iloc[0]
            is_outlier = group_copy['is_outlier'].all()
            date = group_copy['date'].iloc[0]
            
            averaged_row = pd.DataFrame({
                'warehouse_id': [warehouse_id],
                'country': [country],
                'METRICS': [METRICS],
                'predicted_value': [avg_predicted_value],
                'lower_bound': [avg_lower_bound],
                'upper_bound': [avg_upper_bound],
                'model_type': ['averaged'],
                'is_outlier': [is_outlier],
                'rmse': [None],
                'difference': [difference],
                'real_value': [real_value],
                'date': [date],
                'Unnamed: 0': [None]
            })
            
            group_copy = group_copy.append(averaged_row, ignore_index=True)
            self.final_df_list.append(group_copy)

        self.final_df = pd.concat(self.final_df_list)

    def post_averaging(self):
        """ Processing after creating new average model."""
        self.final_df = pd.concat(self.final_df_list)
        self.final_df['date'] = pd.to_datetime(self.final_df['date'])
        self.print_unique_values(self.final_df, ['model_type', 'is_outlier'])

        self.final_df.sort_values('date', inplace=True)
        self.print_unique_values(self.final_df, ['date'])

        self.filtered_df = self.final_df[self.final_df['is_outlier'].astype(str) == 'True']
        self.filter_and_print(self.filtered_df['is_outlier'] == True, "length of the dataframe after outliering")
        self.print_unique_values(self.filtered_df, ['date', 'model_type', 'is_outlier'])

        self.filtered_df = self.filtered_df.drop("Unnamed: 0", axis=1).drop_duplicates()

        yesterday = self.today - pd.DateOffset(days=1)
        yesterday_df = self.filtered_df[self.filtered_df['date'] == yesterday].drop_duplicates(subset=['warehouse_id', 'country', 'METRICS'])

        self.df = self.load_kpi_report()
        self.df['date'] = pd.to_datetime(self.df['date']).dt.tz_localize(None)
        self.df['date'] = pd.to_datetime(self.df['date'])

        
        print("DF Columns:", self.df.columns)
        print("DF Date Types:", self.df['date'].dtypes)
        filterek = self.df[(self.df['date'] == yesterday) & (self.df['country'] == 'cz')]
        print("Filtered DF Length:", len(filterek))

        if len(filterek) > 0:
            unique_puf = filterek['puf'].unique()[0]
            print("Unique PUF:", unique_puf)
            yesterday_df['puf'] = unique_puf
        else:
            print("No matching records for the given date and country.")

        if not yesterday_df.empty:
            print(yesterday_df.head())
            print(len(yesterday_df))
            yesterday_df.to_csv('forecast_diffs.csv')
            print("yesterday df saved")
        else:
            print("No outliers detected for yesterday.")

    def save_results(self, path='final_results.csv'):
        self.final_df.to_csv(path)

if __name__ == '__main__':
    processor = ModelResultsProcessor()
    processor.load_and_concat()
    processor.drop_unnecessary_columns()
    processor.preprocess_data()
    processor.postprocess_data()
    processor.averaging()
    processor.post_averaging()
    processor.save_results()
    
class SnifferProcessor:
    def __init__(self, data_path='forecast_diffs.csv'):
        self.df = pd.read_csv(data_path)
    
    def add_day(self):
        self.df['day'] = pd.to_datetime(self.df['date']).dt.weekday + 1
                
    def create_image_from_df(self, df, filename):
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(4.5, 2)
        plt.savefig(filename, format='png', bbox_inches='tight')

    def image_creation(self):
        if os.path.exists('forecast_diffs.csv'):
            if not self.df.empty:
                filtered_data = self.serious_sniffer()
                # Re-creating table with sniffing anomaly results
                sniff_counts = filtered_data['Sniff'].value_counts().reset_index()
                sniff_counts['index'] = sniff_counts['index'].apply(lambda x: f'*{x}*')
                sniff_counts = sniff_counts.to_string(index=False, header=False)
                sniff_counts = sniff_counts.replace('|', '').strip()

                # Drop unwanted columns and format remaining ones
                filtered_data = filtered_data.drop(columns=['Unnamed: 0', 'is_outlier', 'puf', 'day', 'rmse'])
                
                # Convert columns to numeric
                columns_to_format = ['real_value', 'predicted_value', 'lower_bound', 'difference', 'upper_bound']
                for col in columns_to_format:
                    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
                filtered_data.dropna(subset=columns_to_format, inplace=True)
                
                for col in columns_to_format:
                    filtered_data[col] = filtered_data[col].apply(lambda x: '{:,.3f}'.format(x).replace(',', ' '))
                    
                filtered_data.rename(columns={'METRICS': 'metrics', 'Sniff': 'sniff'}, inplace=True)                
                filtered_data.rename(columns={'METRICS': 'metrics', 'Sniff': 'sniff'}, inplace=True)
                utc_time = pd.Timestamp.now(tz='UTC')
                local_time = utc_time.tz_convert('Europe/Prague')
                filtered_data['timestamp'] = local_time.strftime('%Y-%m-%d %H:%M:%S')

                # Finally, create the image
                self.create_image_from_df(filtered_data, 'image.png')
            else:
                print(" DataFrame is empty!")
        else:
            print("Oops! 'forecast_diffs.csv' doesn't exist!")
        return filtered_data, sniff_counts
    
    def serious_sniffer(self):
        self.add_day() 
        self.df['Sniff'] = ''

        for index, row in self.df.iterrows():
            real_value = row['real_value']
            upper_bound = row['upper_bound']
            lower_bound = row['lower_bound']
            
            if real_value > upper_bound:
                self.df.loc[index, 'Sniff'] = 'Metric is higher than expected.'
            elif real_value < lower_bound:
                self.df.loc[index, 'Sniff'] = 'Metric is lower than expected.'
        filtered_data = self.df.copy()
        
        return filtered_data
    
if __name__ == '__main__':
    sniffer_processor = SnifferProcessor()
    sniffer_processor.image_creation()


# def funny_sniffer(data):
#     """
#     The Fun-Loving Foodie of sniffers. 
#     Tired of those pesky alerts? This one recommends what you should have for lunch instead.
#     This sniffer's as useful as a rubber sword, but at least it's entertaining.
    
#     Editable: Feel free to change up the menu if you're on a diet or something.
#     """
#     df = pd.DataFrame(data)
#     day(df)
#     food_menu = ['first menu', 'second menu', 'third menu', '4th menu', 'try a speciality!', 'radsi jdi na bagetu', 'give gyros a try']
    
#     for index, row in df.iterrows():
#         random_menu = random.choice(food_menu)
#         df.loc[index, 'Sniff'] = random_menu

#     return df

# send_notification("We came up to the sniffer(), yay! Let's open champagne bottle pls.")
bot_token = ''
from os import path
import os

# This function sends the above created image-tables to the channel, with some emojis and accompanying messages
def send_notification(channel_id=''):
    """
    Purpose:
        Sends notifications and tables to a Slack channel.
                        
    Used tool:
        markdown language
                
    Parameters:
        channel_id (str): The ID of the channel to send the notifications. .
        name of the channel: slack-test-anomaly-detection
    
    Returns:
        Served-up stats, glorious images of table, and messages that may or may not contain emojis.

    Context:
        Every table is accompannied by message and emojis.
    
    Message edit:
        Feel like a god and edit text or emoji blocks to your heart's content.
        
    Notes:
        - It sends notifications, tables, and images to the specified Slack channel.
        - The function checks for the existence of image files before uploading them to Slack.
        - After uploading the images, they are deleted to prevent uploading outdated statistics.
        - No anomalies? Expect a message that's the equivalent of a digital hug.
    """
    client = WebClient(token=bot_token)
    channel = ''
    filtered_data, sniff_counts = sniffer_processor.image_creation()
    
    if path.exists('forecast_diffs.csv'):
        forecast_diffs = pd.read_csv('forecast_diffs.csv') 

    # Markdown language which makes it work
    try:
        # Posting initial comment before sending the tables and summaries itself
        initial_comment = {
            "type": "section", 
            "text": {"type": "mrkdwn", "text": "Good morning! I am The Sniffer and here I come with new fresh sniffing results. Enjoy!"}
        }

        response = client.chat_postMessage(
            channel=channel_id,
            blocks=[initial_comment]
        )
        print(response)
    except SlackApiError as e:
        print(f"Error posting initial comment: {e.response['error']}")
        
    
    if path.exists('forecast_diffs.csv'):
        if not forecast_diffs.empty and filtered_data is not None:
            if not filtered_data.empty:
                try:
                    # Send statistics
                    emoji_message = ":rohlik_gif: :dancing_banana:" # you can edit the emoji here, if you want to send other ones
                    blocks_content = [
                        {
                            # "type": "section", "text": {"type": "mrkdwn", "text": f"{sniff_counts} {emoji_message}"}
                            "type": "section", "text": {"type": "mrkdwn", "text": f"{sniff_counts} {emoji_message}"}

                        }
                    ]

                    response = client.chat_postMessage(
                        channel=channel_id,
                        blocks=blocks_content
                    )
                    print(response)

                    if path.exists('image.png'):

                        # Upload image after the statistics
                        response = client.files_upload(
                            channels=channel_id,
                            file="image.png"
                        )
                        print(response)
                        # os.remove('image.png') # Important! This will delete the image after posting, therefore it prevent uploading image with old statistics.
                    else:
                        no_predictions_message = "_I swear it worked yesterday._"  
                        blocks_content = [
                            {
                                "type": "section", "text": {"type": "mrkdwn", "text": no_predictions_message}
                            }
                        ]
                        response = client.chat_postMessage(
                            channel=channel_id,
                            blocks=blocks_content
                        )
                        print(response)


                except SlackApiError as e:
                    print(f"Error posting message: {e.response['error']}")

    else:
        result_message = "_Everything is fine, take a rest._"
        blocks_content = [
            {
                "type": "section", "text": {"type": "mrkdwn", "text": result_message}
            }
        ]

        try:
            response = client.chat_postMessage(
                channel=channel_id,
                blocks=blocks_content
            )
            print(response)
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}") 
            
sniffer_processor = SnifferProcessor()
send_notification()
