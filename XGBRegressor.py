import pandas as pd

# Load the data
data = pd.read_csv('train.csv')

print(data.shape)

submission = pd.read_csv('sample_submission_joined.csv')
# Split the DataFrame into two based on 'Id' starting with 'P' and 'ST'
df_p = submission[submission['Id'].str.startswith('P')].copy()
df_st = submission[submission['Id'].str.startswith('ST')].copy()

# Function to extract necessary information from 'Id' column
def extract_info(row):
    parts = row['Id'].split(',')
    if row['Id'].startswith('P'):
        return parts[0], parts[1]
    elif row['Id'].startswith('ST'):
        return parts[0], parts[1]

# Apply the function to each row and create new columns
df_p[['PRODUCT', 'AREA_HYERARCHY1']] = df_p.apply(
    lambda row: pd.Series(extract_info(row)), axis=1)

# Apply the function to each row and create new columns for initial_dataset_st
df_st[['PH1', 'AREA_HYERARCHY2']] = df_st.apply(
    lambda row: pd.Series(extract_info(row)), axis=1)

df_st['AREA_HYERARCHY2'] = df_st['AREA_HYERARCHY2'].astype(float)
df_p['AREA_HYERARCHY1'] = df_p['AREA_HYERARCHY1'].astype(float)

# Drop duplicates and reset index for df_p and df_st
df_p_unique = df_p[['PRODUCT', 'AREA_HYERARCHY1']].drop_duplicates().reset_index(drop=True)
df_st_unique = df_st[['PH1', 'AREA_HYERARCHY2']].drop_duplicates().reset_index(drop=True)

# Filter 'data' DataFrame based on 'PRODUCT' and 'AREA_HYERARCHY1' for df_p
filtered_data_p = data.merge(df_p_unique, on=['PRODUCT', 'AREA_HYERARCHY1'], how='inner')
# Filter 'data' DataFrame based on 'PH1' and 'AREA_HYERARCHY2' for df_st
filtered_data_st = data.merge(df_st_unique, on=['PH1', 'AREA_HYERARCHY2'], how='inner')


# Function to preprocess the data for forecasting
def preprocess_data_week(initial_data):
    # copy the dataframe to ensure that I make no changes to the initial data
    data = initial_data.copy()
    data.sort_values(by=['PRODUCT', 'AREA_HYERARCHY1'], inplace=True)

    # Convert DATE column to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['AREA_HYERARCHY1'] = data['AREA_HYERARCHY1'].astype(int).astype(str)
    data['AREA_HYERARCHY2'] = data['AREA_HYERARCHY2'].astype(int).astype(str)

    # Aggregate by PRODUCT and AREA_HYERARCHY1 for weekly forecasting
    weekly_data = data.groupby(['PRODUCT', 'AREA_HYERARCHY1', pd.Grouper(key='DATE', freq='1W')])['QTY'].sum().reset_index()

    # Convert numeric columns to strings before concatenation

    # Add ID column for weekly data
    weekly_data['Id'] = weekly_data['PRODUCT'] + ',' + weekly_data['AREA_HYERARCHY1'] + ',' + (weekly_data.groupby(['PRODUCT', 'AREA_HYERARCHY1']).cumcount() + 1).astype(str)

    return weekly_data

# Preprocess the data
weekly_forecast_data = preprocess_data_week(filtered_data_p)


def preprocess_data_month(initial_data):
    # copy the dataframe to ensure that I make no changes to the initial data
    data = initial_data.copy()
    data.sort_values(by=['PH1', 'AREA_HYERARCHY2'], inplace=True)
    # Convert DATE column to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['AREA_HYERARCHY1'] = data['AREA_HYERARCHY1'].astype(int).astype(str)
    data['AREA_HYERARCHY2'] = data['AREA_HYERARCHY2'].astype(int).astype(str)

    # Aggregate by PH1 and AREA_HYERARCHY2 for monthly forecasting
    monthly_data = data.groupby(['PH1', 'AREA_HYERARCHY2', pd.Grouper(key='DATE', freq='M')])['QTY'].sum().reset_index()

    # Add ID column for monthly data
    monthly_data['Id'] = monthly_data['PH1'] + ',' + monthly_data['AREA_HYERARCHY2'] + ',' + (monthly_data.groupby(['PH1', 'AREA_HYERARCHY2']).cumcount() + 1).astype(str)

    return monthly_data

monthly_forecast_data = preprocess_data_month(filtered_data_st)


from itertools import product
from datetime import timedelta
from datetime import timedelta

# Pad forecast with zeros for the required 12 weeks
def pad_forecast_week(initial_data, weeks=12):

    data = initial_data.copy()
    data.sort_values(by=['PRODUCT', 'AREA_HYERARCHY1'], inplace=True)
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['AREA_HYERARCHY1'] = data['AREA_HYERARCHY1'].astype(int).astype(str)
    data['AREA_HYERARCHY2'] = data['AREA_HYERARCHY2'].astype(int).astype(str)

    last_date = data['DATE'].max()
    end_date = last_date + timedelta(weeks=weeks+1)
    dates = pd.date_range(start=last_date + timedelta(weeks=1), end=end_date, freq='1W')

    # Get unique combinations of 'PRODUCT' and 'AREA_HYERARCHY1'
    unique_ids = data[['PRODUCT', 'AREA_HYERARCHY1']].drop_duplicates()

    # Create an empty list to store the padded data
    padded_data = []

    # Iterate over each unique combination of 'PRODUCT' and 'AREA_HYERARCHY1'
    for _, row in unique_ids.iterrows():
        product_id, area_id = row['PRODUCT'], row['AREA_HYERARCHY1']

        # Iterate over each date
        for i, date in enumerate(dates, start=1):
            padded_data.append([f"{product_id},{area_id},{i}", 0.0])

    # Convert the list of lists to a DataFrame
    padded_data_df = pd.DataFrame(padded_data, columns=['Id', 'QTY'])

    # Reset the index and drop the existing index column
    padded_data_df.reset_index(drop=True, inplace=True)

    return padded_data_df


padded_weekly_forecast_data = pad_forecast_week(filtered_data_p)

# Pad forecast with zeros for the required 3 months
def pad_forecast_month(initial_data, months=3):

    data = initial_data.copy()
    data.sort_values(by=['PH1', 'AREA_HYERARCHY2'], inplace=True)
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['AREA_HYERARCHY1'] = data['AREA_HYERARCHY1'].astype(int).astype(str)
    data['AREA_HYERARCHY2'] = data['AREA_HYERARCHY2'].astype(int).astype(str)

    last_date = data['DATE'].max()
    end_date = last_date + pd.DateOffset(months=months+1)
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end=end_date, freq='1M')

    # Get unique combinations of 'PRODUCT' and 'AREA_HYERARCHY1'
    unique_ids = data[['PH1', 'AREA_HYERARCHY2']].drop_duplicates()

    # Create an empty list to store the padded data
    padded_data = []

    # Iterate over each unique combination of 'PH1' and 'AREA_HYERARCHY1'
    for _, row in unique_ids.iterrows():
        ph_id, area_id = row['PH1'], row['AREA_HYERARCHY2']

        # Iterate over each date
        for i, date in enumerate(dates, start=1):
            padded_data.append([f"{ph_id},{area_id},{i}", 0.0])

    # Convert the list of lists to a DataFrame
    padded_data_df = pd.DataFrame(padded_data, columns=['Id', 'QTY'])

    return padded_data_df


padded_montly_forecast_data = pad_forecast_month(filtered_data_st)

result = pd.concat([padded_weekly_forecast_data,padded_montly_forecast_data])

unique_series = weekly_forecast_data.groupby(['PRODUCT', 'AREA_HYERARCHY1'])
week_predictions = pd.DataFrame(columns=['Id', 'QTY'])


from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster

import logging

# Configure logging
logging.basicConfig(filename='error_xgb.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load checkpoint
def load_checkpoint():
    try:
        with open('checkpoint_xgb.txt', 'r') as checkpoint_file:
            return int(checkpoint_file.read())
    except FileNotFoundError:
        return 0

# Function to save checkpoint
def save_checkpoint(counter):
    with open('checkpoint_xgb.txt', 'w') as checkpoint_file:
        checkpoint_file.write(str(counter))

# Load last saved checkpoint
counter = load_checkpoint()


for i, ((product, area), data) in enumerate(unique_series, start=1):
  if i <= counter:
    continue  # Skip already processed entries
  logging.info('------------------------------------------------')
  logging.info(f"Processing {product}, {area}, {i}")
  print('------------------------------------------------')
  print(f"Processing {product}, {area}, {i}")
  product_id = product
  area_hyerarchy1 = area
  tmseries = data.copy()

  try:
    columns_to_drop = ['PRODUCT', 'AREA_HYERARCHY1', 'Id']
    tmseries.drop(columns=columns_to_drop, inplace=True)
    tmseries.set_index('DATE', inplace=True) 
    tmseries = tmseries['QTY']  # Extract 'QTY' column
    
    # Reset index to RangeIndex
    tmseries.reset_index(drop=True, inplace=True)
    

    forecaster = ForecasterAutoreg(
                  regressor = XGBRegressor(
                                  enable_categorical = True,
                                  random_state = 123
                              ),
                  lags = 5
              )
    lags_grid = [1, 2, 3, 4, 5, 6, 7]

    # Regressor hyperparameters search space
    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 400, 1200, step=100),
            'max_depth'       : trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 1),
            'subsample'       : trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'gamma'           : trial.suggest_float('gamma', 0, 1),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1),
        } 
        return search_space

    results_search, frozen_trial = bayesian_search_forecaster(
                                      forecaster         = forecaster,
                                      y                  = tmseries,
                                      search_space       = search_space,
                                      lags_grid          = lags_grid,
                                      steps              = 30,
                                      refit              = False,
                                      metric             = 'mean_absolute_error',
                                      initial_train_size = round(2*len(tmseries)/3),
                                      fixed_train_size   = False,
                                      n_trials           = 30,
                                      random_state       = 123,
                                      return_best        = True,
                                      n_jobs             = 'auto',
                                      verbose            = False,
                                      show_progress      = True
                                  )
    predictions = forecaster.predict(12)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.reset_index(drop=True, inplace=True)
    prediction_df['Id'] = product + ',' + area + ',' + (prediction_df.index + 1).astype(str)
    prediction_df = prediction_df.rename(columns={'pred': 'QTY'})
    
  except Exception as e:
      print(f"############### An error occurred: {e}")
      # Use the mean of tmseries as predictions if an error occurs
      mean_value = tmseries.mean()
      prediction_df = pd.DataFrame({'QTY': [mean_value] * 12})
      prediction_df['Id'] = product_id + ',' + area_hyerarchy1 + ',' + (prediction_df.index + 1).astype(str)

  # Open CSV file in append mode and write predictions
  with open('week_predictions_xgboost_grid.csv', 'a') as f:
      prediction_df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
  # Save checkpoint in every iteration
  save_checkpoint(i)


unique_series = monthly_forecast_data.groupby(['PH1', 'AREA_HYERARCHY2'])
month_predictions = pd.DataFrame(columns=['Id', 'QTY'])



# Configure logging
logging.basicConfig(filename='error_xgb2.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load checkpoint
def load_checkpoint2():
    try:
        with open('checkpoint_xgb2.txt', 'r') as checkpoint_file:
            return int(checkpoint_file.read())
    except FileNotFoundError:
        return 0

# Function to save checkpoint
def save_checkpoint2(counter):
    with open('checkpoint_xgb2.txt', 'w') as checkpoint_file:
        checkpoint_file.write(str(counter))

# Load last saved checkpoint
counter = load_checkpoint2()

for i, ((ph1, area), data) in enumerate(unique_series, start=1):
  if i <= counter:
    continue  # Skip already processed entries
  logging.info('------------------------------------------------')
  logging.info(f"Processing {ph1}, {area}, {i}")
  print('------------------------------------------------')
  print(f"Processing {ph1}, {area}, {i}")
  tmseries = data.copy()
  try:
    columns_to_drop = ['PH1', 'AREA_HYERARCHY2', 'Id']
    tmseries.drop(columns=columns_to_drop, inplace=True)
    tmseries.set_index('DATE', inplace=True) 
    tmseries = tmseries['QTY']  # Extract 'QTY' column
    
    # Reset index to RangeIndex
    tmseries.reset_index(drop=True, inplace=True)
    

    forecaster = ForecasterAutoreg(
                  regressor = XGBRegressor(
                                  enable_categorical = True,
                                  random_state = 123
                              ),
                  lags = 5
              )
    lags_grid = [1, 2, 3, 4, 5, 6, 7]

    # Regressor hyperparameters search space
    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 400, 1200, step=100),
            'max_depth'       : trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 1),
            'subsample'       : trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'gamma'           : trial.suggest_float('gamma', 0, 1),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1),
        } 
        return search_space

    results_search, frozen_trial = bayesian_search_forecaster(
                                      forecaster         = forecaster,
                                      y                  = tmseries,
                                      search_space       = search_space,
                                      lags_grid          = lags_grid,
                                      steps              = 30,
                                      refit              = False,
                                      metric             = 'mean_absolute_error',
                                      initial_train_size = round(2*len(tmseries)/3),
                                      fixed_train_size   = False,
                                      n_trials           = 30,
                                      random_state       = 123,
                                      return_best        = True,
                                      n_jobs             = 'auto',
                                      verbose            = False,
                                      show_progress      = True
                                  )
    predictions = forecaster.predict(3)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.reset_index(drop=True, inplace=True)
    prediction_df['Id'] = ph1 + ',' + area + ',' + (prediction_df.index + 1).astype(str)
    prediction_df = prediction_df.rename(columns={'pred': 'QTY'})
    
  except Exception as e:
      print(f"############### An error occurred: {e}")
      # Use the mean of tmseries as predictions if an error occurs
      mean_value = tmseries.mean()
      prediction_df = pd.DataFrame({'QTY': [mean_value] * 3})
      prediction_df['Id'] = ph1 + ',' + area + ',' + (prediction_df.index + 1).astype(str)

  # Open CSV file in append mode and write predictions
  with open('month_predictions_xgboost_grid.csv', 'a') as f:
      prediction_df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
  # Save checkpoint in every iteration
  save_checkpoint2(i)