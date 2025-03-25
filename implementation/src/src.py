# imports

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn import set_config
set_config(transform_output='pandas')


### DEFINE THE LOG FUNCTION TRANSFORMER ###
def log_transform_df(X):
  return np.log1p(X)


### DEFINE GET_DATA FUNCTION ###

def get_data():
  # import data from csv's
  df_application = pd.read_csv('../data/application_train.csv')
  df_bureau = pd.read_csv('../data/bureau.csv')
  df_credit_card_balance = pd.read_csv('../data/credit_card_balance.csv')
  df_previous_application = pd.read_csv('../data/previous_application.csv')

  ### CLEANING DAYS_EMPLOYED ###
  def clean_days_employed(df_application):
    df_application['DAYS_EMPLOYED'] = df_application['DAYS_EMPLOYED'].replace(365243, 365)
    return df_application

  ### ENGINEERING NUM_CLOSED_BUREAU_CREDITS ###
  def engineer_num_closed_bureau_credits(df_application, df_bureau):
    # Feature Engineering from Bureau: num_closed_bureau_credits
    # get a count of how many closed ones there are per client
    bureau_num_closed_credits = pd.DataFrame(df_bureau[df_bureau['CREDIT_ACTIVE']=='Closed'].groupby('SK_ID_CURR')['CREDIT_ACTIVE'].count()).reset_index()
    bureau_num_closed_credits.columns = ['SK_ID_CURR','num_closed_bureau_credits']

    # add new feature to existing dataframe of engineered features
    df_application = pd.merge(df_application, bureau_num_closed_credits, on='SK_ID_CURR', how='outer')

    # fill in NaN with 0, because it means they have no closed credits with the Credit Bureau
    df_application['num_closed_bureau_credits'] = df_application['num_closed_bureau_credits'].fillna(0)
    return df_application


  ### ENGINEERING AVG_RATIO_BUREAU_CR_DEBT ###
  def engineer_avg_ratio_bureau_cr_debt(df_application, df_bureau):
    # calculate the ratio of debt to credit
    # add one to denominator to prevent dividing by 0
    df_bureau['Debt_Ratio'] = df_bureau['AMT_CREDIT_SUM_DEBT'] / (df_bureau['AMT_CREDIT_SUM'] + 1)

    # get the highest max overdue per client and save to dataframe
    ratio_debt_to_credit = pd.DataFrame(df_bureau.groupby('SK_ID_CURR')['Debt_Ratio'].mean()).reset_index()
    ratio_debt_to_credit.columns = ['SK_ID_CURR', 'avg_ratio_bureau_cr_debt']

    # add to existing engineered features dataframe
    df_application = pd.merge(df_application, ratio_debt_to_credit, on='SK_ID_CURR', how='outer')

    # fill in null with 0, because NaN indicates never prolonged a credit
    df_application['avg_ratio_bureau_cr_debt'] = df_application['avg_ratio_bureau_cr_debt'].fillna(0)
    return df_application


  ### ENGINEERING TTL_BUREAU_CC_LIMIT ###
  # total the AMT_CREDIT_SUM_LIMIT per client using groupby
  def engineer_ttl_bureau_cc_limit(df_application, df_bureau):
    total_bureau_cc_limit = pd.DataFrame(df_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].sum()).reset_index()
    total_bureau_cc_limit.columns = ['SK_ID_CURR', 'ttl_bureau_cc_limit']

    # add to engineered dataframe
    df_application = pd.merge(df_application, total_bureau_cc_limit, on='SK_ID_CURR', how='outer')
    return df_application


  ### ENGINEERING CC_AVG_CREDIT_USAGE_RATIO ###
  # calculate the average ratio of credit card balance to limit usage ratio
  def engineer_cc_avg_credit_usage_ratio(df_credit_card_balance):
    # create an df with SK_ID_CURR and SK_ID_PREV
    engineered_cc = df_credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].copy()
    engineered_cc
    # calculate the ratio for each row
    df_credit_card_balance['credit_usage_ratio'] = df_credit_card_balance['AMT_BALANCE'] / (df_credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'])
    df_credit_card_balance['credit_usage_ratio'] = df_credit_card_balance['credit_usage_ratio'].replace([np.inf, -np.inf], 0)
    df_credit_card_balance['credit_usage_ratio'] = df_credit_card_balance['credit_usage_ratio'].fillna(0)

    # calculate the average ratio by loan using groupby
    avg_usage_ratio_per_loan = pd.DataFrame(df_credit_card_balance.groupby('SK_ID_PREV')['credit_usage_ratio'].mean()).reset_index()
    avg_usage_ratio_per_loan.columns = ['SK_ID_PREV','cc_avg_credit_usage_ratio']

    # merge on SK_ID_PREV
    engineered_cc = pd.merge(engineered_cc, avg_usage_ratio_per_loan, on="SK_ID_PREV", how="outer")
    return engineered_cc


  ### ENGINEERING AVG_CC_CNT_ATM_DRAWINGS ###
  def engineer_avg_cc_cnt_atm_drawings(df_credit_card_balance, engineered_cc):
    # use groupby to get on average how many ATM drawings there are per month per loan
    avg_ATM_drawings_per_loan = pd.DataFrame(df_credit_card_balance.groupby('SK_ID_PREV')['CNT_DRAWINGS_ATM_CURRENT'].mean()).reset_index()
    avg_ATM_drawings_per_loan.columns = ['SK_ID_PREV', 'avg_cc_cnt_ATM_drawings']
    avg_ATM_drawings_per_loan = avg_ATM_drawings_per_loan.fillna(0)
    avg_ATM_drawings_per_loan

    # merge into the engineered features dataframe
    engineered_cc = pd.merge(engineered_cc, avg_ATM_drawings_per_loan, on='SK_ID_PREV', how='outer')
    return engineered_cc


  ### GROUPBY ENGINEERED_CC DF AND THEN MERGE INTO APPLICATION
  def groupby_engineered_cc(df_application, engineered_cc):
    # groupby SK_ID_CURR
    engineered_cc = engineered_cc.groupby('SK_ID_CURR').mean()

    # reset index
    engineered_cc = engineered_cc.reset_index()
    engineered_cc.drop(columns='SK_ID_PREV', inplace=True)

    # merge to df_application
    df_application = pd.merge(df_application, engineered_cc, on="SK_ID_CURR", how='left')
    return df_application


### ENGINEERING PREV_AVG_RATIO_CREDIT_APPROVED ###
  def engineer_prev_avg_ratio_credit_approved(df_application, df_previous_application):
    # calculate the ratio of AMT_CREDIT to AMT_APPLICATION
    df_previous_application['prev_ratio_credit_approved'] = df_previous_application['AMT_CREDIT'] / df_previous_application['AMT_APPLICATION']

    # replace inf with 1
    df_previous_application['prev_ratio_credit_approved'] = df_previous_application['prev_ratio_credit_approved'].replace({np.inf: 1})

    # groupby SK_ID_CURR and average the values

    ratio_credit_approved = pd.DataFrame(df_previous_application.groupby('SK_ID_CURR')['prev_ratio_credit_approved'].mean()).reset_index()
    ratio_credit_approved.columns = ['SK_ID_CURR','prev_avg_ratio_credit_approved']

    # merge to df_application
    df_application = pd.merge(df_application, ratio_credit_approved, on="SK_ID_CURR", how='left')
    return df_application

### ENGINEERING NAME_CONTRACT_STATUS FEATURES ###
  def engineer_name_contract_status_features(df_application, df_previous_application):
    # separate columns
    contract_status_df = df_previous_application[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']].copy()

    # create binary columns for all the value types
    contract_status_df['prev_status_approved'] = (df_previous_application['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    contract_status_df['prev_status_refused'] = (df_previous_application['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)

    # drop  unnecessary columns
    contract_status_df.drop(columns=['NAME_CONTRACT_STATUS','SK_ID_PREV'], inplace=True)

    # groupby SK_ID_CURR, sum the values
    binary_contract_status = contract_status_df.groupby('SK_ID_CURR').sum()
    binary_contract_status = binary_contract_status.reset_index()

    # merge into the application
    df_application = pd.merge(df_application, binary_contract_status, on='SK_ID_CURR', how='outer')
    return df_application


### ENGINEERING PREV_YIELD_GROUP ###
  def engineer_prev_yield_group_features(df_application, df_previous_application):
    # separate the columns from the df
    name_yield_group = df_previous_application[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_YIELD_GROUP']].copy()

    # ohe the NAME_YIELD_GROUP column using a column transformer
    yield_group_preprocess = ColumnTransformer([
      ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['NAME_YIELD_GROUP'])
    ], remainder='passthrough', verbose_feature_names_out=False)
    ohe_yield_group = yield_group_preprocess.fit_transform(name_yield_group)

    # rename columns to be consistent with other engineered features
    ohe_yield_group = ohe_yield_group.rename(columns={
      'NAME_YIELD_GROUP_XNA':'prev_yield_XNA',
      'NAME_YIELD_GROUP_high': 'prev_yield_high',
      'NAME_YIELD_GROUP_low_action': 'prev_yield_low_action',
      'NAME_YIELD_GROUP_low_normal': 'prev_yield_low_normal',
      'NAME_YIELD_GROUP_middle': 'prev_yield_middle'
    })

    # groupby SK_ID_CURR and sum the values
    ohe_yield_group = ohe_yield_group.groupby('SK_ID_CURR').sum()

    # drop SK_ID_PREV
    ohe_yield_group.drop(columns=['SK_ID_PREV','prev_yield_XNA','prev_yield_low_normal','prev_yield_middle'], inplace=True)

    # reset index
    ohe_yield_group.reset_index(inplace=True)

    # merge into application
    df_application = pd.merge(df_application, ohe_yield_group, on='SK_ID_CURR', how='outer')
    return df_application


  ### ENGINEERING APPLICATIONS OHE FEATURES ###
  def filtering_ohe_features(df_application):
    # define the ohe
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # define a column transformer
    encoder = ColumnTransformer([
      ('ohe', ohe, ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS'])
    ], remainder='passthrough', verbose_feature_names_out=False)

    # fit and transform
    df_application = encoder.fit_transform(df_application)

    return df_application


  df_application = clean_days_employed(df_application)
  df_application = engineer_num_closed_bureau_credits(df_application, df_bureau)
  df_application = engineer_avg_ratio_bureau_cr_debt(df_application, df_bureau)
  df_application = engineer_ttl_bureau_cc_limit(df_application, df_bureau)
  engineered_cc = engineer_cc_avg_credit_usage_ratio(df_credit_card_balance)
  engineered_cc = engineer_avg_cc_cnt_atm_drawings(df_credit_card_balance, engineered_cc)
  df_application = groupby_engineered_cc(df_application, engineered_cc)
  df_application = engineer_prev_avg_ratio_credit_approved(df_application, df_previous_application)
  df_application = engineer_name_contract_status_features(df_application, df_previous_application)
  df_application = engineer_prev_yield_group_features(df_application, df_previous_application)
  df_application = filtering_ohe_features(df_application)

  return df_application
