from sklearn.model_selection import train_test_split
from src.ML_pipeline.Utils import read_dataset
from src.ML_pipeline.Utils import merge_dataframes
from src.ML_pipeline.Utils import remove_outliers
from src.ML_pipeline.Utils import year_from_date
from src.ML_pipeline.Impute import impute
from src.ML_pipeline.Cat_to_num import cat_to_num
from src.ML_pipeline.Utils import month_from_date
from src.ML_pipeline.Train_model import train_model
from src.ML_pipeline.Evaluate_results import evaluate_results
from src.ML_pipeline.Feature_importance import feature_importance
import pandas as pd
import numpy as np
import joblib

def inference_method(df, store_df):

    # Combining 2 dataframes
    # combined_data = merge_dataframes(store_df, df, 'Store')

    # Imputation
    store_details = impute(store_df, 'Promo2SinceWeek', method='value')
    store_details = impute(store_df, 'Promo2SinceYear', method='value')
    store_details = impute(store_df, 'PromoInterval', method='value')
    store_details = impute(store_df, 'CompetitionDistance', method='mean')
    store_details = impute(store_df, 'CompetitionOpenSinceMonth', method='mode')
    store_details = impute(store_df, 'CompetitionOpenSinceYear', method='mode')
    combined_data = merge_dataframes(df, store_details, 'Store')

    # Removing Exceptions
    combined_data.drop(combined_data.loc[(combined_data['Open'] == 1) &
                                        (combined_data['StateHoliday'] == 0) &
                                        (combined_data['SchoolHoliday'] == 0)].index, inplace=True)

    # Extract year from date
    combined_data = year_from_date(combined_data, 'Date', 'Year')

    # Extract month from date
    combined_data = month_from_date(combined_data, 'Date', 'Month')

    # Catagorical to numerical
    combined_data = cat_to_num(combined_data, 'Assortment', 'default')
    combined_data = cat_to_num(combined_data, 'StoreType', 'default')
    impute_dict = {
        "Jan,Apr,Jul,Oct": 1,
        "Feb,May,Aug,Nov": 2,
        "Mar,Jun,Sept,Dec": 3
    }
    combined_data = cat_to_num(
        combined_data, 'PromoInterval', 'custom', values=impute_dict)
    impute_dict_2 = {
        'a': 1,
        'b': 2,
        'c': 3
    }
    combined_data = cat_to_num(
        combined_data, 'StateHoliday', 'custom', values=impute_dict_2)

    # Convert to numeric
    combined_data['StateHoliday'] = pd.to_numeric(combined_data['StateHoliday'])
    combined_data['PromoInterval'] = pd.to_numeric(combined_data['PromoInterval'])

    # Train test split
    combined_data_subset = combined_data[combined_data['Open'] == 1]
    model = joblib.load('./output/dt.pkl')

    predictions = model.predict(combined_data_subset.drop([ 'Date', 'Open', 'Promo'], axis=1)).tolist()
    print("here")
    return predictions


