from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    objects_types = working_train_df.select_dtypes(include='object')
    # Ordinal Encoder
    two_col_names = objects_types.loc[:, objects_types.nunique() == 2].columns
    # One Hot Encoder
    four_col_names = objects_types.loc[:, objects_types.nunique() > 2].columns

    ordinal_encoder = OrdinalEncoder()
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    # Lets fit them...
    ordinal_encoder.fit(working_train_df[two_col_names])
    one_hot_encoder.fit(working_train_df[four_col_names])
    # Lets train them...
    #Ordinal encoders
    working_train_df[two_col_names] = ordinal_encoder.transform(working_train_df[two_col_names])
    working_val_df[two_col_names] = ordinal_encoder.transform(working_val_df[two_col_names])
    working_test_df[two_col_names] = ordinal_encoder.transform(working_test_df[two_col_names])
    # One hot encoders
    ohe_wtdf_transformed = pd.DataFrame(one_hot_encoder.transform(working_train_df[four_col_names]), columns=one_hot_encoder.get_feature_names_out(four_col_names), index=working_train_df.index)
    ohe_wvdf_transformed = pd.DataFrame(one_hot_encoder.transform(working_val_df[four_col_names]), columns=one_hot_encoder.get_feature_names_out(four_col_names), index=working_val_df.index)
    ohe_wtsdf_transformed = pd.DataFrame(one_hot_encoder.transform(working_test_df[four_col_names]), columns=one_hot_encoder.get_feature_names_out(four_col_names), index=working_test_df.index)

    working_train_df = pd.concat([working_train_df, ohe_wtdf_transformed], axis=1)
    working_train_df.drop(columns=four_col_names, inplace=True, axis=1)
    working_val_df = pd.concat([working_val_df, ohe_wvdf_transformed], axis=1)
    working_val_df.drop(columns=four_col_names, inplace=True, axis=1)
    working_test_df = pd.concat([working_test_df, ohe_wtsdf_transformed], axis=1)
    working_test_df.drop(columns=four_col_names, inplace=True, axis=1)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    simple_imputer = SimpleImputer(strategy='median')
    simple_imputer.fit(working_train_df)

    working_train_df = simple_imputer.transform(working_train_df)
    working_val_df = simple_imputer.transform(working_val_df)
    working_test_df = simple_imputer.transform(working_test_df)


    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(working_train_df)

    working_train_df = min_max_scaler.transform(working_train_df)
    working_val_df = min_max_scaler.transform(working_val_df)
    working_test_df = min_max_scaler.transform(working_test_df)


    return working_train_df, working_val_df, working_test_df
