''' feature generation module '''
import numpy as np
import pandas as pd
from sklearn import linear_model
from tqdm.notebook import tqdm


def features_ratio(df, features):
    """ Adds new features that are the ratio between numerical features """
    new_cols = []
    for numerator in range(len(features)):
        for denominator in range(numerator + 1, len(features)):
            # Generate ratios
            col_name = features[numerator] + "_on_" + features[denominator]
            df[col_name] = df[features[numerator]] / df[features[denominator]]
            new_cols.append(col_name)

    # replace invalid results to NaN
    df[new_cols] = df[new_cols].replace([np.inf, -np.inf, float(0)], np.nan)

    return df


def summary_statistics_features(df, features, ignore_last=False, full_history=False, horizons=[24, 72]):
    """
    Generates historical summary statistics features.

    Args:
        df: Dataframe containing "Patient ID", "DateTime" and numerical features.
        features: Numerical features on which to calculate the statistics.
        ignore_last: ignore the current result when calculating the mean
        full_history: whether to calculate summary statistics based on the entire hospitalization period so far
        horizons: Time windows during which to calculate the statistics.

    Returns: df with the new features
    """

    df.index = df.DateTime
    stat_list = ['mean', 'min', 'max', 'std']  # desired statistics

    if full_history:
        horizon = df['patient_id'].value_counts().max() + 1
        horizons = [horizon]

    for horizon in horizons:
        rolling_horizon = horizon if full_history else str(horizon) + 'h'
        len_lamda, stat_lambdas = get_stat_lamdas(rolling_horizon)

        for feat_name in tqdm(features):
            for feat_index, feat_stat in enumerate(stat_list):
                suffix = '' if full_history else ("_" + rolling_horizon)
                col_name = feat_name + "_" + feat_stat + suffix
                df[col_name] = df.groupby('patient_id')[feat_name].transform(stat_lambdas[feat_index])

                if feat_stat == 'mean':
                    if ignore_last:
                        df[col_name] = df[col_name].mul(df.groupby('patient_id')[feat_name].transform(len_lamda)).sub(
                            df[feat_name])
                        df[col_name] = df[col_name].div(df.groupby('patient_id')[feat_name].transform(len_lamda).sub(1))

                    df[feat_name + "_delta_mean" + suffix] = df[feat_name] - df[col_name]

                # Remove statistics if feature value is null
                df.loc[df[feat_name].isnull(), (col_name)] = np.NaN

    df = df.reset_index(drop=True)

    return df


def get_stat_lamdas(rolling_horizon):
    """
        Defines statistics calculations over historical longitudinal data.
        An helper function of summary_statistics_features()
    """
    len_lamda = lambda x: x.rolling(rolling_horizon, min_periods=2).count()
    stat_lambdas = [lambda x: x.rolling(rolling_horizon, min_periods=1).mean(),  # mean
                    lambda x: x.rolling(rolling_horizon, min_periods=1).min(),  # min
                    lambda x: x.rolling(rolling_horizon, min_periods=1).max(),  # max
                    lambda x: x.rolling(rolling_horizon, min_periods=1).std()]  # std
    return len_lamda, stat_lambdas


def add_lr_slope(df, feat_list):
    """ Adds the linear regression coef for each of the listed features, for each patient. """
    regressor = linear_model.LinearRegression()
    for feat_name in feat_list:
        slopes_per_feature = []

        for pid in df['patient_id'].unique():
            # Create a df per patient
            patient_df = df[df['patient_id'] == pid]
            patient_df = patient_df[['time_since_admission', feat_name, 'patient_id']]
            n_rows = len(patient_df)
            if patient_df.size:
                for row_index in range(n_rows):
                    # If the value exists, calculate slope until this value.
                    if not pd.isnull(patient_df.iloc[row_index, 1]):
                        slope = get_slope(patient_df, row_index, regressor)
                        slopes_per_feature.extend([slope])
                    else:
                        slopes_per_feature.extend([np.nan])
            else:  # Otherwise, fill with nan.
                slopes_per_feature.extend([np.nan] * n_rows)
        df[feat_name + "_lr_slope"] = slopes_per_feature

    return df


def get_slope(patient_df, row_index, regressor):
    """ helper function of add_lr_slope() """
    slope_df = patient_df.iloc[:row_index + 1, :].dropna()
    date_times = slope_df.iloc[:, :1].values
    feat_values = slope_df.iloc[:, 1].values
    regressor.fit(date_times, feat_values)
    slope = regressor.coef_[0]
    return slope
