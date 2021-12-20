''' Data preprocessing '''
import pandas as pd
import numpy as np


def create_time_series_data(baseline_df, vit_df, labs_df):
    """
    Create time-series format containing both longitudinal and baseline features.

    Args:
        baseline_df: Baseline dataframe, including demographics, background diseases etc.
        vit_df: Vital signs dataframe.
        labs_df: Lab test results dataframe.

    Returns: dataframe in time-series format [columns: features, rows: patients' observations].
    """
    # Merge longitudinal and baseline DFs
    longitudinal_df = pd.concat([labs_df, vit_df])
    long_data = pd.merge(longitudinal_df, baseline_df, on='patient_id', how='left')

    # Pivot table
    baseline_features = list(baseline_df.columns)  # these features are set as indices in the pivoted DF
    pivot_df = pivot_data_frame(long_data, baseline_features)
    pivot_df = pivot_df.sort_values(by=['admission_datetime', 'patient_id', 'datetime'],
                                    ascending=True).reset_index(drop=True)
    return pivot_df


def pivot_data_frame(df, baseline_features):
    """
    Pivot the dataframe (T). Baseline features remains constant (indices of pivoted table)
    """
    # Nan values are deleted in the pivot operation.
    # Temporary fill Nan
    df[baseline_features] = df[baseline_features].fillna('dummy')

    # Remove NULL values
    df = df[~df.Value.isna()]

    # Define indices (constant columns that are not pivoted)
    indices = baseline_features + ['datetime']

    # Pivot table
    pivoted_df = df.pivot_table(index=indices,
                                columns='Feature',
                                values='Value',
                                aggfunc='first').reset_index().replace('dummy', np.nan)
    return pivoted_df


def create_time_grid(df, time_freq='60T', agg_method='mean'):
    """
    Creates discrete time grid to time-series data, according to fixed frequency.
    Multiple numerical values (within the same time-window) are aggregated by mean. The remaining grouped by first.
    For grouping the target outcome, one should consider agg_method='max'.
    Args:
        df: A dataframe in time-series format.
        time_freq: The grid resolution (default=hourly grid)
        agg_method: The aggregation method for numerical columns

    Returns: discretized dataframe.
    """

    original_cols = df.columns
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    non_numeric_cols = list(df.select_dtypes(exclude=np.number).columns)
    indices = ['patient_id', 'datetime']

    df['datetime'] = pd.DatetimeIndex(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.sort_values(by=indices)

    # Perform discretization: numerical cols by agg_method, and non-numerical cols by "first"
    grouper = [df['patient_id'], pd.Grouper(level='datetime', freq=time_freq)]
    agg_method_df = pd.DataFrame(df.groupby(grouper).agg(agg_method).to_records())[indices + numeric_cols]
    first_df = pd.DataFrame(df.groupby(grouper).agg('first').to_records())[non_numeric_cols]
    out_df = pd.merge(first_df, agg_method_df, how='left', on=indices)

    assert (sorted(out_df.columns) == sorted(original_cols)), "Error! At least one column was lost"

    return out_df


def drop_indifferent_features(df):
    """ Drops features with SD==0 """
    features = df.columns
    drop_list = []
    for feature in features:
        unique_values = df[~df[feature].isna()][feature].unique()
        if len(unique_values) == 1:
            print("%s has std=0" % feature)
            drop_list.append(feature)
    df = df.drop(columns=drop_list)

    return df
