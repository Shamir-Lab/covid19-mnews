''' Data standardization (mean=0, SD=1) '''

def standardize_data(df, features):
    """
        Standardize values of seen_data (TRAINING set).
    """
    df_out = df.copy()
    features_params = {}
    for feature in features:
        std = df_out[feature].std()
        mean = df_out[feature].mean()
        if std == 0:  # if the std is 0, change nothing
            features_params[feature] = {'mean': 0,
                                        'std': 1}
        else:
            df_out[feature] = (df_out[feature] - mean) / std
            features_params[feature] = {'mean': mean,
                                        'std': std}
    return df_out, features_params


def standardize_unseen_data(df, features, train_params):
    """
        Standardize values of unseen_data (TEST set).
    """
    df_out = df.copy()
    for feature in features:
        std = train_params[feature]['std']
        mean = train_params[feature]['mean']
        df_out[feature] = (df_out[feature] - mean) / std
    return df_out


def is_data_standardized(df, features):
    """  Check if the data is already standardized """
    epsilon = 1e-10
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        # Check if mean == 0 and std == 1, up to epsilon
        if not ((0 - epsilon <= mean) and (mean <= 0 + epsilon) and
                (1 - epsilon <= std) and (std <= 1 + epsilon)):
            print("The values of %s are not standardized: mean=%.3f, std=%.3f, epsilon=%f" % (
                feature, mean, std, epsilon))
            return False
    return True