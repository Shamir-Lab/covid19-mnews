import json


def remove_out_of_range_values(df, ranges_path):
    """
    Remove values that exceed the pre-defined clinical range of possible values.

    Args:
        df: Dataframe in time-series format (feature are represented in columns)
        ranges_path: path to json file, defining the possible ranges.

    Returns: The dataframe with the masked values.

    """
    ranges = load_ranges_json(ranges_path)
    null_before = df.isnull().sum().sum()  # num of values before masking

    features, missing_features = [], []
    for x in df.columns:
        features.append(x) if x in ranges.keys() else missing_features.append(x)

    df[features] = df[features].apply(lambda c: c.mask(~c.between(ranges[c.name]['min'], ranges[c.name]['max'])))

    if missing_features:
        print("The following features doesn't have ranges in the ranges path specified, and therefore ignored:\n"
              , missing_features)
    null_after = df.isnull().sum().sum()  # num of values after masking
    print(f"In total, {str(null_after - null_before)} invalid values were removed.")

    return df


def load_ranges_json(path_for_ranges):
    """
    Loads the json containing all the features' possible ranges

    Args: path_for_ranges: path to ranges json file

    Returns: ranges dict
    """
    with open(path_for_ranges) as json_ranges:
        json_ranges = json.load(json_ranges)

    ranges = {}
    for feature_ranges in json_ranges["human ranges"]:
        ranges[feature_ranges["feature"]] = {"min": feature_ranges["min"], "max": feature_ranges["max"]}

    return ranges
