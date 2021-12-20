## *A Machine Learning Model for Predicting Deterioration of COVID-19 Inpatients*

Public repository containing research code for the COVID-19 prediction model, described in the manuscript 
*"A Machine Learning Model for Predicting Deterioration of COVID-19 Inpatients"*  

#### Authors:
Omer Noy*, Dan Coster*, Maya Metzger, Itai Attar, Shani Shenhar-Tsarfaty, Shlomo Berliner, Galia Rahav, Ori Rogowski, Ron Shamir

## Code:
#### Install requirements
```
python -m pip install -r requirements.txt
```
This code was tested with python 3.7.

#### Modules
The repository is organized as follows:
* **`anomaly_scores`** Anomaly detection approaches, used as anomaly features
* **`data_preprocessing`** Code for preprocessing the parsed data, including time-series formatting, imputation, etc. 
* **`feature_generation`** Code for features engineering, including historical summary statistics and trend features.
* **`feature_selection`** Code for feature selection strategies.
* **`ml_models`** ML model classes, including pre-training processing, fit and evaluation methods.
* **`outlier_removal`** Values Removal according to predefined clinical ranges.

## Data:
The data used in our study cannot be shared. This section describes the data format used for the code. 

Our data-specific parser generates 3 main pandas dataframes:
* **`Baseline_df`** Contains demographics and background disease (static features).
* **`vital_df`** Vital signs (longitudinal features).
* **`labs_df`** Lab tests (longitudinal features).

#### Baseline dataframe format (Baseline_df):
|   Columns           | Data type      | 
|---------------------|----------------|
| Patient ID          | object         |
| Admission Date      | datetime64[ns] |
| Gender              | bool           |
| Age                 | float64        |
| ...                 | String         |
| Background diseases | bool           | 

#### Longitudinal dataframes format (vital_df, labs_df):
|   Columns    |   Data type    | 
|--------------|----------------|
| Patient ID   | object         |
| Date Time    | datetime64[ns] |
| Feature Name | String         |
| Value        | float64        | 

Using `data_preprocessing/create_time_series_data`, the dataframes can be merged and pivoted into a time-series format, 
with columns representing features and rows representing the longitudinal patients' observations.
