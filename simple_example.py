#!/usr/bin/env python

# import dependencies
import pandas as pd
import numerapi as napi
import sklearn.linear_model

# Get your API keys from https://numer.ai/tutorial
public_id = "REPLACEME"
secret_key = "REPLACEME"

# download the latest training dataset
print('loading training data')
training_data = pd.read_csv(
    "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz"
)

# download the latest tournament dataset
print('loading tournament data')
tournament_data = pd.read_csv(
    "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz"
)

# find only the feature columns
feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]

# select those columns out of the training dataset
training_features = training_data[feature_cols]

# create a model and fit the training data
print('training model')
model = sklearn.linear_model.LinearRegression()
model.fit(training_features, training_data.target_kazutsugi)

# select the feature columns from the tournament data
live_features = tournament_data[feature_cols]

# predict the target on the live features
print('running predictions')
predictions = model.predict(live_features)

# predictions must have an `id` column and a `prediction_kazutsugi` column
predictions_df = tournament_data["id"].to_frame()
predictions_df["prediction_kazutsugi"] = predictions
predictions_df.head()

# Upload your predictions
print('uploading predictions')
napi = napi.NumerAPI(public_id=public_id, secret_key=secret_key)
predictions_df.to_csv("predictions.csv", index=False)
submission_id = napi.upload_predictions("predictions.csv")
