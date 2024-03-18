from scripts.parquetreader import read_parquet_file
from scripts.model_transformer import prepare_data_and_create_transformer_model
from scripts.dataprep import prepare_and_split_data_grouped
import os,sys
import pandas as pd

current_directory = os.getcwd()
parquet_file_path = os.path.join(current_directory, 'input', 'sample_trainingdata.parquet')

dfmodelstep1=read_parquet_file(parquet_file_path)
print(dfmodelstep1.shape[0])

# Assuming dfmodelstep1 is your DataFrame, 'target' is your target column
savcols=['customer_ID','S_2']
Target='target'
group='customer_ID'

X_train, X_test, y_train, y_test, identifiers_train, identifiers_test = prepare_and_split_data_grouped(dfmodelstep1, Target,savcols, group)

model,early_stopping_callback = create_model_with_transformer(input_shape=(X_train.shape[1], X_train.shape[2]))

print('Model summary to verify architecture')
print(model.summary())