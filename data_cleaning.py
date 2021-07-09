import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

# load breast cancer dataset
def load_cancer_dataset():
	bc_dataset = load_breast_cancer(as_frame = True)["frame"]
	bc_dataset.drop(bc_dataset.columns[10:30], axis = 1, inplace = True)
	bc_dataset["target"] = bc_dataset["target"].apply(lambda x : 1.0 if x == 0 else 0.0)
	return bc_dataset

# apply scaling to every feature
def normalize_data():
	df = load_cancer_dataset()
	scaler = MinMaxScaler()

	scaled_df = pd.DataFrame(scaler.fit_transform(df))
	scaled_df.columns = df.columns

	return scaled_df

# create training & test samples
def create_training_samples():
	norm_df = normalize_data()

	X = norm_df[norm_df.columns[:-1]]
	y = norm_df["target"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
	return (X_train, X_test, y_train, y_test)
	