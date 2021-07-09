from data_cleaning import normalize_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

norm_bc_df = normalize_data()
x_cols = norm_bc_df.iloc[:, :-1].columns

def dimensionality_reduction():
	X = norm_bc_df.iloc[:, :-1]
	y = norm_bc_df.iloc[:, -1]
	X_embedded = TSNE(n_components = 2, random_state = 0).fit_transform(np.array(X))

	targets = np.array(y)
	malignant = X_embedded[targets == 1]
	benign = X_embedded[targets == 0]

	plt.scatter(malignant[:, 0], malignant[:, 1], label = "M", color = "Red")
	plt.scatter(benign[:, 0], benign[:, 1], label = "B", color = "Blue")
	plt.legend()
	plt.show()

def plot_error_rates():
	df = pd.read_excel("tables/knn_adaptive_results.xlsx").set_index("K")

	error_df = 100 - df
	error_df.columns = error_df.columns.str.replace("\(%\)", "")
	error_df["AVERAGE"] = error_df.mean(axis = 1)

	ax = error_df.iloc[:, :-1].plot(linestyle = "dashed", 
		title = "Error Rates for kNN Modifications", figsize = (12, 7), 
		style = "o-")
	ax.set_ylabel("Error Rates (%)")
	error_df.iloc[:, -1].plot(linestyle = "solid", legend = True, 
		color = "black")

	plt.show()

def plot_normalized_error_rates():
	df = pd.read_excel("tables/knn_adaptive_results.xlsx").set_index("K")
	df.columns = df.columns.str.replace("\(%\)", "")
	df.columns = df.columns.str.strip()

	error_df = df.subtract(df["STANDARD KNN"], axis = 0)

	ax = error_df.iloc[:, 1:].plot(linestyle = "dashed", 
		title = "Error Rates for kNN Modifications", figsize = (12, 7), 
		style = "o-")
	ax.set_ylabel("Error Rates (%)")
	error_df.iloc[:, 0].plot(linestyle = "solid", legend = True, 
		color = "black")

	plt.show()
	