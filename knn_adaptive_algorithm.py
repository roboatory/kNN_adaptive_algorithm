from data_cleaning import create_training_samples
import math
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = create_training_samples()

## STANDARD KNN 

def unweighted_euclidean_distance(point1, point2):
	vector_one = np.array(list(point1))
	vector_two = np.array(list(point2))
	return np.linalg.norm(vector_two - vector_one)

def get_k_neighbors(query_point, k):
	query_point_relations = []

	for row in range(X_train.shape[0]):
		point_features = list(X_train.iloc[row])
		distance = unweighted_euclidean_distance(query_point, point_features)
		classification = y_train.iloc[row]
		query_point_relations.append((distance, classification, point_features))

	query_point_relations.sort(key = lambda x : x[0])
	return query_point_relations[:k]

def classify_classical_unweighted(neighbors):
	count_malignant = 0
	count_benign = 0

	for neighbor in neighbors:
		if (neighbor[1] == 1):
			count_malignant += 1
		else:
			count_benign += 1

	return 1 if count_malignant > count_benign else 0

def classify_classical_weighted(neighbors):
	malignant_weight = 0
	benign_weight = 0

	for neighbor in neighbors:
		weight = 1 / (neighbor[0] + 0.001)
		if (neighbor[1] == 1):
			malignant_weight += weight
		else:
			benign_weight += weight

	return 1 if malignant_weight > benign_weight else 0

def evaluate_accuracy(y_true, y_pred):
	test_length = len(y_true)
	count_matches = 0

	for classification in range(test_length):
		if y_true[classification] == y_pred[classification]:
			count_matches += 1

	return (count_matches / test_length)

## GAUSSIAN CCW (METHOD ADOPTED + MODIFIED FROM LIU + CHAWLA'S PAPER: CLASS 
## CONFIDENCE WEIGHTED KNN ALGORITHMS FOR IMBALANCED DATA SETS)

def feature_stats():
	data = X_train.merge(y_train, left_index = True, right_index = True)

	malignant_samples = data.loc[data["target"] == 1]
	benign_samples = data.loc[data["target"] == 0]

	std_mfeatures = malignant_samples.iloc[:, :-1].std(axis = 0)
	std_bfeatures = benign_samples.iloc[:, :-1].std(axis = 0)
	mean_mfeatures = malignant_samples.iloc[:, :-1].mean(axis = 0)
	mean_bfeatures = benign_samples.iloc[:, :-1].mean(axis = 0)

	return std_mfeatures, std_bfeatures, mean_mfeatures, mean_bfeatures

def apply_gaussian_ccw(point, mean_class, std_class):
	total_weight = 1

	for feature in range(len(point)):
		distance = unweighted_euclidean_distance([point[feature]], [mean_class[feature]])

		std_feature = std_class[feature]
		gaussian_weight = (np.exp(-(distance ** 2) / (2 * (std_feature ** 2))) * 
			1 / (math.sqrt(2 * math.pi) * std_feature))

		total_weight *= gaussian_weight
	
	return total_weight

def determine_class_gaussian(neighbors):
	std_mfeatures, std_bfeatures, mean_mfeatures, mean_bfeatures = feature_stats()
	gaussian_mal = []
	gaussian_ben = []

	for neighbor in neighbors: 
		if neighbor[1] == 0: 
			total_weight = apply_gaussian_ccw(neighbor[2], 
				list(mean_bfeatures), list(std_bfeatures))
			gaussian_ben.append(total_weight)
		else:
			total_weight = apply_gaussian_ccw(neighbor[2], 
				list(mean_mfeatures), list(std_mfeatures))
			gaussian_mal.append(total_weight)

	if len(gaussian_mal) == 0:
		return 0
	else: 
		ccw_ratio = sum(gaussian_ben) / sum(gaussian_mal)
		classification = 0 if ccw_ratio > 1 else 1
		return classification

## FUZZY KNN (METHOD ADOPTED FROM SARKER'S AND LEONG'S PAPER: 
## APPLICATION OF K-NEAREST NEIGHBORS ALGORITHM ON BREAST CANCER DIAGNOSIS PROBLEM)

def fuzzy_knn(neighbors):
	mal_confidence = 0
	ben_confidence = 0
	denom = sum([1 / (neighbor[0] ** 2) for neighbor in neighbors])

	for neighbor in neighbors:
		numerator = 1 / (neighbor[0] ** 2) 
		probability = numerator / denom

		if neighbor[1] == 1:
			mal_confidence += probability
		else:
			ben_confidence += probability

	classification = (1 if mal_confidence > ben_confidence else 0)
	return classification

## POINT RADII (METHOD ADOPTED FROM WANG'S, NESKOVIC'S, AND COOPER'S PAPER:
## IMPROVING NEAREST NEIGHBOR RULE WITH A SIMPLE ADAPTIVE DISTANCE MEASURE)

def calc_point_radii():
	point_radii = []
	for point_one in range(len(X_train)):
		distances = []
		for point_two in range(len(X_train)):
			distance = unweighted_euclidean_distance(X_train.iloc[point_one], 
				X_train.iloc[point_two])
			
			if (y_train.iloc[point_one] != y_train.iloc[point_two]):
				distances.append(distance)
		point_radii.append(min(distances))

	return point_radii

def get_radii_neighbors(query_point, k, point_radii):
	query_point_relations = []

	for row in range(X_train.shape[0]):
		point_features = list(X_train.iloc[row])
		distance = unweighted_euclidean_distance(query_point, point_features) / point_radii[row]
		classification = y_train.iloc[row]
		query_point_relations.append((distance, classification, point_features))

	query_point_relations.sort(key = lambda x : x[0])
	return query_point_relations[:k]

## COEFFICIENT MEAN DISTANCE FUNCTION: CALCULATES THE MEAN DISTANCE 
## BETWEEN THE NEIGHBORS AND THE QUERY POINT (µref) AS WELL AS THE 
## NEIGHBORS AND THEIR RESPECTIVE CLASS MEANS (µclass)

def coefficient_mean_distance(neighbors):
	std_mfeatures, std_bfeatures, mean_mfeatures, mean_bfeatures = \
		feature_stats()
	
	malignant_neighbors = []
	benign_neighbors = []
	mcluster_distance = []
	bcluster_distance = []

	for neighbor in neighbors:
		if neighbor[1] == 1:
			malignant_neighbors.append(neighbor[0])
			mean_distance = unweighted_euclidean_distance(neighbor[2], mean_mfeatures)
			mcluster_distance.append(mean_distance)
		else:
			benign_neighbors.append(neighbor[0])
			mean_distance = unweighted_euclidean_distance(neighbor[2], mean_bfeatures)
			bcluster_distance.append(mean_distance)

	if len(malignant_neighbors) == 0:
		return 0
	if len(benign_neighbors) == 0:
		return 1

	avg_m_distance = np.mean(malignant_neighbors)
	avg_b_distance = np.mean(benign_neighbors)

	avg_mcluster_distance = np.mean(mcluster_distance)
	avg_bcluster_distance = np.mean(bcluster_distance)

	m_alpha = len(malignant_neighbors) / len(neighbors)
	b_alpha = len(benign_neighbors) / len(neighbors)

	mval = m_alpha * (avg_m_distance / avg_mcluster_distance)
	bval = b_alpha * (avg_b_distance / avg_bcluster_distance)

	classification =  (1 if mval > bval else 0)
	return classification

## LOCAL GAUSSIAN FUNCTION: TRIES TO FIND THE CONFIDENCE OF THE QUERY POINT IN RELATION 
## TO BOTH MALIGNANT AND BENIGN LOCAL PROBABILITY DISTRIBUTIONS (USING MEAN AND STANDARD DEVIATION)

def local_gaussian(neighbors, query_point):
	malignant = []
	benign = []
	malignant_count = 0
	benign_count = 0

	for neighbor in neighbors:
		if neighbor[1] == 1:
			malignant.append(neighbor[2])
			malignant_count += 1
		else:
			benign.append(neighbor[2])
			benign_count += 1
	
	if malignant_count > 1 and benign_count > 1:
		mean_malignant_neighbors = np.mean(malignant, axis = 0)
		std_malignant_neighbors = np.std(malignant, axis = 0)
		mean_benign_neighbors = np.mean(benign, axis = 0)
		std_benign_neighbors = np.std(benign, axis = 0)

		mal_gaussian = apply_gaussian_ccw(query_point, mean_malignant_neighbors, 
			std_malignant_neighbors)
		ben_gaussian = apply_gaussian_ccw(query_point, mean_benign_neighbors, 
			std_benign_neighbors)

		classification = (1 if mal_gaussian > ben_gaussian else 0)
		return classification
	else:
		classification = classify_classical_unweighted(neighbors)
		return classification

def main():
	start_val = None # start value for k (inclusive)
	end_val = None # end value for k (inclusive)

	for k in range(start_val, end_val + 2, 2):
		y_pred = []

		for row in range(len(X_test)):
			query_point = X_test.iloc[row]
			neighbors = None # choose get neighbors method
			prediction = None # choose prediction method
			y_pred.append(prediction)
	 
		accuracy = evaluate_accuracy(list(y_test), y_pred)
		print("Accuracy at k = " + str(k) + ": " + str(accuracy))

main()
