# -------------------------------------------------------------------------
# AUTHOR: Aslak Djuve
# FILENAME: pca.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 4 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv("heart_disease_dataset.csv")
print(df.head())
#Create a training matrix without the target variable (Heart Diseas)
#--> add your Python code here
df_features = df

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here
num_features = len(df_features.columns)
print(num_features)
max_explain = 0
# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = df.drop(df.columns[i], axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_variance = pca.explained_variance_ratio_[0]
    feature_removed = df_features.columns[i]
    print(f"PC1 variance: {pc1_variance} when removing {feature_removed}")
    if pc1_variance > max_explain:
        max_explain = pc1_variance
        holdout_var = feature_removed

# Find the maximum PC1 variance
# --> add your Python code here
#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {max_explain} when removing {holdout_var}")







