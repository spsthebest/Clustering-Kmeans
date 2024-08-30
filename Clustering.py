import numpy as np  # Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Load the crime analysis data from a CSV file
data_in = pd.read_csv('C:\\Crime_Analysis.csv')  

# Extract the latitude and longitude columns
x1 = data_in['Y']  
y1 = data_in['X']  

# Convert the data into a numpy array
data = np.array(list(zip(x1, y1))).reshape(len(x1), 2)

# Initialize an empty list to store the distortions
distortions = []

# Define the range of clusters to test
clusters = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Iterate over the range of clusters
for k in clusters:
    # Create a KMeans model with k clusters
    kmean = KMeans(n_clusters=k)
    # Fit the model to the data
    kmean.fit(data)
    # Calculate the distortion for the current k
    distortions.append(sum(np.min(cdist(data, kmean.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow graph
plt.plot(clusters, distortions, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()