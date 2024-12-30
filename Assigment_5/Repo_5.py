#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# In[2]:


# Load dataset
wine = load_wine()
X = wine.data


# In[3]:


# Determine optimal number of clusters
scores = []
clusters = range(2, 10)
for k in clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    scores.append(silhouette_score(X, labels))


# In[4]:


# Plot silhouette scores
plt.plot(clusters, scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Cluster Number')
plt.show()


# In[5]:


# Apply K-Means with optimal clusters
optimal_k = clusters[scores.index(max(scores))]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)


# In[6]:


# Visualize clusters
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering")
plt.show()


# In[ ]:




