#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# Welcome to "***Clustering (Customer Segmentation) Project***". This is the last medium project of ***Machine Learning*** course.
# 
# At the end of this project, you will have performed ***Cluster Analysis*** with an ***Unsupervised Learning*** method.
# 
# ---
# 
# In this project, customers are required to be segmented according to the purchasing history obtained from the membership cards of a big mall.
# 
# This project is less challenging than other projects. After getting to know the data set quickly, you are expected to perform ***Exploratory Data Analysis***. You should observe the distribution of customers according to different variables, also discover relationships and correlations between variables. Then you will spesify the different variables to use for cluster analysis.
# 
# The last step in customer segmentation is to group the customers into distinct clusters based on their characteristics and behaviors. One of the most common methods for clustering is ***K-Means Clustering***, which partitions the data into k clusters based on the distance to the cluster centroids. Other clustering methods include ***hierarchical clustering***, density-based clustering, and spectral clustering. Each cluster can be assigned a label that describes its main features and preferences.
# 
# - ***NOTE:*** *This project assumes that you already know the basics of coding in Python. You should also be familiar with the theory behind Cluster Analysis and scikit-learn module as well as Machine Learning before you begin.*

# ---
# ---

# # #Tasks

# Mentoring Prep. and self study####
# 
# #### 1. Import Libraries, Load Dataset, Exploring Data
# - Import Libraries
# - Load Dataset
# - Explore Data
# 
# #### 2. Exploratory Data Analysis (EDA)
# 
# 
# #### 3. Cluster Analysis
# 
# - Clustering based on Age and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Clustering based on Annual Income and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Hierarchical Clustering
# 
#     *i. Determine optimal number of clusters using Dendogram*
# 
#     *ii. Apply Agglomerative Clustering*
# 
#     *iii. Visualizing and Labeling All the Clusters*
# 
# - Conclusion

# ---
# ---

# ## 1. Import Libraries, Load Dataset, Exploring Data
# 
# There is a big mall in a specific city that keeps information of its customers who subscribe to a membership card. In the membetrship card they provide following information : gender, age and annula income. The customers use this membership card to make all the purchases in the mall, so tha mall has the purchase history of all subscribed members and according to that they compute the spending score of all customers. You have to segment these customers based on the details given.

# #### Import Libraries
# 

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Load Dataset

# In[22]:


df = pd.read_csv("C:/Users/HTC/Downloads/Mall_Customers.csv")
df


# #### Explore Data
# 
# You can rename columns to more usable, if you need.

# In[23]:


df.info()


# In[ ]:





# In[ ]:





# ---
# ---

# ## 2. Exploratory Data Analysis (EDA)
# 
# After performing Cluster Analysis, you need to know the data well in order to label the observations correctly. Analyze frequency distributions of features, relationships and correlations between the independent variables and the dependent variable. It is recommended to apply data visualization techniques. Observing breakpoints helps you to internalize the data.
# 
# 
# 
# 

# In[24]:


sns.pairplot(df)
plt.show()


# In[25]:


df.describe()


# In[26]:


df.nunique()


# In[27]:


df.isnull().sum()


# In[28]:


# check for each numerical feature boxplot and skew also we have now two categorical variables
cat_cols=df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)


for col in num_cols:
    print(col)
    print('Skew :', round(df[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    df[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.show()


# In[29]:


df.shape


# In[30]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[31]:


plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['Spending Score (1-100)'], bins=20)
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.show()



plt.hist(df['Annual Income (k$)'], bins=20)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show()


# In[32]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Spending Score (1-100)'], df['Annual Income (k$)'], df['Age'], cmap='rainbow')
ax.set_xlabel('Spending Score (1-100)')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Age')
plt.show()


# ---
# ---

# ## 3. Cluster Analysis

# The purpose of the project is to perform cluster analysis using [K-Means](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) and [Hierarchical Clustering](https://medium.com/analytics-vidhya/hierarchical-clustering-d2d92835280c) algorithms.
# Using a maximum of two variables for each analysis can help to identify cluster labels more clearly.
# The K-Means algorithm requires determining the number of clusters using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering), while Hierarchical Clustering builds a dendrogram without defining the number of clusters beforehand. Different labeling should be done based on the information obtained from each analysis.
# Labeling example:
# 
# - **Normal Customers**  -- An Average consumer in terms of spending and Annual Income
# - **Spender Customers** --  Annual Income is less but spending high, so can also be treated as potential target customer.

# ### Clustering based on Age and Spending Score

# #### *i. Create a new dataset with two variables of your choice*

# In[33]:


selected_columns = ['Age', 'Spending Score (1-100)']
new_df = df[selected_columns]


# #### *ii. Determine optimal number of clusters*

# In[34]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ssd = []

K = range(2,10)

for k in K:
    model = KMeans(n_clusters =k,
                   random_state=42)
    model.fit(new_df)
    ssd.append(model.inertia_) # Inertia calculates WCSS

# Plotting the Elbow graph

plt.plot(K, ssd, "bo--")
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('inertia-error')  # Within-cluster sum of squares
plt.show()
# Determine optimal number of clusters = 4


# In[ ]:





# In[ ]:





# ### Why silhouette_score is negative?

# ![image.png](data:image/png;base64,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1ooVKwj/5IN01J7F/V++QxvavXvL5sGdO/vOO/+j/+WLLRde1Oy+gBWLxa6urtTDEwfy0ksvjaJo586d5513HuH/D+FUjuMMDw8fPXpUkqSurq6lS5ee+HK/oCilVasjw/yH8YlbS61IIyd6enry+fzevXvL5XKKl7ISP2VVFGXTpk133XXXyMhI0rEwe7Ztv/DCC/fee+/dd9/9rW996/7779+6davnec3uC2aQy+X27t179913b9q0KcXwBIEMw7BSqTz33HPf+c53jh49muJgMHtvvPHGvffe297eft99933jG99wXfe+++7bt29fs/uChqIo0jRtx44d995778aNG9MVSfCU9dChQ4888simTZts2ya8tgZO5Xnes88+29nZ+e1vfzu+DHBgYOD222//zW9+MzAwQHgZDRAyTXPjxo2PPfbY6OjofHyk+eDg4DPPPPOZz3zm/vvvT3cwmKVarfbqq6+uXbt2+qLNvr6+q6666rXXXjNNs7m9QSOHDx9+9NFHly5d+sMf/vCKK65IVyTBI2RfX9/DDz98zTXXbNu2Ld3BYJampqaOHDnS398//RYEWZb7+/u3bNlSr9dbWlqa2x6cVqFQ+MIXvnDTTTcFc9i6IUEgr7766tSHgUSCIPA876TX6HK5XLVa1Y5fNAe8WbJkyec//3nG2MTEROoi+GuER0EQhGFI+DZRmDdR2je+xU7/CDk8PLx///7pm2vWrJm+GGCOx4PZUBRFUZTptzjGgiCIv96srmA2zkogH3vssXvuuWf65pYtW6666qq5HAYSUVW1paXlwIED018Jw3BiYqKnp4fqjW/Ap9MHsr+///rrr5++iWvl5llHR8fAwMDQ0JBt2/E7oTRN27lz5wUXXFAoFGYcDgvX6QN522233XbbbfPcCkzL5/Pr1q372c9+tnnz5tWrV4dh+Pvf/35oaOib3/ymqqrN7g7OIrxvg0eCIKxfv/7FF1/8/ve/v3btWs/zXnrppWuvvXbdunXNbg3OrjSBLJfLV155Ja4sP6uWLFny3e9+91e/+tXmzZvz+fxnP/vZW265pa2trdl9wcwURbnssst6enpSjE0cSMdxVq5c+dOf/jTFwSCRnp6eO++8884772x2I5CA53ktLS2pr2ZL/H/IvXv32rad7mAA73tHjx6tVCqphycO5NatW48dO5b6eADvb6Oj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)
silhouette_score = (b-a)/max(a,b)

b : the mean nearest-cluster distance
a : the mean intra-cluster distance

for red point,

b = 1
a = ((1+1)**0.5 + (1+1)**0.5)/2  ==> 1.41

silhouette_score = (1-1.41)/1.41 ==> -0.29

############################################----ANSWER-----###############################

#Why silhouette_score is negative?
# Answer : this is the worst case is silhouette score is negative this mean the distance within cluster is close to infinity and the distance to nearest cluster is close to 0

################
# #### *iii. Apply K Means*

# In[35]:


from sklearn.cluster import KMeans

K_means_model = KMeans(n_clusters=4,
                       random_state=42)


# In[36]:


K_means_model.fit(new_df)


# In[37]:


K_means_model.predict(new_df)


# In[38]:


K_means_model.fit_predict(new_df)


# In[ ]:





# #### *iv. Visualizing and Labeling All the Clusters*

# In[39]:


K_means_model.labels_


# In[ ]:





# In[ ]:





# In[43]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
def hopkins(X, ratio=0.05):

    if not isinstance(X, np.ndarray):
      X=X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0] * ratio) #0.05 (5%) based on paper by Lawson and Jures

    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))

    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]

    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)

    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour

    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H


# In[44]:


new_df.shape


# In[45]:


hopkins(new_df, 0.05)


# In[46]:


hopkins(new_df, 1)


# In[48]:


from sklearn.metrics import silhouette_score
silhouette_score(new_df, K_means_model.labels_)


# In[49]:


range_n_clusters = range(2,9)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters,
                    random_state=42)
    kmeans.fit(new_df)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(new_df, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")


# In[50]:


pip install yellowbrick


# In[102]:


from sklearn.cluster import KMeans

from yellowbrick.cluster import SilhouetteVisualizer

model3 = KMeans(n_clusters=4,
                random_state=42)          # we decided n_clusters=3!
visualizer = SilhouetteVisualizer(model3)

visualizer.fit(new_df)    # Fit the data to the visualizer
visualizer.poof();


# In[103]:


model3.labels_


# In[104]:


visualizer.silhouette_samples_


# In[105]:


labels_1 = (model3.labels_ == 1)


# In[106]:


visualizer.silhouette_samples_[labels_1]


# In[107]:



mean_silhoutte_score_1 = visualizer.silhouette_samples_[labels_1].mean()
mean_silhoutte_score_1


# In[108]:


for i in range(4):
    label = (model3.labels_== i)
    print(f"mean silhouette score for label {i:<4} : {visualizer.silhouette_samples_[label].mean()}")
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# In[110]:


clusters1 = model3.labels_
new_df["predicted_clusters"] = clusters1
new_df


# ### Clustering based on Annual Income and Spending Score

# In[ ]:





# #### *i. Create a new dataset with two variables of your choice*

# In[58]:


selected_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
new_df1 = df[selected_columns]


# In[ ]:





# #### *ii. Determine optimal number of clusters*

# In[59]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ssd = []

K = range(2,10)

for k in K:
    model = KMeans(n_clusters =k,
                   random_state=42)
    model.fit(new_df1)
    ssd.append(model.inertia_) # Inertia calculates WCSS

# Plotting the Elbow graph

plt.plot(K, ssd, "bo--")
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('inertia-error')  # Within-cluster sum of squares
plt.show()
# Determine optimal number of clusters = 5


# #### *iii. Apply K Means*

# In[60]:


from sklearn.cluster import KMeans

K_means_model1 = KMeans(n_clusters=5,
                       random_state=42)


# In[61]:


K_means_model1.fit(new_df1)


# In[62]:


K_means_model1.predict(new_df1)


# In[63]:


K_means_model1.fit_predict(new_df1)


# #### *iv. Visualizing and Labeling All the Clusters*

# In[64]:


K_means_model1.labels_


# In[66]:


# Calculate cluster centers
cluster_centers = kmeans.cluster_centers_

# Print cluster centers (for 2D data)
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i} Center: Annual Income={center[0]}, Spending Score={center[1]}")


# In[67]:


hopkins(new_df1, 0.05)


# In[68]:


hopkins(new_df1, 1)


# In[69]:


silhouette_score(new_df1, K_means_model1.labels_)


# In[70]:


range_n_clusters = range(2,9)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters,
                    random_state=42)
    kmeans.fit(new_df1)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(new_df1, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")


# In[111]:


from sklearn.cluster import KMeans

from yellowbrick.cluster import SilhouetteVisualizer

model3 = KMeans(n_clusters=5,
                random_state=42)          # we decided n_clusters=3!
visualizer = SilhouetteVisualizer(model3)

visualizer.fit(new_df1)    # Fit the data to the visualizer
visualizer.poof();


# In[112]:


model3.labels_


# In[113]:


visualizer.silhouette_samples_


# In[114]:


labels_1 = (model3.labels_ == 4)
visualizer.silhouette_samples_[labels_1]
mean_silhoutte_score_1 = visualizer.silhouette_samples_[labels_1].mean()
mean_silhoutte_score_1


# In[115]:


for i in range(5):
    label = (model3.labels_== i)
    print(f"mean silhouette score for label {i:<6} : {visualizer.silhouette_samples_[label].mean()}")
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# In[ ]:


clusters2 = model3.labels_
new_df["predicted_clusters"] = clusters2
new_df


# ### Hierarchical Clustering

# ### *i. Determine optimal number of clusters using Dendogram*

# ### Clustering based on Age and Spending Score- x1

# In[76]:


selected_columns_H = ['Age', 'Spending Score (1-100)']
df1_hier = df[selected_columns_H]


# In[77]:


from scipy.cluster.hierarchy import dendrogram, linkage

import scipy.cluster.hierarchy as sch


# In[78]:



# Create a dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(df1_hier, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# ## Clustering based on Annual Income and Spending Score- x2

# In[79]:


selected_columns_H1 = ['Annual Income (k$)', 'Spending Score (1-100)']
df11_hier = df[selected_columns_H1]


# In[80]:


# Create a dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(df11_hier, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# ### ii. *Apply Agglomerative Clustering*

# #### Age and Spending Score- x1

# In[81]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# In[82]:


K = range(2,10)

for k in K:
    model = AgglomerativeClustering(n_clusters=k)
    model.fit_predict(df1_hier)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(df1_hier, model.labels_)}')


# In[94]:


# 2 OR 4 K  but i chose based on dendogram 
model = AgglomerativeClustering(n_clusters=2,
                                 affinity='euclidean',
                                linkage="ward")
cluster_labels = model.fit_predict(df1_hier)


# In[ ]:





# #### Annual Income and Spending Score- x2

# In[84]:


# 5 
K = range(2,10)

for k in K:
    model = AgglomerativeClustering(n_clusters=k)
    model.fit_predict(df11_hier)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(df11_hier, model.labels_)}')


# In[85]:


model = AgglomerativeClustering(n_clusters=5,
                                 affinity='euclidean',
                                linkage="ward")
cluster_labels1 = model.fit_predict(df11_hier)


# ### iii. *Visualizing and Labeling All the Clusters*

# #### Age and Spending Score- x1

# In[95]:


df1_hier['Cluster'] = cluster_labels
df1_hier


# In[87]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', data=df1_hier, palette='Set1', s=100)
plt.title('Clusters of Customers based on Age and Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[90]:


# try with 3 cluster 
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', data=df1_hier, palette='Set1', s=100)
plt.title('Clusters of Customers based on Age and Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[93]:


# try with 4 cluster 
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', data=df1_hier, palette='Set1', s=100)
plt.title('Clusters of Customers based on Age and Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[96]:



plt.figure(figsize = (10,10))
sns.pairplot(data=df1_hier, hue="Cluster", palette="bright");
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Annual Income and Spending Score- x2

# In[97]:


df11_hier['Cluster1'] = cluster_labels1
df11_hier


# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Spending Score (1-100)', y='Annual Income (k$)', hue='Cluster1', data=df11_hier, palette='Set1', s=100)
plt.title('Clusters of Customers based on Spending Score and Annual Income ')
plt.xlabel('Spending Score')
plt.ylabel('Annual Income (k$)')
plt.show()


# In[99]:


plt.figure(figsize = (10,10))
sns.pairplot(data=df11_hier, hue="Cluster1", palette="bright");
plt.show()


# #### Interpretation based on Age and Spending Score- x1

# In[ ]:





# ### Conclusion

# **cluster 0** : The average age is around 55, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct to more spending.
# 
# **cluster 1**: The average age is around 45, the annula_income is high but the spending_scores are very low.
# This group is our target audience and specific strategies should be developed to drive this group to spend.
# 
# **cluster 2** :The average age is around 30. The annula_income is high and spending_scores are very high.
# This group consists of our loyal customers. Our company derives the main profit from this group. Very
# special promotions can be made in order not to miss it.    
#     
# **cluster 3**: The average age is around 25.both annula_income and spending_scores are on average.
# It should be researched what can be done to direct to more spending.

# #### Interpretation based on Annual Income and Spending Score- x2

# In[ ]:





# ## Conclusion

# ### Female
# 
# **cluster 0** : The average age is around 40, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 45, the annula_income is very high but the spending_scores is low.
# This group is our target audience and special strategies need to be developed for this group.    
# 
# **cluster 2** :The average age is around 45. Both annula_income and spending_scores are low. It can be
# directed to shopping with gift certificates.
# 
# **cluster 3**: The average age is around 25. Low annual_incomes but very high spending scores. This
# group does a lot of shopping, but they do not bring much profit.
# 
# **cluster 4**: The average age is around 30, the annual income and the spending_score
# is very high. This group consists of our loyal customers. Our company derives the main profit from this group.
# Very special promotions can be made in order not to miss it.

# ### Male
# 
# **cluster 0** : The average age is around 45, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 40, the annula_income is very high but the spending_scores is very low.
# This group is our target audience and special strategies need to be developed for this group.    
# 
# **cluster 2** :The average age is around 50. Both annula_income and spending_scores are low. It can be
# directed to shopping with gift certificates.
# 
# **cluster 3**: The average age is around 25. Low annual_incomes but very high spending scores. This
# group does a lot of shopping, but they do not bring much profit.
# 
# **cluster 4**: The average age is around 30, the annual income and the spending_score
# is very high. This group consists of our loyal customers. Our company derives the main profit from this group.
# Very special promotions can be made in order not to miss it.

# In[ ]:





# **cluster 0** : The average age is around 40, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 30, both annula_income and spending_scores are very high.
# This group consists of our loyal customers. Our company derives the main profit from this group. Very
# special promotions can be made in order not to miss it.
# 
# **cluster 2** :The average age is around 45. Both annula_income and spending_scores are low. It can be
# directed to shopping with gift certificates.
# 
# **cluster 3**: The average age is around 25. Low annual_incomes but very high spending scores. This
# group does a lot of shopping, but they do not bring much profit.
# 
# **cluster 4**: The average age is around 40, their annual income is very high but their spending_score
# is very low. This group is our target audience and special strategies need to be developed for this
# group.

# 
# 

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
