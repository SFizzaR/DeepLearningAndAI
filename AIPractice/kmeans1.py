import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder #convert string to int 

df = pd.read_csv("Mall_Customers.csv")


#First 5 rows of df 
df.head()

df_fixed = df.copy()

le = LabelEncoder()

df_fixed['Gender'] = le.fit_transform(df_fixed['Gender'])

#extact variables [r, c]
x = df_fixed.iloc[:, 1:5].values 

wcss_list = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

mtp.plot(range(1, 11), wcss_list) #(x,y)
mtp.title('The Elbow Method Graph')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('wcss_list')
mtp.show()


#training the clusters without scalar
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(x)

#visualizing the clusters 
mtp.scatter(x[y_predict==0, 0], x[y_predict==0, 1], s=100, c='red', label= 'Cluster 1') #first cluster
mtp.scatter(x[y_predict==1, 0], x[y_predict==1, 1], s=100, c='orange', label= 'Cluster 2')
mtp.scatter(x[y_predict==2, 0], x[y_predict==2, 1], s=100, c='yellow', label= 'Cluster 3')
mtp.scatter(x[y_predict==3, 0], x[y_predict==3, 1], s=100, c='green', label= 'Cluster 4')
mtp.scatter(x[y_predict==4, 0], x[y_predict==4, 1], s=100, c='blue', label= 'Cluster 5')

mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroid')
mtp.title('Clusters of Customers')
mtp.xlabel('Feature 1')
mtp.ylabel('Feature 2')
mtp.legend()
mtp.show()

#with scaling 
age = df_fixed.iloc[:, 2].values.reshape(-1, 1)

features_to_scale = df_fixed.iloc[:, [1, 3, 4]].values
scaled_part = StandardScaler().fit_transform(features_to_scale)

x_custom = np.concatenate((age, scaled_part), axis= 1)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict=kmeans.fit_predict(x_custom)

#visualizing the clusters 
mtp.scatter(x_custom[y_predict==0, 0], x_custom[y_predict==0, 1], s=100, c='red', label= 'Cluster 1') #first cluster
mtp.scatter(x_custom[y_predict==1, 0], x_custom[y_predict==1, 1], s=100, c='orange', label= 'Cluster 2')
mtp.scatter(x_custom[y_predict==2, 0], x_custom[y_predict==2, 1], s=100, c='yellow', label= 'Cluster 3')
mtp.scatter(x_custom[y_predict==3, 0], x_custom[y_predict==3, 1], s=100, c='green', label= 'Cluster 4')
mtp.scatter(x_custom[y_predict==4, 0], x_custom[y_predict==4, 1], s=100, c='blue', label= 'Cluster 5')

mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroid')
mtp.title('Clusters of Customers')
mtp.xlabel('Feature 1')
mtp.ylabel('Feature 2')
mtp.legend()
mtp.show()