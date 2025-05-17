import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
	
data = {
    'vehicle_serial_no': [5, 3, 8, 2, 4, 7, 6, 10, 1, 9],
    'mileage': [150000, 120000, 250000, 80000, 100000, 220000, 180000, 300000, 75000, 280000],
    'fuel_efficiency': [15, 18, 10, 22, 20, 12, 16, 8, 24, 9],
    'maintenance_cost': [5000, 4000, 7000, 2000, 3000, 6500, 5500, 8000, 1500, 7500],
    'vehicle_type': ['SUV', 'Sedan', 'Truck', 'Hatchback', 'Sedan', 'Truck', 'SUV', 'Truck', 'Hatchback', 'SUV']
}

df = pd.DataFrame(data)

df_fixed = df.copy()

le = LabelEncoder()

df_fixed['vehicle_type'] = le.fit_transform(df['vehicle_type'])

x =df_fixed.iloc[:, :].values
wcss_list = []

for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

mtp.plot(range(1,11), wcss_list)
mtp.title('Elbow Method graph')
mtp.xlabel('Number of clusters k')
mtp.ylabel('wcss_list')
mtp.show()

#without scaling 
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(x)

mtp.scatter(x[y_predict == 0, 0], x[y_predict==0, 1], s=100, c="purple", label= 'Cluster 1')
mtp.scatter(x[y_predict == 1, 0], x[y_predict==1, 1], s=100, c="blue", label= 'Cluster 2')
mtp.scatter(x[y_predict == 2, 0], x[y_predict==2, 1], s=100, c="green", label= 'Cluster 3')
mtp.scatter(x[y_predict == 3, 0], x[y_predict==3, 1], s=100, c="yellow", label= 'Cluster 4')
mtp.scatter(x[y_predict == 4, 0], x[y_predict==4, 1], s=100, c="orange", label= 'Cluster 5')
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label = 'Centroid')

mtp.title('Clusters of vehicals')
mtp.xlabel('Serial number ')
mtp.ylabel('mileage')
mtp.legend()
mtp.show()

features_to_scale = df[['vehicle_serial_no', 'mileage', 'fuel_efficiency', 'maintenance_cost']]


scaled_part = StandardScaler().fit_transform(features_to_scale)
custom_x = scaled_part
kmeans= KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(custom_x)

mtp.scatter(custom_x[y_predict==0, 0], custom_x[y_predict==0, 1], s=100, c='purple', label='Cluster 1')
mtp.scatter(custom_x[y_predict==1, 0], custom_x[y_predict==1, 1], s=100, c='blue', label='Cluster 2')
mtp.scatter(custom_x[y_predict==2, 0], custom_x[y_predict==2, 1], s=100, c='green', label='Cluster 3')
mtp.scatter(custom_x[y_predict==3, 0], custom_x[y_predict==3, 1], s=100, c='yellow', label='Cluster 4')
mtp.scatter(custom_x[y_predict==4, 0], custom_x[y_predict==4, 1], s=100, c='orange', label='Cluster 5')

mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label = 'Centroid')

mtp.title('Vehicals Cluster')
mtp.xlabel('Serial number ')
mtp.ylabel('mileage')
mtp.legend()
mtp.show()