import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# Step 1: Load the dataset
data = pd.read_csv('DatasetProject.csv')

# Check the columns in the dataset
print("Columns in the dataset:", data.columns)

# Standardize column names
data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace

# Step 2: Select the desired columns (by index)
selected_columns = data.iloc[:, [2, 3, 4, 5]]
print("Selected columns:\n", selected_columns.head())

# Step 3: Standardize numerical features
scaler = StandardScaler()
selected_columns = selected_columns.astype(float)  # Ensure all features are float
scaled_features = scaler.fit_transform(selected_columns)

# Step 4: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method result
plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply KMeans Clustering with the chosen number of clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Step 6: Map clusters to Grades A, B, C
cluster_to_grade = {0: 'A', 1: 'B', 2: 'C'}
data['Grade'] = data['cluster'].map(cluster_to_grade)

# Step 7: Calculate evaluation metrics
silhouette_avg = silhouette_score(scaled_features, data['cluster'])
print("Silhouette Score:", silhouette_avg)

davies_bouldin = davies_bouldin_score(scaled_features, data['cluster'])
print("Davies-Bouldin Score:", davies_bouldin)

calinski_harabasz = calinski_harabasz_score(scaled_features, data['cluster'])
print("Calinski-Harabasz Score:", calinski_harabasz)

# Step 8: Visualize the clusters (using PCA if high-dimensional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)
data['pca1'] = X_pca[:, 0]
data['pca2'] = X_pca[:, 1]

# Scatter plot for clusters based on PCA components
plt.figure(figsize=(10, 7))
sns.scatterplot(x='pca1', y='pca2', hue='Grade', data=data, palette='Set2')
plt.title('Clusters Visualized with PCA')
plt.show()

# Step 9: Prepare for classification (optional)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['Grade'], test_size=0.3, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Step 10: Predict on the test set
y_pred = classifier.predict(X_test)

# Step 11: Classify new unseen data
# Define new unseen data using the exact column names as in the original dataset
new_data = pd.DataFrame({
    'previous grades (cgpa)': [8.5, 6.5, 4.5],
    'Overall attendance rate': [85, 65, 45],
    'study hours per week': [18, 12, 7],
    'number of projects completed': [7, 3, 1]
})

# Standardize the new data
new_data_scaled = scaler.transform(new_data)  # This uses the original scaler fitted on the training data

# Predict grade for the new data
new_grade_predictions = classifier.predict(new_data_scaled)

print("Predicted grades for new unseen data:", new_grade_predictions)

# Step 12: Evaluate the classifier (optional)
print("Classification Report:")
print(classification_report(y_test, y_pred))
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# Step 1: Load the dataset
data = pd.read_csv('last.csv')

# Check the columns in the dataset
print("Columns in the dataset:", data.columns)

# Standardize column names
data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace

# Step 2: Select the desired columns (by index)
selected_columns = data.iloc[:, [2, 3, 4, 5]]
print("Selected columns:\n", selected_columns.head())

# Step 3: Standardize numerical features
scaler = StandardScaler()
selected_columns = selected_columns.astype(float)  # Ensure all features are float
scaled_features = scaler.fit_transform(selected_columns)

# Step 4: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method result
plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply KMeans Clustering with the chosen number of clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Step 6: Map clusters to Grades A, B, C
cluster_to_grade = {0: 'A', 1: 'B', 2: 'C'}
data['Grade'] = data['cluster'].map(cluster_to_grade)

# Step 7: Calculate evaluation metrics
silhouette_avg = silhouette_score(scaled_features, data['cluster'])
print("Silhouette Score:", silhouette_avg)

davies_bouldin = davies_bouldin_score(scaled_features, data['cluster'])
print("Davies-Bouldin Score:", davies_bouldin)

calinski_harabasz = calinski_harabasz_score(scaled_features, data['cluster'])
print("Calinski-Harabasz Score:", calinski_harabasz)

# Step 8: Visualize the clusters (using PCA if high-dimensional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)
data['pca1'] = X_pca[:, 0]
data['pca2'] = X_pca[:, 1]

# Scatter plot for clusters based on PCA components
plt.figure(figsize=(10, 7))
sns.scatterplot(x='pca1', y='pca2', hue='Grade', data=data, palette='Set2')
plt.title('Clusters Visualized with PCA')
plt.show()





# Step 9: Prepare for classification (optional)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['Grade'], test_size=0.3, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Step 10: Predict on the test set
y_pred = classifier.predict(X_test)

# Step 11: Classify new unseen data
# Define new unseen data using the exact column names as in the original dataset
new_data = pd.DataFrame({
    'previous grades (cgpa)': [8.5, 6.5, 4.5],
    'Overall attendance rate': [85, 65, 45],
    'study hours per week': [18, 12, 7],
    'number of projects completed': [7, 3, 1]
})

# Standardize the new data
new_data_scaled = scaler.transform(new_data)  # This uses the original scaler fitted on the training data

# Predict grade for the new data
new_grade_predictions = classifier.predict(new_data_scaled)

print("Predicted grades for new unseen data:", new_grade_predictions)

# Step 12: Evaluate the classifier (optional)
print("Classification Report:")
print(classification_report(y_test, y_pred))
