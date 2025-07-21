# Student Grade Prediction System 🎓

## Overview
This project presents a student performance analysis system designed to automatically classify and interpret various student performance metrics based on engagement and academic attributes. The primary goal is to evaluate students’ academic standing, attendance, participation in projects, and co-curricular activities, providing insights into areas that contribute to or hinder student success. We utilize various machine learning algorithms, including **KMeans Clustering**, **PCA**, **Support Vector Machine (SVM)**, **Logistic Regression**, **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, and **XGBoost**, to classify students into different performance levels.

---

## Motivation
The primary purpose of this project is to create a tool that can classify students based on various academic and engagement metrics, assisting educational institutions in identifying students' strengths and areas requiring support or improvement. This classification can guide targeted support strategies, helping institutions proactively enhance student success and make data-driven decisions to improve academic interventions.

---

## Dataset
* **Source:** The project utilizes an uploaded dataset named `DatasetProject.csv`.
* **Data Fields:** Includes information on student academic performance and engagement metrics, such as GPA, attendance rate, study hours per week, project and internship completion, participation in co-curricular activities, and internet access.
* **Preprocessing:**
    * Initial preprocessing involves handling missing values, removing extraneous whitespace, normalizing numeric values, and encoding categorical responses. This step ensures a clean and consistent dataset for training and evaluation.
    * Each student’s performance is labeled into categories (e.g., high, medium, low) based on academic and engagement metrics. This labeling enables a supervised learning approach to classify students' performance levels, allowing models to learn patterns associated with different achievement levels.

---

## Model Architectures
The project explores a variety of machine learning models for student performance classification:

* **KMeans Clustering & PCA:** Used for grouping students and visualizing the clusters based on performance metrics.
* **Support Vector Machine (SVM):** Effective with high-dimensional data, making it suitable for classifying students based on multiple engagement and academic features.
* **Logistic Regression:** A probabilistic model that works well for binary or multiclass classification, providing interpretable probabilities for student performance categories.
* **Decision Tree & Random Forest:** Suitable for capturing complex, non-linear relationships in student data, these models can identify the most influential factors contributing to academic success.
* **K-Nearest Neighbors (KNN):** Classifies based on the similarity between student profiles, using proximity to other data points in the feature space.
* **XGBoost:** An optimized, gradient-boosting approach offering robust performance, especially for structured educational data, by learning from patterns within multiple features.

---

## Implementation Details
* **Platform:** Compatible with Windows, macOS, and Linux. Typically developed in Jupyter Notebook or Spyder IDE.
* **Language:** Python.
* **Libraries:** Pandas, scikit-learn, Matplotlib, Seaborn, PCA (Principal Component Analysis).
* **Data Storage:** The data is stored in an Excel or CSV format.
* **Hardware Requirements (Recommended):**
    * **Processor:** A Quad-core processor or higher is recommended for efficient computation, enabling smooth data processing and model training.
    * **RAM:** A minimum of 8GB of RAM is required to handle data processing tasks and facilitate efficient training of machine learning models.
    * **Hard Disk:** A minimum of 10GB of free storage space is necessary to store the dataset.

---

## Results and Evaluation
Each model is evaluated based on **accuracy, classification reports, and other metrics such as precision, recall, and F1 scores**. Visualizations like confusion matrices and plots aided in assessing model robustness and interpretability.

### Key Visualizations & Metrics (Sample Output):
* **Elbow Method for Optimal Clusters:** Demonstrates the selection of optimal clusters for KMeans.
    ![Elbow Method Plot](path/to/your/elbow_method_plot.png)
* **Clusters Visualized with PCA:** Shows the distribution of student grades (A, B, C) in a reduced 2D PCA space.
    ![PCA Clusters Plot](path/to/your/pca_clusters_plot.png)
* **Clustering Metrics:**
    * Silhouette Score: 0.671407544309226
    * Davies-Bouldin Score: 0.46990158197266707
    * Calinski-Harabasz Score: 1470.1356516437836
    ![Clustering Scores Output](path/to/your/clustering_scores_output.png)
* **Clusters of Students (Previous Grade vs. Attendance):** Visual representation of student grouping.
    ![Student Clusters Plot](path/to/your/student_clusters_plot.png)
* **True vs Predicted Clusters (Test Set):** Compares actual and predicted clusters.
    ![True vs Predicted Clusters Plot](path/to/your/true_vs_predicted_clusters_plot.png)
* **Classification Report (Sample):**
    | Class | Precision | Recall | F1-Score | Support |
    | :---- | :-------- | :----- | :------- | :------ |
    | A     | 1.00      | 1.00   | 1.00     | 30      |
    | B     | 1.00      | 1.00   | 1.00     | 31      |
    | C     | 1.00      | 1.00   | 1.00     | 29      |
    | **accuracy** |           |        | **1.00** | 90      |
    | macro avg | 1.00      | 1.00   | 1.00     | 90      |
    | weighted avg | 1.00      | 1.00   | 1.00     | 90      |
    ![Classification Report Output](path/to/your/classification_report_output.png)
* **Confusion Matrix (Sample):**
    ```
    [[30 0 0]
     [ 0 31 0]
     [ 0 0 29]]
    ```
    * **Accuracy Score:** 1.0
    ![Confusion Matrix Output](path/to/your/confusion_matrix_output.png)

*The models achieved perfect classification on the provided test set, as indicated by the reported metrics.*

---

## Conclusion
The project successfully developed a student performance analysis system capable of classifying students based on various academic and engagement metrics. By applying machine learning algorithms like KMeans Clustering and PCA for grouping, and various classifiers (SVM, Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost) for prediction, the system provides valuable insights into student success factors. This system is intended to provide valuable insights for student support strategies and academic improvements.

---

## Future Enhancements
1.  **Advanced Text Representation:** Experiment with embeddings like BERT for richer text representations if textual data is integrated.
2.  **Multi-Platform Data Integration:** Extend the tool to gather data from additional educational platforms for broader analysis.
3.  **Model Optimization:** Implement automated hyperparameter tuning and ensemble learning for better accuracy.

---

## Real-Time Applications
* **Student Insights:** Educational institutions can use this tool to gauge student standing, identify strengths, and address areas needing improvement, thereby enhancing academic support and curriculum development strategies.
* **Intervention Optimization:** By analyzing performance trends, educators can identify at-risk students and optimize support resources proactively.
* **Personalized Learning:** This tool enables personalized feedback and learning paths based on individual student profiles.
