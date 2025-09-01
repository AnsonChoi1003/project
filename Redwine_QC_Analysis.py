# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""Step 1: Loading the Dataset"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Explanation:** The dataset is loaded into a Pandas DataFrame. This is the first step in any data analysis process, as it allows me to work with the data in a structured format.

**Justification:** Using Pandas is a standard practice in data analysis due to its powerful data manipulation capabilities.

Step 2: Exploring the Data
"""

print(data.head())
print("Columns in DataFrame:", data.columns)

"""**Explanation:** The first few rows of the dataset are displayed along with the column names.

**Justification:** This initial exploration helps us understand the structure of the dataset, including the types of data present and the names of the columns.

Step 3: Checking Data Types
"""

print(data.dtypes)

"""**Explanation:** The data types of each column are printed.

 **Justification:** Understanding the data types is crucial for data analysis, as it informs us which columns can be treated as numeric and which are categorical.

Step 4: Handling Categorical Data
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})

"""**Explanation:** If the dataset contains a 'type' column, it is converted to binary values (0 for white wine and 1 for red wine).

**Justification:** Converting categorical data to numeric values allows for easier analysis and correlation with other numeric features.

Step 5: Converting Data to Numeric
"""

data = data.apply(pd.to_numeric, errors='coerce')

"""**Explanation:** All columns are attempted to be converted to numeric types, coercing any non-convertible values to NaN.

**Justification:** This step ensures that all data is in a format suitable for numerical analysis, which is essential for statistical computations.

Step 6: Dropping Missing Values
"""

data = data.dropna()

"""**Explanation:** Rows with NaN values are removed from the dataset.

**Justification:** Missing values can lead to inaccurate analysis and predictions. Dropping them helps maintain the integrity of the analysis.

Step 7: Descriptive Statistics
"""

print(data.describe())

"""**Explanation:** Basic statistics (mean, median, min, max, etc.) for each numeric column are displayed.

**Justification:** Descriptive statistics provide a summary of the data distribution and help identify any potential outliers or anomalies.

Step 8: **Correlation Matrix**
"""

correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""**Explanation:** A correlation matrix is computed and visualized using a heatmap.

**Justification:** The correlation matrix helps identify relationships between variables, guiding feature selection for predictive modeling. Strong correlations can indicate potential predictors for quality.

Step 9: Adding features values
"""

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulfates', 'alcohol', 'quality']

"""Step 10: **Scatter Plots**"""

for feature in features:
    if feature in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=feature, y='quality')
        plt.title(f'Relationship between {feature} and Quality')
        plt.xlabel(feature)
        plt.ylabel('Quality')
        plt.show()

"""**Explanation:** Scatter plots are created for each feature against the quality.

**Justification:** Visualizing the relationships between individual features and the target variable (quality) allows for a better understanding of how each feature influences quality.

Step 11: **Box Plot**
"""

if 'alcohol' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quality', y='alcohol', data=data)
    plt.title('Alcohol Content by Wine Quality')
    plt.xlabel('Quality')
    plt.ylabel('Alcohol Content')
    plt.show()

"""**Explanation:** A box plot is created to show the distribution of alcohol content across different quality ratings.

**Justification:** Box plots are useful for visualizing the central tendency and variability of numerical data across categories, helping to identify trends in alcohol content relative to quality.

**Discussion of Results**

-After analyzing the correlation matrix, we may find that certain features (like alcohol content, fixed acidity, and volatile acidity) have strong correlations with wine quality. This indicates that these features could be significant predictors of the quality of wine.

-The scatter plots may reveal nonlinear relationships, suggesting that more complex models (like polynomial regression) might be necessary for accurate predictions.

-The box plot will likely show how the distribution of alcohol content varies with quality, which can inform winemakers about the desired alcohol levels for higher quality wines.

**Predictions**

1.Quality Prediction: Using features such as alcohol content and acidity levels, we could build a predictive model (e.g., linear regression, decision tree, or random forest) to predict wine quality.

2.Feature Importance: We can identify which features most significantly impact quality, guiding winemakers in adjusting their processes (e.g., fermentation time, sugar levels) to improve wine quality.

3.Market Trends: If certain types of wine (red or white) consistently score higher in quality based on specific features, this could inform marketing strategies and consumer preferences.

In summary, I analyze wine quality data that can lead to improved production practices and better quality wines.


---

#Pairwise Relationship

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""Step 1: Data Loading"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Justification:** Load the dataset to work with it in a structured format.

Step 2: Style Setting
"""

sns.set(style="whitegrid")

"""**Justification:** Setting the style improves the aesthetics of the plots by adding gridlines.

Step 3: Creating the Pairplot
"""

pairplot = sns.pairplot(data,
                        hue='quality',
                        diag_kind='kde',
                        markers='o',
                        palette='Set2',
                        height=2.5)

"""-hue='quality': Colors the points based on the quality of the wine, making it easier to see how different qualities relate to other features.

-diag_kind='kde': Uses kernel density estimates on the diagonal instead of histograms, providing a smoother representation of the distributions.

-markers='o': Specifies the shape of the markers used in the scatter plots.

-palette='Set2': Chooses a specific color palette for better visual differentiation.

-height=2.5: Sets the size of each subplot in the grid.

Step 4: Adding a Title
"""

plt.suptitle('Pairwise Relationships in Wine Quality Dataset', y=1.02, fontsize=16)

"""**Justification:** Adding a title provides context for the visualization, and adjusting the y parameter moves the title slightly above the plots for better visibility.

Step 5: Displaying the Plot
"""

plt.show()

"""**Justification:** This command renders the visualizations to the screen.

**Discussion of Results**

1.Overall Visualization:

The pairplot provides a comprehensive view of how each feature in the dataset relates to every other feature. By coloring the points based on the quality variable, we can identify trends and clusters that may exist within the data.

2.Scatter Plots:

Each scatter plot in the grid shows the relationship between two features. For example, if there is a clear upward or downward trend in the scatter plots involving alcohol and quality, it suggests a correlation between these features.
If certain quality ratings cluster in specific regions of the scatter plots (e.g., high alcohol content with high quality), it indicates that those features may be important indicators of wine quality.

3.Diagonal KDE Plots:

The kernel density estimates on the diagonal provide insights into the distribution of each feature. For instance, if the distribution of fixed acidity shows a peak at lower values, it suggests that most wines have low acidity.
Observing the distribution of quality can highlight whether the dataset is balanced across different quality ratings or if it skews toward certain ratings.

**Predictions**

1.From Alcohol and Quality:

If the analysis shows that higher alcohol content is associated with higher quality ratings, winemakers might focus on fermentation techniques that increase alcohol levels to enhance wine quality.

2.From Acidity Levels:

If lower levels of fixed acidity correlate with higher quality, it may suggest that reducing acidity during production could lead to better quality wines.

3.General Insights:

The pairplot can help identify other features that interact significantly with quality, guiding winemakers in adjusting their processes based on the factors that most influence wine quality.

In summary, I visualize pairwise relationships in a wine quality dataset using Seaborn's pairplot. The insights gained from these visualizations can guide winemakers in optimizing their processes and improving wine quality, ultimately leading to better products for consumers.


---

#Kernel PCA Implementation

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""Step 1: Loading the Dataset:"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Explanation:** The dataset is loaded into a Pandas DataFrame for analysis.

Step 2: Data Preprocessing
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

"""**Explanation:** The 'type' column is converted to binary values (if it exists), and all columns are converted to numeric types. Rows with NaN values are dropped.

**Justification:** This step ensures that the data is in a suitable format for analysis and that any non-numeric values do not interfere with the PCA process.

Step 3: Defining Features and Target Variable
"""

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** The features are separated from the target variable (quality).

Step 4: Standardizing the Features
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**Explanation:** The features are standardized to have a mean of 0 and a variance of 1.

**Justification:** Standardization is important for PCA and Kernel PCA because it ensures that all features contribute equally to the analysis, especially when they are on different scales.

Step 5: Applying Kernel PCA
"""

kpca = KernelPCA(kernel='rbf', n_components=2)
X_kpca = kpca.fit_transform(X_scaled)

"""**Explanation:** Kernel PCA is applied using the Radial Basis Function (RBF) kernel to reduce the dimensionality of the data to 2 components.

**Justification:** The RBF kernel is commonly used for its ability to handle non-linear relationships, making it suitable for complex datasets like wine quality.

Step 6: Creating a DataFrame for Transformed Features
"""

kpca_df = pd.DataFrame(data=X_kpca, columns=['PC1', 'PC2'])
kpca_df['quality'] = y.values

"""**Explanation:** A new DataFrame is created to hold the transformed features (principal components) along with the quality labels.

Step 7: Plotting the Results
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(data=kpca_df, x='PC1', y='PC2', hue='quality', palette='viridis', alpha=0.7)
plt.title('Kernel PCA of Wine Quality Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Quality')
plt.grid()
plt.show()

"""**Explanation:** A scatter plot is generated to visualize the first two principal components, colored by wine quality.

**Justification:** Visualizing the reduced dimensions helps to understand the underlying structure of the data and how different quality levels are distributed in the transformed space.

**Discussion of Results**

-The scatter plot generated from Kernel PCA provides insights into how the different wine qualities are distributed in the new feature space defined by the principal components.

-If the points are well-separated based on quality, it indicates that the features used in the analysis effectively capture the variance associated with different quality levels.

-Clusters of points corresponding to different quality ratings can suggest that certain combinations of features lead to higher or lower quality wines.

**Proposed Predictions**

1.Quality Classification: With the transformed features from Kernel PCA, further classification algorithms (like KNN, SVM, etc.) can be applied to predict wine quality based on the reduced dimensions.

2.Feature Insights: The results can provide insights into which features are most important for distinguishing between different quality levels, guiding winemakers in optimizing their production processes.

3.Market Analysis: Understanding how different wine types (red vs. white) cluster in the PCA space can inform marketing strategies and consumer preferences.

4.Further Analysis: Kernel PCA can serve as a preprocessing step for other machine learning models, potentially improving their performance by reducing noise and focusing on the most relevant features.

In summary, this Kernel PCA implementation provides a powerful way to visualize and understand the relationships between features and wine quality, paving the way for more advanced analyses and predictions.


---

#KNN Implementation
#The confusion matrix, Classification report, Accuracy score, Hyperparameter Tuning, Cross validation

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

"""Step 1: Loading the Dataset:"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Explanation:** The dataset is loaded into a Pandas DataFrame.

**Justification:** This is the first step in any data analysis process, as it allows me to work with the data in a structured format, making it easy to manipulate and analyze.

Step 2: Data Preprocessing
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

"""**Explanation:** If the dataset contains a 'type' column, it is converted to binary values (0 for white wine and 1 for red wine). Then, all columns are converted to numeric types, and any rows with NaN values are dropped.

**Justification:** Converting categorical data (like wine type) to numeric values is essential for KNN, as it requires numerical input. Dropping NaN values ensures that the model is trained on complete data, which is crucial for accurate predictions.

Step 3: Defining Features and Target Variable
"""

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** Here, X contains all features of the dataset except the target variable y, which is the 'quality' column.

**Justification:** Separating features from the target variable is a standard practice in supervised learning. This allows the model to learn the relationship between the features and the quality.

Step 4: Splitting the Data
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**Explanation:** The dataset is split into training and testing sets using an 80-20 split.

**Justification:** This split allows us to train the model on one subset of the data and evaluate its performance on a separate subset, which helps prevent overfitting and ensures the model generalizes well to new data.

Step 5: KNN Classifier Initialization
"""

knn = KNeighborsClassifier(n_neighbors=5)

"""**Explanation:** A KNN classifier is initialized with n_neighbors set to 5.

**Justification:** The choice of n_neighbors is crucial, as it determines how many nearest neighbors will be considered when making predictions. A value of 5 is commonly used as a starting point, but it can be optimized through experimentation.

Step 6: Model Fitting
"""

knn.fit(X_train, y_train)

"""**Explanation:** The KNN model is fitted to the training data.

**Justification:** This step allows the model to learn the relationships between the features and the target variable, which is necessary for making predictions.

Step 7: Making Predictions
"""

y_pred = knn.predict(X_test)

"""**Explanation:** Predictions are made on the test set.

**Justification:** Making predictions on the test set allows us to evaluate how well the model performs on unseen data, which is critical for understanding its effectiveness.

Step 8: **Model Evaluation**
"""

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

"""**Explanation:** The confusion matrix, classification report, and accuracy score are printed to evaluate the model's performance.

**Justification:** These metrics provide insights into how well the model is performing, including precision, recall, and overall accuracy. The confusion matrix specifically helps identify how many predictions were correct versus incorrect.

Step 9: Visualizing the **Confusion Matrix**
"""

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""**Explanation:** A heatmap is generated to visualize the confusion matrix.

**Justification:** Visualizing the confusion matrix provides a clearer understanding of the model's performance, allowing for quick identification of misclassifications.

Step 10: Hyperparameter Tuning (I think it can test for different values of k in some data testing case when are required, but not necessary in this case.)
"""

k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid()
plt.show()

"""**Justification:** Testing different values of k allows us to identify which number of neighbors provides the best accuracy. It’s common practice to visualize this relationship.

Step 11: Perform Cross-Validation
"""

features = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                 'density', 'pH', 'sulphates', 'alcohol']]
target = data['quality']

cv_scores = cross_val_score(knn, features, target, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

"""**Justification:** The cross_val_score function evaluates the model using cross-validation. Here, I use 5-fold cross-validation, which means the dataset is split into 5 parts, and the model is trained and tested 5 times, each time using a different part as the test set and the remaining parts as the training set. This helps to mitigate overfitting and provides a more reliable estimate of the model's performance.

**Discussion of Results**

-After running the KNN model, we will typically observe the accuracy score, which indicates the percentage of correct predictions made by the model. This score can vary based on the dataset and the chosen n_neighbors. This indicates the percentage of correctly predicted instances. A higher accuracy (e.g., above 70%) suggests that the model is performing well.

-The classification report provides detailed metrics such as precision, recall, and F1-score for each quality class, which helps assess the model's performance beyond just accuracy.

-The confusion matrix visualizes how many instances of each class were correctly classified and how many were misclassified, offering insights into specific areas where the model may be struggling.

-Cross-Validation Scores: These scores give me an idea of how the model performs across different subsets of the data. If the scores are consistent and high, it indicates that the model is robust.

-Mean Cross-Validation Score: This provides a single value that summarizes the model's performance across all folds.

**Proposed Predictions**

1.Quality Prediction: Based on the features such as acidity levels, sugar content, and alcohol content, the KNN model can predict the quality of wine. This can help winemakers understand which factors are most influential in producing high-quality wines.

2.Feature Importance: By analyzing the model's performance, we can identify which features contribute most to the predictions. For example, if alcohol content is found to have a strong correlation with quality, winemakers may focus on optimizing fermentation processes to achieve desired alcohol levels.

3.Market Trends: If the model can accurately predict quality, it may inform marketing strategies. For example, if certain types of wine (e.g., red vs. white) are consistently rated higher, producers can adjust their marketing efforts accordingly.

4.Model Optimization: Further improvements could be made by tuning hyperparameters (like n_neighbors), using cross-validation, or exploring other classification algorithms to compare performance.

In summary, this KNN implementation provides a structured approach to predicting wine quality based on various chemical properties, offering valuable insights that can enhance wine production and marketing strategies. Also, the cv helps ensure that the model generalizes well to unseen data.


---

#K-Means Clustering Model, KNN Decision Boundaries Model
#Accuracy score

Step 0: Install relevant packages and Load the relevant function
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

"""**Justification:** Importing necessary libraries allows us to handle data manipulation (pandas), numerical operations (numpy), model training and evaluation (sklearn), and visualization (matplotlib). Each library serves a specific purpose in the workflow.

Step 1: Load the Dataset
"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Justification:** Loading the dataset into a DataFrame is the first step in data analysis. It allows me to manipulate and analyze the data efficiently.

Step 2: Explore the Data
"""

print(data.dtypes)

"""**Justification:** Checking the data types helps identify non-numeric columns that may cause issues during model fitting.

Step 3: Prepare Features and Target Variable
"""

X = data.select_dtypes(include=[np.number]).drop(['quality'], axis=1).values
y = data['quality']

"""**Justification:** Here, I select only numeric features for clustering and drop the 'quality' column from the features set (X). The 'quality' column is retained as the target variable (y). This step ensures that KMeans receives only numerical data.

Step 4: Encode the Target Variable
"""

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

"""**Justification:** Encoding the target variable transforms categorical labels into numerical format, which is necessary for training the KNN model. This step allows the model to interpret the target variable correctly.

Step 5: K-Means Clustering
"""

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

"""**Justification:** I initialize the KMeans algorithm and fit it to the dataset. The n_clusters parameter is set to 4, indicating I want to find 4 clusters in the data. The random state ensures reproducibility. After fitting, I predict the cluster labels for my data.

Step 6: Plot K-Means Clustering Results
"""

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
plt.title('K-Means Clustering')

"""**Justification:** Visualizing the KMeans clustering results helps us understand how the data points are grouped. The cluster centers are highlighted in red, providing insight into the central tendencies of the clusters formed.

Step 7: K-Nearest Neighbors Classification
"""

plt.subplot(1, 2, 2)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

"""**Justification:**

-PCA: I apply PCA to reduce the dimensionality of the dataset to 2 components for visualization purposes, making it easier to plot and interpret.

-Train-Test Split: The dataset is split into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data.

-Feature Scaling: Scaling the features is essential for KNN, as it is sensitive to the magnitude of the features. StandardScaler standardizes the features to have a mean of 0 and a standard deviation of 1.

-KNN Training: I create a KNN classifier with 2 neighbors and fit it to the training data.

Step 8: Plot KNN Decision Boundaries
"""

plt.figure(figsize=(14, 6))
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, Z, cmap='coolwarm', shading='auto')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title("K-Nearest Neighbors")

"""**Justification:** This step visualizes the decision boundaries created by the KNN classifier. It helps us understand how the model classifies different regions of the feature space based on the training data.

Step 9: Evaluate the KNN Model
"""

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of K-Nearest Neighbors:", accuracy)

plt.tight_layout()
plt.show()

"""**Justification:** I predict the labels for the test set and calculate the accuracy of the KNN model. This provides a quantitative measure of how well the model performs on unseen data.

**Discussion of Results**

1.K-Means Clustering: The KMeans clustering results will show how the data points are grouped into clusters. The cluster centers indicate the average feature values of each cluster. This can provide insights into the characteristics of different wine types.

2.KNN Accuracy: The accuracy score printed at the end indicates the percentage of correct predictions made by the KNN model on the test set. A higher accuracy (e.g., above 70%) suggests that the model is performing well in classifying wine quality.

3.Visualizations: The scatter plots for KMeans and KNN provide intuitive visual insights into how the data is structured and how the KNN model makes decisions based on the training data.

**Predictions**

Based on the trained KNN model, we can make predictions about the quality of wines given their features (e.g., acidity, sugar content, alcohol level). For instance, if we have a new wine sample with specific values for these features, we can use the knn.predict method to classify its quality.

In summary, the use of PCA for dimensionality reduction and visualizations aids in interpreting the results. Future enhancements could involve hyperparameter tuning for KNN(which I did it before in KNN implementation), exploring other classification algorithms, or using ensemble methods for potentially better performance.


---

#t-SNE Implementation

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

"""Step 1: Loading the Dataset"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""Explanation: The dataset is loaded into a Pandas DataFrame for analysis.

Step 2: Data Preprocessing
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

"""**Explanation:** If the dataset contains a 'type' column, it is converted to binary values (0 for white wine and 1 for red wine). All columns are converted to numeric types, and any rows with NaN values are dropped.

**Justification:** This ensures the data is in a suitable format for analysis, preventing any non-numeric values from interfering with the t-SNE process.

Step 3: Defining Features and Target Variable
"""

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** The features are separated from the target variable (quality).

Step 4: Standardizing the Features
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**Explanation:** The features are standardized to have a mean of 0 and a variance of 1.

**Justification:** Standardization is important for t-SNE because it ensures that all features contribute equally to the analysis, especially when they are on different scales.

Step 5: Applying t-SNE
"""

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

"""**Explanation:** t-SNE is applied to reduce the dimensionality of the data to 2 components for visualization.

**Justification:** Reducing to 2 dimensions allows us to visualize the high-dimensional data in a scatter plot, which can reveal patterns and clusters.

Step 6: Creating a DataFrame for Transformed Features
"""

tsne_df = pd.DataFrame(data=X_tsne, columns=['Dimension 1', 'Dimension 2'])
tsne_df['quality'] = y.values

"""**Explanation:** A new DataFrame is created to hold the transformed features along with the quality labels.

Step 7: Plotting the Results
"""

plt.figure(figsize=(10, 6))
plt.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'], c=tsne_df['quality'], cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Wine Quality Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='Quality')
plt.grid()
plt.show()

"""**Explanation:** A scatter plot is generated to visualize the first two dimensions obtained from t-SNE, colored by wine quality.

**Justification:** Visualizing the reduced dimensions helps to understand the underlying structure of the data and how different quality levels are distributed in the transformed space.

**Discussion of Results**

-The scatter plot generated from t-SNE provides insights into how different wine qualities are distributed in the new feature space defined by the two dimensions.

-If the points corresponding to different quality ratings are well-separated, it indicates that the features used in the analysis effectively capture the variance associated with different quality levels.

-Clusters of points can suggest that certain combinations of features lead to higher or lower quality wines.

**Proposed Predictions**

1.Quality Classification: The t-SNE visualization can inform further classification algorithms (like KNN, SVM, etc.) to predict wine quality based on the reduced dimensions.

2.Feature Insights: The results can provide insights into which features are most important for distinguishing between different quality levels, guiding winemakers in optimizing their production processes.

3.Market Analysis: Understanding how different wine types (red vs. white) cluster in the t-SNE space can inform marketing strategies and consumer preferences.

4.Further Analysis: t-SNE can serve as a preprocessing step for other machine learning models, potentially improving their performance by reducing noise and focusing on the most relevant features.

In summary, this t-SNE implementation provides a powerful way to visualize and understand the relationships between features and wine quality, paving the way for more advanced analyses and predictions.


---

#ANOVA
#Residual Analysis

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scipy statsmodels

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

"""Step 1: Data Loading"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

print(data.head())

"""**Justification:** The first step is to load the dataset using pandas. This allows me to work with the data in a structured format (DataFrame). Checking the first few rows helps me understand the structure of the data, including the features and the target variable.

Step 2: Data Exploration
"""

print(data.columns)

"""**Justification:** Before performing any statistical analysis, it's essential to explore the dataset to ensure that all required columns are present and correctly named. This step helps avoid issues like KeyError when accessing columns.

Step 3: Model Fitting
"""

model = ols('alcohol ~ C(quality)', data=data).fit()

"""**Justification:** Here, I am using the Ordinary Least Squares (OLS) method to fit a linear model where alcohol is the dependent variable and quality is treated as a categorical independent variable. The C() function indicates that quality is a categorical variable, which is crucial for ANOVA, as it compares the means of different groups (in this case, wine qualities).

Step 4: ANOVA Calculation
"""

anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

"""**Justification:** The anova_lm function computes the ANOVA table, which provides information about the variance explained by the model (between-group variance) and the variance within the groups. The typ=2 argument specifies the type of sums of squares to use, which is suitable for models with unbalanced designs.

Step 5: Result Interpretation
"""

alpha = 0.05
if anova_table['PR(>F)'][0] < alpha:
    print("There is a significant difference in alcohol content between wine quality groups.")
else:
    print("There is no significant difference in alcohol content between wine quality groups.")

"""**Justification:** In this step, I assess whether the p-value (PR(>F) in the ANOVA table) is less than the significance level (alpha = 0.05). If it is, I conclude that there are significant differences in alcohol content among the different quality groups. This decision is based on the common threshold for statistical significance.

**Discussion of Results**

After running the ANOVA, the output will include an ANOVA table with the F-statistic and the p-value.

-If the p-value is less than 0.05: This indicates that at least one group mean (alcohol content for a specific quality) differs significantly from the others. This suggests that the quality of wine may be associated with its alcohol content, which can be an important factor for wine producers and consumers.

-If the p-value is greater than 0.05: This suggests that there is no significant difference in alcohol content across the quality groups, implying that alcohol level may not be a distinguishing factor for wine quality.

**Proposed Predictions**

1.If Significant Differences Exist: If we find significant differences, we could predict that wines with higher alcohol content may generally receive higher quality ratings. This could inform winemakers to adjust fermentation processes to enhance quality.

2.If No Significant Differences Exist: If no significant differences are found, it may suggest that factors other than alcohol content (such as acidity, sugar levels, etc.) are more influential in determining wine quality. Future studies could focus on these other factors.

In summary, I provide a structured approach to performing ANOVA on a wine quality dataset. The results can lead to actionable insights for winemakers and help consumers understand the relationship between wine characteristics and quality.


---

#Linear Regression Implementation
#Evaluation metrics

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

"""Step 1: Loading the Dataset"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Explanation:** Load the dataset into a Pandas DataFrame.

**Justification:** Loading the dataset is the first step in any data analysis or modeling process. It allows us to access the data we need for analysis and modeling. In this case, I am interested in the chemical properties of wine and their associated quality ratings.

Step 2: Data Preprocessing
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

"""**Explanation:** If the dataset contains a 'type' column, convert it to binary values (0 for white wine and 1 for red wine). Convert all columns to numeric types and drop NaN values.

**Justification:**

-Converting categorical variables (like wine type) into numeric values is essential for machine learning algorithms, which generally require numerical input. This transformation simplifies analysis and modeling.

-Converting all columns to numeric and dropping NaN values ensures that the dataset is clean and suitable for training a model. Missing values can lead to errors or biased results.

Step 3: Defining Features and Target Variable
"""

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** Separate the features from the target variable (quality).

**Justification:** Separating the features (input variables) from the target variable (output variable) is a standard practice in supervised learning. This allows the model to learn the relationship between the features and the quality ratings.

Step 4: Standardizing the Features
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**Explanation:** Standardize the features to have a mean of 0 and a variance of 1.

**Justification:** Standardizing the features to have a mean of 0 and a variance of 1 is important for many machine learning algorithms, including linear regression. This ensures that all features contribute equally to the model training and prevents features with larger scales from dominating the learning process.

Step 5: Splitting the Data
"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""**Explanation:** Split the dataset into training and testing sets.

**Justification:** Splitting the dataset into training and testing sets is crucial for evaluating the model's performance. Training on one subset and testing on another helps ensure that the model generalizes well to unseen data, reducing the risk of overfitting.

Step 6: Initializing the Linear Regression Model
"""

linear_model = LinearRegression()

"""**Explanation:** Initialize a Linear Regression model.

**Justification:** Initializing a Linear Regression model provides a straightforward way to model the relationship between input features and the target variable. Linear regression is often one of the first algorithms to try for regression tasks due to its simplicity and interpretability.

Step 7: Fitting the Model
"""

linear_model.fit(X_train, y_train)

"""**Explanation:** Fit the Linear Regression model to the training data.

**Justification:** Fitting the Linear Regression model to the training data allows it to learn the relationships between the features and the target variable. This step is essential for the model to make predictions.

Step 8: Making Predictions
"""

y_pred = linear_model.predict(X_test)

"""**Explanation:** Make predictions on the test set.

**Justification:** Making predictions on the test set allows us to evaluate how well the model performs on unseen data. This is critical for assessing the model's effectiveness and generalizability.

Step 9: Calculating Evaluation Metrics
"""

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

"""**Explanation:** Calculate the RMSE, MAE, and R² values to evaluate the model's performance.

**Justification:**

-RMSE provides a measure of how well the model predicts the target variable, with lower values indicating better performance. It gives more weight to larger errors.

-MAE measures the average magnitude of the errors in a set of predictions, providing a straightforward interpretation of prediction accuracy.

-R² indicates how well the model explains the variability of the target variable. An R² value close to 1 indicates a good fit, while a value close to 0 indicates that the model does not explain much of the variability.

Step 10: Printing the Metrics
"""

print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

"""**Explanation:** Print the calculated metrics.

**Justification:** Printing the calculated metrics provides a clear summary of the model's performance. This information is essential for understanding how well the model is performing and for making decisions about potential improvements.

Step 11: Visualizing Actual vs. Predicted Values
"""

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')
plt.grid()
plt.show()

"""**Explanation:** Create a scatter plot to visualize the actual vs. predicted wine quality values.

**Justification:** Visualizing the actual vs. predicted values helps to assess the model's performance visually. A scatter plot with a diagonal line indicates perfect predictions. Deviations from this line can help identify areas where the model may be underperforming.

**Discussion of Results**

-The evaluation metrics (RMSE, MAE, and R²) provide insights into the model's performance. Lower RMSE and MAE values indicate better predictive accuracy, while a higher R² value indicates that the model explains a larger proportion of the variance in the target variable.

-The scatter plot helps visualize how well the model's predictions align with the actual quality ratings.

**Proposed Predictions**

1.Quality Prediction: The Linear Regression model can be used to predict wine quality based on its features, providing a tool for winemakers to assess quality based on chemical properties.

2.Feature Insights: Insights from this analysis can guide winemakers in optimizing production processes to enhance wine quality based on the most influential features.

3.Market Analysis: Understanding the key factors influencing quality can assist in marketing strategies, emphasizing wines with desirable characteristics.

In summary, this implementation of Linear Regression provides a straightforward method for predicting wine quality, offering valuable insights that can enhance decision-making in wine production and marketing.


---

#Feature Importance Using Random Forest

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Step 1: Loading the dataset and processing the data, Defining Features and Target Variable"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** The features are separated from the target variable (quality).

Step 2: Standardizing the Features
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**Explanation:** The features are standardized to have a mean of 0 and a variance of 1.

**Justification:** Standardization is important for many machine learning algorithms, including tree-based models, to ensure that all features contribute equally.

Step 3: Splitting the Data
"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""**Explanation:** The dataset is split into training and testing sets using an 80-20 split.

**Justification:** This allows for training the model on one subset and evaluating its performance on another, which helps prevent overfitting.

Step 4: Initializing the Random Forest Classifier
"""

rf = RandomForestClassifier(n_estimators=100, random_state=42)

"""**Explanation:** A Random Forest classifier is initialized with 100 trees.

**Justification:** Random Forest is a robust ensemble learning method that can handle non-linear relationships and is effective for feature importance analysis.

Step 5: Fitting the Model
"""

rf.fit(X_train, y_train)

"""**Explanation:** The Random Forest model is fitted to the training data.

**Justification:** This step allows the model to learn the relationships between the features and the target variable.

Step 6: Getting Feature Importances
"""

importances = rf.feature_importances_

"""**Explanation:** The feature importances are extracted from the fitted Random Forest model.

**Justification:** This provides a quantitative measure of how much each feature contributes to the model's predictions.

Step 7: Creating a DataFrame for Feature Importances
"""

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

"""**Explanation:** A DataFrame is created to hold the feature names and their corresponding importance scores, sorted in descending order.

**Justification:** This makes it easier to visualize and interpret the importance of each feature.

Step 8: Plotting the Feature Importances
"""

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance for Predicting Wine Quality')
plt.gca().invert_yaxis()
plt.show()

"""**Explanation:** A horizontal bar plot is generated to visualize the feature importances.

**Justification:** Visualizing feature importance helps to quickly identify which features are most influential in predicting wine quality.

**Discussion of Results**

-The bar plot will show the relative importance of each feature in the Random Forest model. Features with higher importance scores contribute more to the model's predictions.

-Commonly important features might include "alcohol," "volatile acidity," and "fixed acidity," which are known to influence wine quality significantly.

-This analysis helps winemakers understand which characteristics are most critical for producing high-quality wines.

**Proposed Predictions**

1.Quality Classification: Based on the feature importance analysis, further classification algorithms can be applied to predict wine quality using the most important features.

2.Feature Insights: Insights from this analysis can guide winemakers in optimizing production processes to enhance wine quality based on the most influential features.

3.Market Analysis: Understanding the key features can help in marketing strategies, emphasizing wines with the desired characteristics based on consumer preferences.

In summary, this implementation of feature importance using Random Forest provides valuable insights into the factors that contribute to wine quality, aiding in better decision-making for wine production and marketing.


---

#Decision Tree
#The confusion matrix, Classification report, Accuracy score

Step 0: Install the relevant packages and Load the relevant function
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

"""**Justification:** I import pandas for data manipulation, numpy for numerical operations, and various functions from sklearn for model building and evaluation. matplotlib is used for visualizing the decision tree.

Step 1: Load the Dataset
"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Justification:** Loading the dataset into a DataFrame allows me to easily manipulate and analyze it.

Step 2: Explore the Data
"""

print(data.head())

print(data.isnull().sum())

"""**Justification:** Exploring the data helps us understand its structure and check for any missing values that need to be addressed before training the model.

Step 3: Select Features and Target Variable
"""

features = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                 'density', 'pH', 'sulphates', 'alcohol']]
target = data['quality']

"""**Justification:** I select the specified features for training and the 'quality' column as the target variable. This is critical as it defines what we want to predict.

Step 4: Split the Dataset into Training and Testing Sets
"""

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

"""**Justification:** Splitting the dataset helps me evaluate the model's performance on unseen data. I use 80% of the data for training and 20% for testing, which is a common practice.

Step 5: Create and Train the Decision Tree Classifier
"""

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

"""**Justification:** I initialize the DecisionTreeClassifier and train it on the training data. The random state ensures reproducibility of results.

Step 6: Make Predictions
"""

y_pred = clf.predict(X_test)

"""**Justification:** Once the model is trained, I use it to make predictions on the test set to evaluate its performance.

Step 7: Evaluate the Model
"""

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=features.columns, class_names=[str(i) for i in np.unique(target)])
plt.show()

"""**Justification:** Evaluating the model's accuracy and generating a classification report provides insights into its performance. A confusion matrix helps visualize the model's predictions against actual values. Visualizing the decision tree aids in understanding how decisions are made.

**Discussion of Results**

After running the above code, we will also receive an accuracy score, a classification report, and a confusion matrix.

-Accuracy: This indicates the percentage of correctly predicted instances. A higher accuracy (e.g., above 70%) suggests that the model is performing well.

-Classification Report: This includes precision, recall, and F1-score for each class, providing a deeper understanding of model performance.

-Confusion Matrix: This shows the number of correct and incorrect predictions broken down by class, helping identify any specific areas where the model may be underperforming.

**Predictions**

Based on the model, we can make predictions about wine quality given certain input characteristics. For example, if we have a wine with specific values for acidity, sugar, and alcohol content, we can predict its quality rating.

In summary, the Decision Tree Classifier is a straightforward yet powerful tool for classification tasks. Its interpretability allows for easy understanding of how features influence predictions. Future steps could include hyperparameter tuning, trying different algorithms, or using ensemble methods like Random Forests for potentially better performance.

Also, by comparison, I think it is the best classification model because it has the highest accuracy.


---

#GBM
#Evaluation metrics

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""Step 1: Data Loading"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Justification:** I load the dataset to work with csv format. The pandas library enables easy data manipulation.

Step 2: Feature and Target Selection
"""

features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
target = 'quality'

"""**Justification:** I define the features (independent variables) and the target variable (dependent variable) we want to predict. This helps in organizing the data for the model.

Step 3: Data Splitting
"""

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**Justification:** The dataset is split into training and testing sets to evaluate the model's performance. A common split is 80% for training and 20% for testing.

Step 4: Model Creation and Training
"""

gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_model.fit(X_train, y_train)

"""**Justification:**

I create a Gradient Boosting Regressor model with specified parameters:

-n_estimators: Number of boosting stages to be run.

-learning_rate: Step size shrinkage used in updating the weights.

-max_depth: Maximum depth of the individual regression estimators, controlling overfitting.

The model is then trained using the training data.

Step 5: Making Predictions
"""

y_pred = gbm_model.predict(X_test)

"""**Justification:** I use the trained model to make predictions on the test set, which allows me to evaluate its performance.

Step 6: Model Evaluation
"""

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

"""**Justification:** I also calculate the Mean Squared Error (MSE) and R² score to assess the model's accuracy here. MSE gives us an idea of the average squared difference between actual and predicted values, while R² indicates the proportion of variance explained by the model.

Step 7: Feature Importance Visualization
"""

feature_importance = gbm_model.feature_importances_
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importance')
plt.show()

"""**Justification:** Visualizing feature importance helps us understand which features contribute most to the model's predictions. This can provide insights into the underlying factors affecting wine quality.

**Discussion of Results**

-Low MSE: Indicates that the model's predictions are close to the actual values, suggesting good performance.

-High R² Score: Indicates that a significant proportion of the variance in wine quality can be explained by the features.

**Predictions**

If the model performs well (low MSE and high R²):

-We can predict that certain combinations of features (like alcohol content and acidity) are strong indicators of wine quality.

-Winemakers can use this model to optimize their production processes, focusing on the most impactful features to enhance wine quality.

If the model does not perform well:

-It may suggest that additional features or more complex models could be necessary for better predictions.

-Further analysis could involve exploring interactions between features or using a different modeling approach.

In summary, I implement a Gradient Boosting Machine to predict wine quality using relevant features. By comparison, I think this model has the most significant result, and the results can lead to actionable insights for winemakers and consumers alike.


---

#Performance Metrics : RMSE, MAE, and R² Calculation

In this part, I do a focusing part on these three calculation here to provide a specific calculation steps. Therefore, the justifications are almost the same, so that I just mark down some explanation for each step

Step 0: Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

"""Step 1: Loading the Dataset"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Explanation:** Load the dataset into a Pandas DataFrame.

Step 2: Data Preprocessing
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

"""**Explanation:** Convert categorical variables to numeric and drop NaN values.

Step 3: Defining Features and Target Variable
"""

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** Separate features from the target variable.

Step 4: Standardizing the Features
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**Explanation:** Standardize the features to have a mean of 0 and a variance of 1.

Step 5: Splitting the Data
"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""**Explanation:** Split the dataset into training and testing sets.

Step 6: Initializing the Decision Tree Regressor
"""

dt_regressor = DecisionTreeRegressor(random_state=42)

"""**Explanation:** Initialize a Decision Tree regressor.

Step 7: Fitting the Model
"""

dt_regressor.fit(X_train, y_train)

"""**Explanation:** Fit the Decision Tree model to the training data.

Step 8: Making Predictions
"""

y_pred = dt_regressor.predict(X_test)

"""**Explanation:** Make predictions on the test set.

Step 9: Calculating RMSE, MAE, and R²
"""

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

"""**Explanation:** Calculate the Root Mean Square Error, Mean Absolute Error, and R-squared values.

Step 10: Printing the Metrics
"""

print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

"""**Explanation:** Print the calculated metrics to evaluate the model's performance.

**Discussion of Results**

-RMSE: This metric provides a measure of how well the model predicts the target variable, with lower values indicating better performance. It gives more weight to larger errors.

-MAE: This metric measures the average magnitude of the errors in a set of predictions, without considering their direction. It is a linear score, which means all individual differences are weighted equally.

-R²: This value indicates how well the model explains the variability of the target variable. An R² value closer to 1 indicates a better fit, while a value closer to 0 indicates that the model does not explain much of the variability.

**Proposed Predictions**

1.Quality Prediction: The model can be used to predict wine quality based on the features, providing a tool for winemakers to assess quality based on chemical properties.

2.Feature Insights: The analysis can reveal which features are most influential in predicting quality, guiding improvements in production processes.

3.Market Analysis: Understanding the key factors influencing quality can assist in marketing strategies, emphasizing wines with desirable characteristics.

Overall, these metrics will help understand the performance of the Decision Tree model(did it before) in predicting wine quality, allowing for further refinement and optimization of the model.

---

#Support Vector Classification (SVC)
#The confusion matrix, Classification report, Accuracy score

Step 1: Install the relevant packages and Import Necessary Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

"""Justification: I import libraries for data manipulation (pandas), numerical operations (numpy), model training and evaluation (sklearn), and visualization (matplotlib). Each library serves a specific purpose in the workflow.

Step 2: Load the Dataset
"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""Justification: Loading the dataset into a DataFrame is the first step in data analysis, allowing me to manipulate and analyze the data efficiently.

Step 3: Prepare Features and Target Variable
"""

features = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                 'density', 'pH', 'sulphates', 'alcohol']]
target = data['quality']

"""Justification: I specify the features (input variables) and the target variable (wine quality). This step is critical as it defines what I want to predict.

Step 4: Encode the Target Variable
"""

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target)

"""Justification: Encoding the target variable transforms categorical labels into numerical format, which is necessary for training the SVC model. This allows the model to interpret the target variable correctly.

Step 5: Split the Dataset into Training and Testing Sets
"""

X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

"""Justification: Splitting the dataset helps me evaluate the model's performance on unseen data. We use 80% of the data for training and 20% for testing, which is a common practice.

Step 6: Scale the Features
"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""Justification: Feature scaling is important for SVC, as it is sensitive to the scale of the features. StandardScaler standardizes the features to have a mean of 0 and a standard deviation of 1, which improves the performance of the model.

Step 7: Create and Train the SVC Model
"""

svc = SVC(kernel='linear', random_state=42)

svc.fit(X_train, y_train)

"""Justification: I initialize the SVC model with a linear kernel. The model is then trained on the training data.

Step 8: Make Predictions
"""

y_pred = svc.predict(X_test)

"""Justification: After training the model, I use it to make predictions on the test set to evaluate its performance.

Step 9: Evaluate the Model
"""

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVC: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

"""Justification: Evaluating the model's accuracy and generating a classification report provides insights into its performance. A confusion matrix helps visualize the model's predictions against actual values.

**Discussion of Results**

-Accuracy: The accuracy printed at the end indicates the percentage of correct predictions made by the SVC model on the test set. A higher accuracy (e.g., above 70%) suggests that the model is performing well in classifying wine quality.

-Classification Report: This includes precision, recall, and F1-score for each class, providing a deeper understanding of model performance.

0Confusion Matrix: This shows the number of correct and incorrect predictions broken down by class, helping identify any specific areas where the model may be underperforming.

**Predictions**

Based on the trained SVC model, we can make predictions about the quality of wines given their features (e.g., acidity, sugar content, alcohol level). For example, if we have a new wine sample with specific values for these features, we can use the svc.predict method to classify its quality.

IN summary, I apply Support Vector Classification (SVC) to analyze the wine quality dataset. We can use it for exploring other kernels, or using ensemble methods for potentially better performance.


---

#XGBoost
#Evaluation Metrics

Step 0: Install the relevant packages and Import Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""**Justification:**

-pandas and numpy are essential for data manipulation and numerical operations.

-matplotlib and seaborn are used for data visualization.
train_test_split from sklearn helps in splitting the dataset into training and testing sets.

-StandardScaler is used for feature scaling.

-XGBRegressor is the model we will use for prediction.

-mean_squared_error and r2_score are metrics to evaluate the model's performance.

Step 1: Load the Dataset
"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Justification:** This step is crucial to bring the data into our working environment for analysis.

Step 2: Data Exploration
"""

print(data.dtypes)

data = pd.get_dummies(data, columns=['type'], drop_first=True)

print(data.isnull().sum())

print(data.describe())

"""**Justification:** Understanding the data's structure and checking for missing values helps us identify any necessary data cleaning steps.

Step 3: Data Visualization
"""

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""**Justification:**

-The correlation heatmap helps in identifying which features are most correlated with the target variable (quality).

-The count plot shows the distribution of the quality ratings, which is important for understanding the target variable's balance.

Step 4: Data Preprocessing
"""

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""**Justification:**

-Dropping the target variable (quality) from features ensures we only use the relevant predictors.

-Splitting the data allows me to train the model on one set and test it on another, preventing overfitting.

-Scaling the features ensures that all features contribute equally to the model's performance.

Step 5: Train the XGBoost Model
"""

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

model.fit(X_train, y_train)

"""**Justification:**

-XGBoost is a powerful gradient boosting algorithm that works well with tabular data.

-Setting n_estimators and learning_rate helps control the model's complexity and performance.

Step 6: Model Evaluation
"""

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

"""**Justification:**

-Mean Squared Error (MSE) gives me an idea of how far off our predictions are from the actual values.

-R² Score indicates how well my model explains the variability of the target variable.

Step 7: Predictions and show visualization
"""

importance = model.feature_importances_

feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

"""**Justification:** Understanding which features are most important can guide future data collection and model improvements.

**Results Discussion**

-MSE: A lower value indicates a better fit.

-R² Score: A value close to 1 indicates that the model explains a significant portion of the variance in the quality ratings.

**Future Predictions**

Based on the model, we can predict the quality of new wine samples by inputting their chemical properties into the trained model. This can help winemakers adjust their processes to improve wine quality.

In summary, depending on the performance metrics, we can consider further tuning the model or exploring additional features for better predictions.

By comparison, this is the best model to calcute the performance metrics because the R^2 is the highest and the effective predictive model. Also, it shows the most signicant different of feature importances.


---

#Multi-Layer Perceptron (MLP)
#The confusion matrix, Classification report, Accuracy score

Step 1: Install the relevant packages and Import Necessary Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""Justification: Importing necessary libraries allows us to handle data manipulation (pandas), numerical operations (numpy), model training and evaluation (sklearn), and preprocessing.

Step 2: Load the Dataset
"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""Justification: Loading the dataset into a DataFrame is the first step in data analysis, allowing me to manipulate and analyze the data efficiently.

Step 3: Prepare Features and Target Variable
"""

features = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                 'density', 'pH', 'sulphates', 'alcohol']]
target = data['quality']

"""Justification: I specify the features (input variables) and the target variable (wine quality). This step is critical as it defines what we want to predict.

Step 4: Encode the Target Variable
"""

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target)

"""Justification: Encoding the target variable transforms categorical labels into numerical format, which is necessary for training the MLP model. This allows the model to interpret the target variable correctly.

Step 5: Split the Dataset into Training and Testing Sets
"""

X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

"""Justification: Splitting the dataset helps us evaluate the model's performance on unseen data. I use 80% of the data for training and 20% for testing, which is a common practice.

Step 6: Scale the Features
"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""Justification: Feature scaling is important for MLP, as it is sensitive to the scale of the input features. StandardScaler standardizes the features to have a mean of 0 and a standard deviation of 1, which improves the performance of the model.

Step 7: Create and Train the MLP Model
"""

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

mlp.fit(X_train, y_train)

"""Justification: I initialize the MLP model with one hidden layer containing 100 neurons and set the maximum number of iterations for training to 500. The random state ensures reproducibility. The model is then trained on the training data.

Step 8: Make Predictions
"""

y_pred = mlp.predict(X_test)

"""Justification: After training the model, I use it to make predictions on the test set to evaluate its performance.

Step 9: Evaluate the Model
"""

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of MLP: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

"""Justification: Evaluating the model's accuracy and generating a classification report provides insights into its performance. A confusion matrix helps visualize the model's predictions against actual values.

**Discussion of Results**

-Accuracy: The accuracy printed at the end indicates the percentage of correct predictions made by the MLP model on the test set. A higher accuracy (e.g., above 70%) suggests that the model is performing well in classifying wine quality.

-Classification Report: This includes precision, recall, and F1-score for each class, providing a deeper understanding of model performance.

-Confusion Matrix: This shows the number of correct and incorrect predictions broken down by class, helping identify any specific areas where the model may be underperforming.

**Predictions**

Based on the trained MLP model, we can make predictions about the quality of wines given their features (e.g., acidity, sugar content, alcohol level). For instance, if we have a new wine sample with specific values for these features, we can use the mlp.predict method to classify its quality.

In summary, I also applies a Multi-Layer Perceptron (MLP) to analyze the wine quality dataset. We can also experiment with different architectures, or using ensemble methods for potentially better performance.

---

#CNN Implementation

Step 0: Install the relevant packages and Load the relevant functions
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install tensorflow pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

"""Step 1: Loading the Dataset"""

data = pd.read_csv('/content/wine-quality-white-and-red_20864751.csv')

"""**Explanation:** Load the dataset into a Pandas DataFrame.

Step 2: Data Preprocessing
"""

if 'type' in data.columns:
    data['type'] = data['type'].map({'white': 0, 'red': 1})
data = data.apply(pd.to_numeric, errors='coerce').dropna()

"""**Explanation:** Convert categorical variables to numeric and drop NaN values.

Step 3: Defining Features and Target Variable
"""

X = data.drop('quality', axis=1)
y = data['quality']

"""**Explanation:** Separate the features from the target variable.

Step 4: Standardizing the Features
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""Explanation: Standardize the features to have a mean of 0 and a variance of 1.

Step 5: Reshaping Features for CNN
"""

X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1, 1)

"""**Explanation:** Reshape the features to fit the input shape expected by the CNN (samples, height, width, channels). Here, I treat the features as a "1D image."

**Justification:** Reshaping the features allows us to simulate a 2D input (like an image) that a CNN expects. By treating each feature set as a "1D image," we can leverage the capabilities of CNNs, which are generally designed for spatial data. This step is somewhat unconventional for tabular data but serves to demonstrate how CNNs can also be applied.

Step 6: Converting Target Variable to Categorical
"""

y_categorical = to_categorical(y)

"""**Explanation:** Convert the target variable to a categorical format for multi-class classification.

**Justification:** Converting the target variable to a categorical format is necessary for multi-class classification problems. This allows the model to output probabilities for each class (quality rating) and enables the use of the softmax activation function in the output layer.

Step 7: Splitting the Data
"""

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

"""**Explanation:** Split the dataset into training and testing sets.


**Justification:** Splitting the dataset into training and testing sets is crucial for evaluating the model's performance. Training on one subset and testing on another helps ensure that the model generalizes well to unseen data, reducing the risk of overfitting.

Step 8: Initializing the CNN Model
"""

model = Sequential()

"""**Explanation:** Initialize a sequential model for the CNN.

**Justification:** A sequential model is a straightforward way to build a neural network layer by layer. This approach is suitable for our task of predicting wine quality.

Step 9: Adding Convolutional Layers
"""

model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(Dropout(0.5))

"""**Explanation:** Add convolutional layers followed by dropout layers to prevent overfitting.

**Justification:**

-Convolutional layers are designed to automatically learn spatial hierarchies of features, making them powerful for tasks involving images or structured data. Using 1x1 convolutions allows the model to learn feature interactions in a compact manner.

-Dropout layers are included to prevent overfitting by randomly dropping a fraction of the neurons during training, encouraging the model to learn more robust features.

Step 10: Flattening the Output
"""

model.add(Flatten())

"""**Explanation:** Flatten the output from the convolutional layers to feed into fully connected layers.

**Justification:** Flattening the output from the convolutional layers is necessary to convert the 2D feature maps into a 1D array that can be fed into fully connected layers. This step prepares the data for the next stages of the network.

Step 11: Adding Fully Connected Layers
"""

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

"""**Explanation:** Add fully connected layers, with the final layer using softmax activation for multi-class classification.

**Justification:** Fully connected layers allow the model to learn complex relationships between features. The first dense layer uses ReLU activation to introduce non-linearity, while the output layer uses softmax activation to output class probabilities for each wine quality rating.

Step 12: Compiling the Model
"""

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""**Explanation:** Compile the model with the Adam optimizer and categorical crossentropy loss function.

**Justification:** Compiling the model with the Adam optimizer and categorical crossentropy loss function is standard for multi-class classification tasks. Adam is an efficient optimizer that adapts the learning rate during training, and categorical crossentropy is appropriate for measuring the performance of a model whose output is a probability value between 0 and 1.

Step 13: Fitting the Model
"""

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

"""**Explanation:** Fit the model to the training data.

**Justification:** Training the model on the training dataset allows it to learn the relationships between the features and the target variable. The use of validation data helps monitor the model's performance during training to prevent overfitting.

Step 14: Evaluating the Model
"""

loss, accuracy = model.evaluate(X_test, y_test)

"""**Explanation:** Evaluate the model on the test set.

**Justification:** Evaluating the model on the test set provides insights into how well it generalizes to new, unseen data. This step is crucial for understanding the model's effectiveness.

Step 15: Making Predictions
"""

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

"""**Explanation:** Make predictions on the test set and convert them to class labels.

**Justification:** Making predictions on the test set allows us to assess the model's output. Using np.argmax converts the predicted probabilities into class labels, which is necessary for comparison with the true labels.

Step 16: Print a sample of predictions
"""

y_test_classes = np.argmax(y_test, axis=1)

print("Sample Predictions:", y_pred_classes[:10])
print("Sample True Labels:", y_test_classes[:10])

"""**Discussion of Results**

-The evaluation metrics (loss and accuracy) provide insights into the model's performance. Higher accuracy indicates better predictive performance.

-The CNN architecture may not be optimal for tabular data, but it can still provide insights into how deep learning models can be applied to structured datasets.

**Proposed Predictions**

1.Quality Prediction: The CNN model can be used to classify wine quality based on its features, providing a tool for winemakers to assess quality based on chemical properties.

2.Feature Insights: Insights from this analysis can guide winemakers in optimizing production processes to enhance wine quality based on the most influential features.

3.Market Analysis: Understanding the key factors influencing quality can assist in marketing strategies, emphasizing wines with desirable characteristics.

In summary, this implementation of a CNN provides a way to explore deep learning techniques on structured data, although it's generally more common to use traditional machine learning models for such tasks.


---

#Chemical models of an important Catalyst in wine and some explanation of my works

1. Understanding Catalysts in Wine Production
Role of Catalysts: Catalysts are substances that increase the rate of a chemical reaction without being consumed in the process. In winemaking, certain catalysts can facilitate reactions that improve the flavor, aroma, and overall quality of the wine.
Example: Tartaric acid is a natural component of grapes and plays a crucial role in stabilizing wine. It helps maintain acidity, which is essential for the taste and preservation of wine.
2. Chemical Reactions and Wine Quality
Fermentation: During fermentation, yeast converts sugars into alcohol and carbon dioxide, and various chemical reactions occur. Catalysts can enhance these reactions, affecting the production of esters and phenols, which contribute to the wine's aroma and flavor profile.
Malolactic Fermentation: This is a secondary fermentation process where lactic acid bacteria convert malic acid into lactic acid and carbon dioxide. Catalysts can help facilitate this reaction, softening the wine's acidity and enhancing its complexity.
3. Stability and Preservation
Tartaric Stabilization: Tartaric acid helps prevent the formation of potassium bitartrate crystals, which can occur when wine is chilled. Understanding the chemistry of tartaric acid and its interactions can help winemakers prevent these crystals from forming, ensuring a clearer and more aesthetically pleasing wine.
Oxidation Prevention: Catalysts can also play a role in preventing oxidation reactions that can spoil wine. By controlling these reactions, winemakers can maintain the wine's intended flavor and aroma.
4. Quality Assessment through Chemical Analysis
Chemical Profiling: By creating chemical models of catalysts and other compounds in wine, researchers can analyze how different factors (such as temperature, pH, and concentration) affect the chemical reactions during fermentation and aging. This helps in understanding how these factors influence the final wine quality.
Predictive Modeling: Using computational chemistry and machine learning, models can predict how changes in the winemaking process (like varying the amount of a catalyst) will affect the wine's chemical composition and sensory attributes (taste, smell, mouthfeel).
5. Sensory Analysis Correlation
Flavor and Aroma: The chemical compounds produced during fermentation and aging (like esters and phenols) are responsible for the wine's flavor and aroma. By understanding the role of catalysts in producing these compounds, winemakers can adjust their processes to enhance desirable characteristics.
Consumer Preferences: By analyzing the chemical profiles of wines that are well-received by consumers, winemakers can use this information to guide their choices in using specific catalysts and fermentation techniques to produce wines that align with market preferences.

Step 0: Install the relevant packages and import the relevant function
"""

!pip install rdkit-pypi matplotlib

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

"""Step 1: Define Reactants and Products Using SMILES"""

tartaric_acid_smiles = "C(C(=O)O)(C(C(=O)O)O)C(C(=O)O)O"
malic_acid_smiles = "C(C(=O)O)C(C(=O)O)C(C(=O)O)O"

"""**Justification:**

SMILES Representation: By defining the reactants and products in this format, we can create molecule objects for further manipulation and visualization.

Step 2: Create Molecules from SMILES
"""

tartaric_acid = Chem.MolFromSmiles(tartaric_acid_smiles)
malic_acid = Chem.MolFromSmiles(malic_acid_smiles)

"""**Justification:**

Molecule Creation: The MolFromSmiles function converts the SMILES strings into RDKit molecule objects as it allows us to perform chemical operations and visualizations on these structures.

Step 4: Transition state
"""

transition_state_smiles = "C(C(=O)O)(C(=O)O)C(C(=O)O)O"

transition_state = Chem.MolFromSmiles(transition_state_smiles)

"""**Justification:** Define a simple transition state (for illustration purposes)

Step 3: Draw the Reactants and Products
"""

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

img1 = Draw.MolToImage(tartaric_acid)
ax[0].imshow(img1)
ax[0].set_title('Tartaric Acid')
ax[0].axis('off')

img2 = Draw.MolToImage(transition_state)
ax[1].imshow(img2)
ax[1].set_title('Transition State')
ax[1].axis('off')

img3 = Draw.MolToImage(malic_acid)
ax[2].imshow(img3)
ax[2].set_title('Malic Acid')
ax[2].axis('off')

plt.suptitle('Transformation Process: Tartaric Acid to Malic Acid', fontsize=16)
plt.show()

"""**Justification:**

Visualization: I create subplots to display the chemical structures of tartaric acid and malic acid side by side. This visual representation helps in understanding the structural differences between the reactant and product, which is essential for analyzing the chemical reaction.

**Discussion of Results**

-The visual output consists of two images representing tartaric acid and malic acid. This representation highlights the transformation from tartaric acid, a common wine component, to malic acid, which can occur during fermentation processes in winemaking.

-Structural Differences: By examining the structures, one can observe how tartaric acid, which has multiple carboxylic acid groups, can lose or rearrange functional groups to form malic acid. This is a typical transformation in wine chemistry, where acids play a crucial role in flavor and stability.

**Predictions**

-Impact on Wine Quality: The conversion of tartaric acid to malic acid can influence the acidity and flavor profile of the wine. Malic acid is generally perceived as having a sharper taste compared to tartaric acid, which can affect the overall sensory experience of the wine.

-Fermentation Dynamics: Understanding this reaction can help winemakers predict how changes in fermentation conditions (e.g., temperature, yeast strains) can impact the final wine composition. For instance, if conditions favor the conversion of tartaric acid to malic acid, the wine may have a different taste profile.

-Future Research Directions: Further studies could involve modeling additional reactions in winemaking, analyzing how different catalysts affect these transformations, and exploring the sensory implications of these changes. This could lead to more tailored winemaking processes that enhance desired flavor characteristics.

In summary, I show the catalyst model by SMILES, by understanding the molecular transformations that occur during winemaking, we can gain insights into how these processes affect wine quality and flavor, ultimately benefiting winemakers and consumers alike.


---

Below, I create the 3D figures of them, just add some new elements for visualiztion. Therefore, I did not explain each step because this part is not crucial. Just have a look~
"""

!pip install rdkit-pypi
!pip install py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

tartaric_acid_smiles = "C(C(=O)O)(C(C(=O)O)O)C(C(=O)O)O"

tartaric_acid = Chem.MolFromSmiles(tartaric_acid_smiles)

AllChem.EmbedMolecule(tartaric_acid)
AllChem.UFFOptimizeMolecule(tartaric_acid)

block = Chem.MolToPDBBlock(tartaric_acid)

viewer = py3Dmol.view(width=800, height=400)
viewer.addModel(block, "pdb")
viewer.setStyle({'stick': {}})
viewer.setBackgroundColor('white')
viewer.zoomTo()
viewer.show()

!pip install rdkit-pypi
!pip install py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

malic_acid_smiles = "C(C(=O)O)C(C(=O)O)C(C(=O)O)O"

malic_acid = Chem.MolFromSmiles(malic_acid_smiles)

AllChem.EmbedMolecule(malic_acid)
AllChem.UFFOptimizeMolecule(malic_acid)

malic_block = Chem.MolToPDBBlock(malic_acid)

viewer = py3Dmol.view(width=1000, height=400)

viewer.addModel(malic_block, "pdb")
viewer.setStyle({'stick': {'colorscheme': 'orangeCarbon'}})

viewer.setBackgroundColor('white')
viewer.zoomTo()
viewer.show()

!pip install rdkit-pypi
!pip install py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

transition_state_smiles = "C(C(=O)O)(C(=O)O)C(C(=O)O)O"
transition_state = Chem.MolFromSmiles(transition_state_smiles)

AllChem.EmbedMolecule(transition_state)
AllChem.UFFOptimizeMolecule(transition_state)

transition_block = Chem.MolToPDBBlock(transition_state)

viewer = py3Dmol.view(width=1000, height=400)

viewer.addModel(transition_block, "pdb")
viewer.setStyle({'stick': {'colorscheme': 'greenCarbon'}})

viewer.setBackgroundColor('white')
viewer.zoomTo()
viewer.show()
