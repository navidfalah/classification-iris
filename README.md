# 🌸 Iris Classification with K-Nearest Neighbors (KNN)

Welcome to the **Iris Classification** repository! This project demonstrates how to classify iris flowers using the **K-Nearest Neighbors (KNN)** algorithm. It covers data loading, preprocessing, model training, and evaluation.

---

## 📂 **Project Overview**

This repository includes the following steps:
- **Loading the Iris Dataset**: Exploring the dataset and splitting it into training and test sets.
- **Data Visualization**: Using scatter plots to visualize relationships between features.
- **Model Training**: Building a KNN classifier.
- **Model Evaluation**: Testing the model on unseen data and calculating accuracy.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**

---

## 📊 **Dataset**

The project uses the **Iris Dataset**, which contains 150 samples of iris flowers, each with 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is the species of iris (setosa, versicolor, or virginica).

---

## 🧠 **Key Concepts**

### 1. **K-Nearest Neighbors (KNN)**
- A simple and effective classification algorithm.
- Predicts the class of a sample based on the majority class of its `k` nearest neighbors.

### 2. **Data Splitting**
- Dividing the dataset into training and test sets to evaluate model performance.

### 3. **Model Evaluation**
- Calculating accuracy to measure how well the model performs on unseen data.

---

## 🚀 **Code Highlights**

### Loading the Dataset
```python
from sklearn.datasets import load_iris
iris_dataset = load_iris()
```

### Splitting the Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
```

### Visualizing the Data
```python
from pandas.plotting import scatter_matrix
scatter_matrix(iris_df, c=Y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
```

### Training the KNN Model
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
```

### Making Predictions
```python
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction:", iris_dataset['target_names'][prediction])
```

### Evaluating the Model
```python
y_pred = knn.predict(X_test)
test_score = np.mean(y_pred == Y_test)
print("Test accuracy:", test_score)
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/iris-classification.git
   cd iris-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook classification_iris.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
