from sklearn.linear_model import LinearRegression
import numpy as np 
import matplotlib.pyplot as plt
#create some dummy data 
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 4])
#create a linear regression object
lr = LinearRegression()
#reshape X to a 2D array
x=x.reshape(-1,1)
#fit the model to the data 
lr.fit(x,y)
#Predict on new data
x_new = np.array([6, 7, 8]).reshape(-1,1)
y_pred = lr.predict(x_new)
#print the coefficients and intercept and the predicted values 
print("coefficients: ", lr.coef_)
print("intercept : ", lr.intercept_)
print("predicted values: ", y_pred)
#visualization of linear regression
#create some dummy data 
x = np.random.rand(50)
y = 2*x + 1 + np.random.rand(50)
#fit a linear regression object
models = LinearRegression().fit(x.reshape(-1,1), y)
#Predict on new data
x_new = np.linspace(0,1,100)
y_pred = models.predict(x_new.reshape(-1,1))
#print the coefficients and intercept and the predicted values 
print("coefficients: ", models.coef_)
print("intercept : ", models.intercept_)
print("predicted values: ", y_pred)
#plot the data and the linear regression model 
plt.scatter(x, y, label='data')
plt.plot(x_new, y_pred, color='red', label='linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression")
plt.legend()
plt.show()
#plot the data 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(x, y, label='data')
ax1.plot(x_new, models.predict(x_new.reshape(-1,1)), color='red', label = 'Linear Regression')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax2.scatter(x, y, label='data')
ax2.scatter(x_new, y_pred, color='pink', label='predictions')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
plt.title("Prediction using Linear Regression")
plt.show()
#Section 2: Logistic Regression
from sklearn.linear_model import LogisticRegression
#generate some sample data
x= np.random.normal(size=(100,2))
y=(x[:, 0] + x[:, 1]>0).astype(int)
#fit the logistic regression model
model = LogisticRegression().fit(x,y)
#make predictions on new data
x_new = np.random.normal(size=(50,2))
y_pred = model.predict(x_new)
#plot the data and the logistic regression model
plt.scatter(x[:,0], x[:,1], c=y)
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
xx , yy= np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
zz = zz.reshape(xx.shape)
plt.contour(xx, yy, zz, levels=[0.5], colors= 'red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Logistic Regression")
plt.show()
#othe presentation of Logistic regression 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Generate some sample data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,
                           random_state=42, n_clusters_per_class=1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Fit the logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
# Print the coefficients and intercept
print("Coefficients: ", clf.coef_)
print("Intercept: ", clf.intercept_)
# Make predictions on the testing data
y_pred = clf.predict(X_test)
# Print the predictions
print("Predictions: ", y_pred)
# Plot the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
# Plot the decision boundary
x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                        np.arange(x2_min, x2_max, 0.1))
Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='viridis')
# Add axis labels and a title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
# Show the plot
plt.show()
#decision tree 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)
# Fit the decision tree classifier to the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = clf.predict(X_test)
# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
report = classification_report(y_test, y_pred)
print("Classification report: \n", report)
# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True)
plt.title("Decision Tree")
plt.show()
#random forest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
# Generate a random dataset for classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=42)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an instance of the RandomForestClassifier class
clf = RandomForestClassifier(n_estimators=100)
# Train the random forest classifier on the training data
clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = clf.predict(X_test)
# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
# Print the classification report
print(classification_report(y_test, y_pred))
# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.title("Random Forest Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
#Support vector machine(SVM)
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.title("Support Vector machines")
plt.show()
# Create a grid of points to evaluate the SVM classifier
xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Create a contour plot of the SVM decision boundary
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# Plot the training data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.title("Support Vector machines")
plt.show()
#KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load the Iris dataset
iris = load_iris()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
# Fit the model to the training data
knn.fit(X_train, y_train)
# Predict labels for the testing data
y_pred = knn.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

print('Classification Report:')
print(classification_report(y_test, y_pred))
#visualization KNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# Generate a random 2D dataset with 2 classes
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the data
knn.fit(X, y)

# Plot the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("KNN")
plt.show()
#K-mean clustering 
from sklearn.cluster import KMeans
# generate a random dataset of 100 points with 2 features
X = np.random.rand(100, 2)
# instantiate the KMeans class with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
# fit the model to the data
kmeans.fit(X)
# predict the cluster labels for the data
y_pred = kmeans.predict(X)
# print the cluster centroids
print("the cluster centroids: ",kmeans.cluster_centers_)
# print the inertia of the model
print("the inertia of the model : ", kmeans.inertia_)
# plot the data points and color them based on their predicted cluster labels
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plot the cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title("K-mean clustering")
plt.show()
#principal component analysis 
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create a PCA instance
pca = PCA(n_components=2)

# Fit the data to the PCA model
iris_pca = pca.fit_transform(iris.data)

# Plot the results
plt.scatter(iris_pca[:, 0], iris_pca[:, 1], c=iris.target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("principal composant analysis")
plt.show()
#Feature scaling
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
# Load the iris dataset
iris = load_iris()
# Get the features and target variable
X = iris.data
y = iris.target
# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Print the scaled features
print(X_scaled)
import matplotlib.pyplot as plt

# Create a figure with one row and two columns
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original iris data in the first subplot
axs[0].scatter(X[:, 0], X[:, 1], c=y)
axs[0].set_title('Original Iris Data')
axs[0].set_xlabel('Sepal Length')
axs[0].set_ylabel('Sepal Width')

# Plot the scaled iris data in the second subplot
axs[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
axs[1].set_title('Scaled Iris Data')
axs[1].set_xlabel('Scaled Sepal Length')
axs[1].set_ylabel('Scaled Sepal Width')

# Set the title for the entire figure
fig.suptitle('Difference between Original and Scaled Iris Data')

plt.show()
#feature encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a small dataset
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Red', 'Green']
})

# Create a LabelEncoder object
encoder = LabelEncoder()

# Apply the encoder to the categorical feature
data['Color'] = encoder.fit_transform(data['Color'])

# Print the encoded feature
print(data['Color'])
# Plot the original categorical feature
plt.subplot(1, 2, 1)
plt.hist(data['Color'])
plt.title('Original Data')

# Plot the encoded categorical feature
plt.subplot(1, 2, 2)
plt.hist(encoder.inverse_transform(data['Color']))
plt.title('Encoded Data')

# Show the plot
# Set the title for the entire figure
plt.suptitle('Difference between original data and encoded data')
plt.show()
#feature imputation 
from sklearn.impute import SimpleImputer
import numpy as np

# Create a sample dataset with missing values
X = np.array([[1, 2, np.nan], [3, np.nan, 4], [5, 6, np.nan], [7, 8, 9]])

# Instantiate the SimpleImputer class with mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the dataset
imputer.fit(X)

# Transform the dataset by replacing missing values with the mean
X_imputed = imputer.transform(X)

print(X_imputed)
#dimensionality reduction 
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()

# Create PCA object with 2 components
pca = PCA(n_components=2)

# Fit and transform the data to 2 components
X_pca = pca.fit_transform(iris.data)

# Plot the transformed data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("Dimensionality reduction")
plt.show()
#feature selection :
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# apply feature selection with SelectKBest
k_best = SelectKBest(f_classif, k=10)
X_train_k_best = k_best.fit_transform(X_train, y_train)
X_test_k_best = k_best.transform(X_test)

# train a logistic regression model on the reduced feature set
logreg = LogisticRegression()
logreg.fit(X_train_k_best, y_train)

# make predictions on the test set and calculate accuracy
y_pred = logreg.predict(X_test_k_best)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")
import matplotlib.pyplot as plt

# plot the scores for each feature
plt.bar(range(len(k_best.scores_)), k_best.scores_)
plt.xticks(range(len(data.feature_names)), data.feature_names, rotation='vertical')
plt.xlabel('Features')
plt.ylabel('F-value')
plt.title('SelectKBest Feature Scores')
plt.show()

#categorical data 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# create a toy dataset with categorical features
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['small', 'medium', 'large', 'medium', 'small'],
    'price': [10, 20, 30, 25, 15]
})

# define the categorical features and apply one-hot encoding
cat_features = ['color', 'size']
transformer = ColumnTransformer([('onehot', OneHotEncoder(), cat_features)], remainder='passthrough')
X = transformer.fit_transform(data)

# print the transformed data
print(X)
#example2 
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# create a toy dataset with a categorical feature
data = pd.DataFrame({
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

# define the categorical feature and apply label encoding
cat_feature = 'size'
encoder = LabelEncoder()
data[cat_feature] = encoder.fit_transform(data[cat_feature])

# print the transformed data
print(data)
#prediction 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load the iris dataset
data = load_iris()
X = data.data
y = data.target

# preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# load the trained model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make predictions on the test set
y_pred = logreg.predict(X_test)

# evaluate the predictions
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:\n{cm}")

cr = classification_report(y_test, y_pred)
print(f"Classification report:\n{cr}")

