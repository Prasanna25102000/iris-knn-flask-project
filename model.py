from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

def predict_flower(sepal_length,
                   sepal_width,
                   petal_length,
                   petal_width):

    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    return iris.target_names[prediction[0]]