import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Classifier")

st.write("Enter flower features to predict the Iris species:")

# Load data and model
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

# Input from user
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = iris.target_names[prediction][0]

st.success(f"Predicted Species: **{predicted_species}**")
