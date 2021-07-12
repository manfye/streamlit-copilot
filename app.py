import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# load iris.csv into dataframe
df = pd.read_csv('iris.csv')

# print the dataframe column names and shape
st.write(df.columns)
st.write(df.shape)

# create a scatter plot of the petalLengthCm with SepalLengthCm using plotly express
st.write(px.scatter(x='SepalLengthCm', y='PetalLengthCm', data_frame=df))

# splitting the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], df[['Species']], test_size=0.2, random_state=42)

# check for optimal K value using testing set
k_range = list(range(1, 20))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
# st.line_chart(k_range, scores)

# ME: what is the optimal K value?
# AI：The optimal K value is 5.

# st.write(scores)
# st.write(k_range)

# create a classifier using the optimal K value
knn = KNeighborsClassifier(n_neighbors=5)

# fit the classifier to the training data and predict the test set
knn.fit(X_train, y_train)

# predict the test set using the knn classifier
y_pred = knn.predict(X_test)

# calculate the accuracy of the knn classifier
st.write(metrics.accuracy_score(y_test, y_pred))

# calculate the confusion matrix using the knn classifier
st.write(metrics.confusion_matrix(y_test, y_pred))

# calculate the classification report using the knn classifier
st.write(metrics.classification_report(y_test, y_pred))

# calculate the f1 score of the knn classifier
# st.write(metrics.f1_score(y_test, y_pred))

# calculate the precision score of the knn classifier
# st.write(metrics.precision_score(y_test, y_pred))
