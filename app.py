import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="K-Means Clustering Demo", layout="centered")

st.title("📊 K-Means Clustering Example")
st.write("This app demonstrates K-Means clustering on student marks data.")

# Dataset
X = np.array([
    [40, 45],
    [50, 55],
    [60, 65],
    [70, 75],
    [75, 78],
    [80, 82],
    [78, 76],
    [90, 92],
])

st.subheader("Dataset")
st.write("Maths Marks vs Science Marks")
st.dataframe(
    {"Maths": X[:, 0], "Science": X[:, 1]}
)

# User input
k = st.slider("Select number of clusters (k)", min_value=2, max_value=5, value=4)

# KMeans model
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Plot
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels)
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)
ax.set_xlabel("Maths Marks")
ax.set_ylabel("Science Marks")
ax.set_title("K-Means Clustering Result")

st.pyplot(fig)

st.subheader("Cluster Labels")
st.write(labels)
