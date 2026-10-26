"""
K-means Clustering: A Small, Runnable Example

"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    # Create a reproducible sample dataset with three natural groups.
    
    X, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.75,
        random_state=42,
    )

    k = 3

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)


    print(f"Number of data points: {len(X)}")
    print(f"Number of clusters (k): {k}")
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Silhouette score: {silhouette_score(X, labels):.3f}")
    print("Cluster centers:")
    for cluster_id, center in enumerate(model.cluster_centers_):
        print(f"  Cluster {cluster_id}: ({center[0]:.2f}, {center[1]:.2f})")
    print("First 10 assigned cluster labels:")
    print(labels[:10])

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=35, alpha=0.8)
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=220,
        label="Cluster centers",
    )
    plt.title("K-means Clustering (k = 3)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=150)
    print("Plot saved as kmeans_clusters.png")
    plt.show()


if __name__ == "__main__":
    main()
