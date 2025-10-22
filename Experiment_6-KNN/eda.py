import matplotlib.pyplot as plt
import numpy as np
from data import load_iris_data, load_wine_data

def eda_iris():
    X, y = load_iris_data()
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    unique_species = np.unique(y)
    colors = ['red', 'green', 'blue']

    print("=== Iris Dataset EDA ===")
    print("Classes:", unique_species)
    print("Feature shape:", X.shape)

    plt.figure(figsize=(10, 8))
    plot_index = 1
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            plt.subplot(2, 3, plot_index)
            for species, color in zip(unique_species, colors):
                plt.scatter(
                    X[y == species, i],
                    X[y == species, j],
                    label=species,
                    alpha=0.7,
                    color=color
                )
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.title(f"{features[i]} vs {features[j]}")
            plot_index += 1

    plt.legend()
    plt.suptitle("Iris Dataset - Feature Pair Relationships", fontsize=14)
    plt.tight_layout()
    plt.show()


def eda_wine():
    """Exploratory Data Analysis for the Wine dataset"""
    X, y = load_wine_data()
    feature_names = [
        "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
        "Total phenols", "Flavanoids", "Nonflavanoid phenols",
        "Proanthocyanins", "Color intensity", "Hue",
        "OD280/OD315 of diluted wines", "Proline"
    ]
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

    print("\n=== Wine Dataset EDA ===")
    print("Classes:", unique_classes)
    print("Feature shape:", X.shape)

    plt.figure(figsize=(12, 10))
    pairs = [(0, 1), (0, 9), (6, 9), (9, 12)] 
    for idx, (i, j) in enumerate(pairs, 1):
        plt.subplot(2, 2, idx)
        for cls, color in zip(unique_classes, colors):
            plt.scatter(
                X[y == cls, i],
                X[y == cls, j],
                label=f"Class {cls}",
                alpha=0.7,
                color=color
            )
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.title(f"{feature_names[i]} vs {feature_names[j]}")
    plt.legend()
    plt.suptitle("Wine Dataset - Selected Feature Relationships", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    eda_iris()
    eda_wine()
