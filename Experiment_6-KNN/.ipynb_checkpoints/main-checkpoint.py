import numpy as np
import matplotlib.pyplot as plt
from data import load_iris_data, load_wine_data
from utils import train_test_split
from knn_classifier import KNNClassifier

X, y = load_iris_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Iris Dataset Results ->")
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

for k in k_values:
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    accuracies.append(acc)
    print(f"K = {k:<2} → Accuracy: {acc * 100:.2f}%")

best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K for Iris dataset: {best_k}")
print(f"Highest Accuracy: {max(accuracies) * 100:.2f}%\n")

# Plot accuracy vs k
plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs K (Iris Dataset)")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

print("Wine Dataset Results -> ")
Xw, yw = load_wine_data()
Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=42)

wine_accuracies = []
for k in k_values:
    wine_model = KNNClassifier(k=k)
    wine_model.fit(Xw_train, yw_train)
    wine_preds = wine_model.predict(Xw_test)
    wine_acc = np.mean(wine_preds == yw_test)
    wine_accuracies.append(wine_acc)
    print(f"K = {k:<2} → Accuracy: {wine_acc * 100:.2f}%")

best_k_wine = k_values[np.argmax(wine_accuracies)]
print(f"\nBest K for Wine dataset: {best_k_wine}")
print(f"Highest Accuracy: {max(wine_accuracies) * 100:.2f}%\n")

plt.plot(k_values, wine_accuracies, marker='o', color='orange')
plt.title("Accuracy vs K (Wine Dataset)")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

print("Summary")
print(f"Iris Dataset → Best K = {best_k}, Accuracy = {max(accuracies)*100:.2f}%")
print(f"Wine Dataset → Best K = {best_k_wine}, Accuracy = {max(wine_accuracies)*100:.2f}%")
