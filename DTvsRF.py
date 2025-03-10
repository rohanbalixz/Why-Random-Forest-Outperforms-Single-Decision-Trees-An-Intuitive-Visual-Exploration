import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

# Create a synthetic dataset (make_moons for non-linear structure)
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

def plot_decision_boundary(ax, clf, X, y, title=""):
    """Helper function to plot a classifier's decision boundary."""
    # Predict class labels for every point in the mesh grid.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define color maps for regions and data points.
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    # Plot decision regions.
    ax.contourf(xx, yy, Z, alpha=0.5, cmap=cmap_light)
    # Plot training points.
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax.set_title(title)

# Left subplot: Static Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X, y)

# Right subplot: Incrementally growing Random Forest
rf_clf = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42)

# Create a figure with two subplots.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(ax1, tree_clf, X, y, title="Decision Tree")

def update(frame):
    # Increase the number of trees (at least one tree is needed).
    n_estimators = frame + 1  
    rf_clf.set_params(n_estimators=n_estimators)
    rf_clf.fit(X, y)
    
    ax2.clear()  # Clear the previous frame.
    plot_decision_boundary(ax2, rf_clf, X, y, title=f"Random Forest: {n_estimators} Trees")
    return ax2,

# Animate over a range of number of trees.
ani = FuncAnimation(fig, update, frames=range(1, 75), interval=200, repeat=False)
plt.tight_layout()
plt.show()
