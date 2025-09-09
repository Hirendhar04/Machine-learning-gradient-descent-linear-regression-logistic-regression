import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from MachineLearningModel import MachineLearningModel
from MachineLearningModel import polynomial_features

def plotDecisionBoundary(X, y, model : MachineLearningModel, exp, title):
    """
    Plots the decision boundary for a binary classification model along with the training data points.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y (array-like): Target labels.
        model (object): Trained binary classification model with a `predict` method.

    Returns:
        None
    """
    #--- Write your code here ---#

        # Define mesh grid step size and boundaries
    step_size = 0.01
    margin = 0.1
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Create mesh grid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )

    # Prepare input for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict probabilities and convert to binary classes
    probabilities = model.predict(mesh_points)
    binary_classes = (probabilities > 0.5).reshape(xx.shape)

    # Define colormaps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, binary_classes, cmap=cmap_light, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, marker='.', edgecolors='k', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()