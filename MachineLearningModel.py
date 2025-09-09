from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import math

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

def polynomial_features(X: np.ndarray, exp: int, ones=True):
    """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one. 
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:
        X (array-like): Features of the data.

        Returns:
        X_poly (array-like): Polynomial features.
    """
    n_samples, n_features = X.shape

    # Initialize output with bias term if ones=True
    X_poly = np.ones((n_samples, 1)) if ones else np.empty((n_samples, 0))

    # Generate power patterns for each degree from 1 to exp
    for degree in range(1, exp + 1):
        patterns = get_polynomial_features_indeces(degree, n_features)
        for pattern in patterns:
            term = np.ones(n_samples)
            for index in pattern:
                term *= X[:, index]
            X_poly = np.hstack((X_poly, term.reshape(-1, 1)))

    return X_poly

def get_polynomial_features_indeces(exp, feature_count):
    def run(power_patterns, exp, feature_count, start=0, depth=1, feature_pattern=None):
        if feature_pattern is None:
            feature_pattern = []

        if depth == exp:
            for i in range(start, feature_count):
                current_pattern = deepcopy(feature_pattern)
                current_pattern.append(i % feature_count)
                power_patterns.append(current_pattern)
        else:
            for i in range(start, feature_count):
                new_pattern = deepcopy(feature_pattern)
                new_pattern.append(i % feature_count)
                run(power_patterns, exp, feature_count, start=i, depth=depth + 1, feature_pattern=new_pattern)

    power_patterns = []
    run(power_patterns, exp, feature_count)
    return power_patterns

        

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.betas = None

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, self.degree)
        XtX = Xmod.T @ Xmod
        Xty = Xmod.T @ y
        self.betas = np.linalg.inv(XtX) @ Xty

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        Xmod = np.c_[np.ones((len(X), 1)), X]
        return Xmod @ self.betas

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        return ((y-X)**2).sum()/len(X)

class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, self.degree)
        y = y.reshape(-1)

        n_samples, n_features = Xmod.shape
        alpha = 2 * self.learning_rate / n_samples

        self.betas = np.zeros(n_features)
        self.cost_per_iteration = []

        for _ in range(self.num_iterations):
            predictions = Xmod @ self.betas
            errors = predictions - y
            gradient = Xmod.T @ errors
            self.betas -= alpha * gradient

            cost = (errors.T @ errors) / n_samples
            self.cost_per_iteration.append(cost.item())

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, self.degree)
        return Xmod @ self.betas    

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        return (((y-X)**2).sum())/len(X)
        

class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, 1)  # Adds bias term (constant feature)
        n_samples, n_features = Xmod.shape

        self.betas = np.zeros(n_features, dtype=np.float64)
        self.cost_per_iteration = []

        for _ in range(self.num_iterations):
            predictions = self._sigmoid(Xmod)
            errors = predictions - y.reshape(-1)
            
            cost = self._cost_function(Xmod, y)
            self.cost_per_iteration.append(cost)

            gradient = (Xmod.T @ errors) / n_samples
            self.betas -= self.learning_rate * gradient

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, 1)
        return self._sigmoid(Xmod)

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        #--- Write your code here ---#
        Xround = np.round(X).reshape(-1, 1)
        accuracy = np.mean(Xround == y)
        return accuracy

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1/(1+np.exp(-np.dot(z, self.betas)))

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        theta = 1e-15
        sigmoid_X = self._sigmoid(X)
        term1 = y.T @ np.log(sigmoid_X + theta)
        term2 = (1 - y).T @ np.log(1 - sigmoid_X + theta).reshape(-1, 1)
        cost = - (term1 + term2) / len(X)
        return cost
    
class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, self.degree)  # adds bias term
        n_samples, n_features = Xmod.shape

        self.betas = np.zeros(n_features, dtype=np.float64)
        self.cost_per_iteration = []

        for _ in range(self.num_iterations):
            predictions = self._sigmoid(Xmod)
            error = predictions - y.reshape(-1)
            
            cost = self._cost_function(Xmod, y)
            self.cost_per_iteration.append(cost)
            
            gradient = (Xmod.T @ error) / n_samples
            self.betas -= self.learning_rate * gradient

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        Xmod = polynomial_features(X, self.degree)
        return self._sigmoid(Xmod)

    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        Xround = np.round(X).reshape(-1, 1)
        accuracy = np.mean(Xround == y)
        return accuracy

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1 / (1 + np.exp(-z @ self.betas))

    def mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        #--- Write your code here ---#

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        theta = 1e-15
        sigmoid_X = self._sigmoid(X)
        term1 = y.T @ np.log(sigmoid_X + theta)
        term2 = (1 - y).T @ np.log(1 - sigmoid_X + theta).reshape(-1, 1)
        cost = - (term1 + term2) / len(X)
        return cost

if __name__ == "__main__":
    for i in range(1,4):
        print(get_polynomial_features_indeces(i,3))





