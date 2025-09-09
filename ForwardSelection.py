import numpy as np
from ROCAnalysis import ROCAnalysis
from MachineLearningModel import MachineLearningModel
from copy import deepcopy

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        #--- Write your code here ---#
        self.X = X
        self.y = y
        self.model : MachineLearningModel = model
        self.selected_features = []
        self.best_cost : float = 0
        self.rand_indices = None

    def create_split(self, X : np.array, y : np.array):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        #--- Write your code here ---#
            # Ensure y is a 2D column vector
        y = y.reshape(-1, 1)
        
        # Calculate split index for 75% training data
        data_len = len(X)
        train_size = round(0.75 * data_len)
        
        # Generate or use random indices for shuffling
        if self.rand_indices is None:
            self.rand_indices = np.random.permutation(data_len)
        
        # Shuffle data using indices
        X_shuffled = X[self.rand_indices]
        y_shuffled = y[self.rand_indices]
        
        # Split data into train and test sets
        X_train = X_shuffled[:train_size, :]
        y_train = y_shuffled[:train_size, :]
        X_test = X_shuffled[train_size:, :]
        y_test = y_shuffled[train_size:, :]
        
        return X_train, y_train, X_test, y_test

    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        #--- Write your code here ---#
            # Select features from data
        X_selected = self.X[:, features]
        
        # Split data into training and testing sets
        X_train, y_train, X_test, y_test = self.create_split(X_selected, self.y)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Generate predictions and round to binary classes
        y_pred = np.round(self.model.predict(X_test))
        
        # Evaluate using ROC analysis
        roc = ROCAnalysis(y_pred, y_test)
        return roc.f_score()

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        #--- Write your code here ---#
        # Initialize set of remaining feature indices
        remaining_features = set(range(self.X.shape[1]))

        while remaining_features:
            # Evaluate each remaining feature
            scores = []
            for feature in remaining_features:
                # Create trial feature set
                trial_features = deepcopy(self.selected_features)
                trial_features.append(feature)
                # Compute F-score for trial feature set
                score = self.train_model_with_features(trial_features)
                scores.append((score, feature))

            # Select the best feature from this iteration
            scores.sort(reverse=True, key=lambda x: x[0])
            best_score, best_feature = scores[0]

            # Check if the best score improves the current best cost
            if best_score > self.best_cost:
                self.selected_features.append(best_feature)
                self.best_cost = best_score
                remaining_features.remove(best_feature)
            else:
                # Stop if no improvement
                break
                
    def fit(self):
        """
        Fits the model using the selected features.
        """
        #--- Write your code here ---#
        X_selected = self.X[:, self.selected_features]
        self.model.fit(X_selected, self.y)

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        #--- Write your code here ---#
        return self.model.predict(X_test)
