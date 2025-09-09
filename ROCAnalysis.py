import numpy as np

class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_predicted = y_predicted
        self.y_true = y_true

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        #--- Write your code here ---#
         # Calculate true positives (TP): both predicted and true are 1
        true_positives = np.sum((self.y_predicted == 1) & (self.y_true == 1))
        
        # Calculate false negatives (FN): predicted 0, true 1
        false_negatives = np.sum((self.y_predicted == 0) & (self.y_true == 1))
        
        # Compute recall: TP / (TP + FN), handle division by zero
        denominator = true_positives + false_negatives
        recall = true_positives / denominator if denominator > 0 else 0.0
        
        # Return recall if positive, else 0.0
        return recall if recall > 0 else 0.0

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        #--- Write your code here ---#
        
        false_positives = np.sum((self.y_predicted == 1) & (self.y_true == 0))
        true_negatives = np.sum((self.y_predicted == 0) & (self.y_true == 0))
        denominator = false_positives + true_negatives
        return false_positives / denominator if denominator > 0 else 0.0


    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        #--- Write your code here ---#
        true_positives = np.sum((self.y_predicted == 1) & (self.y_true == (1)))
        false_positives = np.sum((self.y_predicted == (1)) & (self.y_true == (0)))
        denominator = true_positives + false_positives
        return true_positives / denominator if denominator > 0 else 0.0
  
    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        #--- Write your code here ---#
            # Calculate true positives (TP): both predicted and true are 1
        true_positives = np.sum((self.y_predicted == 1) & (self.y_true == 1))
        
        # Calculate false positives (FP): predicted 1, true 0
        false_positives = np.sum((self.y_predicted == 1) & (self.y_true == 0))
        
        # Calculate false negatives (FN): predicted 0, true 1
        false_negatives = np.sum((self.y_predicted == 0) & (self.y_true == 1))
        
        # Compute precision: TP / (TP + FP), handle division by zero
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        # Compute recall: TP / (TP + FN), handle division by zero
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # Compute F-score: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        denominator = beta**2 * precision + recall
        f_score = ((1 + beta**2) * precision * recall) / denominator if denominator > 0 else 0.0
        
        return f_score
