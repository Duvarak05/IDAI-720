import numpy as np

class Metrics:
    def __init__(self, y, y_pred):
        self.y = np.array(y)
        self.y_pred = np.array(y_pred)

    def acc(self):
        return 1.0 - np.sum(np.abs(self.y - self.y_pred)) / len(self.y)

    def eod(self, s):
        s = np.array(s)
        # Group masks
        s1 = s == 1
        s0 = s == 0
        # TPR for s=1: among y=1 & s=1, how many predicted 1
        tpr1 = np.sum((self.y == 1) & (self.y_pred == 1) & s1) / np.sum((self.y == 1) & s1)
        # TPR for s=0: among y=1 & s=0, how many predicted 1
        tpr0 = np.sum((self.y == 1) & (self.y_pred == 1) & s0) / np.sum((self.y == 1) & s0)
        return tpr1 - tpr0

    def aod(self, s):
        s = np.array(s)
        s1 = s == 1
        s0 = s == 0
        # TPR for s=1 and s=0
        tpr1 = np.sum((self.y == 1) & (self.y_pred == 1) & s1) / np.sum((self.y == 1) & s1)
        tpr0 = np.sum((self.y == 1) & (self.y_pred == 1) & s0) / np.sum((self.y == 1) & s0)
        # FPR for s=1 and s=0
        fpr1 = np.sum((self.y == 0) & (self.y_pred == 1) & s1) / np.sum((self.y == 0) & s1)
        fpr0 = np.sum((self.y == 0) & (self.y_pred == 1) & s0) / np.sum((self.y == 0) & s0)
        return (tpr1 - tpr0 + fpr1 - fpr0) / 2.0

    def spd(self, s):
        s = np.array(s)
        s1 = s == 1
        s0 = s == 0
        # PR = predicted positive rate within each group
        pr1 = np.sum((self.y_pred == 1) & s1) / np.sum(s1)
        pr0 = np.sum((self.y_pred == 1) & s0) / np.sum(s0)
        return np.abs(pr1 - pr0)
