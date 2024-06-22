import numpy as np
import pandas as pd
from logistic_regression import logistic_regression as lr

class oneVsRest:
    def __init__(self):
        self._classifiers = None
        self._number_of_class = None
        self._classes = None

    def fit(self,x,y):
        ynp= y.to_numpy().ravel()
        self._classes = np.unique(ynp)
        self._number_of_class = len(self._classes)
        self._classifiers = []
        for cls in self._classes:
            classifier = lr()
            y_cls=pd.DataFrame([1 if ans == cls else -1 for ans in y])
            classifier.fit(x,y_cls)
            self._classifiers.append(classifier)

    def predict_proba(self, X):
        # Initialize a matrix to store probabilities
        probas = np.zeros((X.shape[0], len(self._classes)))

        # Iterate through each classifier and compute probabilities
        for idx, classifier in enumerate(self._classes):
            probas[:, idx] = classifier.predict_proba(X)[:, 1]

        return probas

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self._classes)))
        for idx, classifier in enumerate(self._classifiers):
            scores[:, idx] = classifier.predict_proba(X)
        test=self._classes[np.argmax(scores, axis=1)]
        return test

    def score(self, x, y):
        y_pred = self.predict(x)
        return self.accuracy_score(y, y_pred)


    def accuracy_score(self,y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions



