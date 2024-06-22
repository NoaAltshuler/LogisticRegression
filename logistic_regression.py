import numpy as np
import matplotlib.pyplot as plt

class logistic_regression:
    def __init__(self,l_rate=0.01, thresh=0.1, max_iter= 2000, confidence=0.5):
        self._weights = None
        self._l_rate = l_rate
        self._threshold = thresh
        self._max_iter = max_iter
        self._confidence = confidence

    def predict(self, x):
        """ classify to the binary classes"""
        y_predicted = self.predict_proba(x)
        y_predicted = [1 if pred > self._confidence else -1 for pred in y_predicted]
        return np.array(y_predicted)

    def sigmoid(self, x):
        """ to limit the value to prevent overflow"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, x):
        dot_product = np.dot(x,self._weights)
        return self.sigmoid(dot_product)

    def score(self, x, y):
        y_predicted = self.predict(x)
        y=y.to_numpy().ravel()
        score = np.sum(y == y_predicted) / len(y_predicted)
        return score


    def gradiant(self,x,y):
        y = y.to_numpy().ravel()
        a = -y*self.sigmoid(-y*(x@self._weights))
        return np.dot(a,x)

    def gradiant_decent(self,x,y):
        inter_count = 0
        g_weights= self.gradiant(x,y)
        """ while the nome if the norm of the gradiant close to zero"""
        while np.linalg.norm(g_weights) > self._threshold and inter_count < self._max_iter:
            self._weights = self._weights - self._l_rate*g_weights
            g_weights = self.gradiant(x,y)
            inter_count+=1


    def fit(self,x,y):
        n_features = x.shape[1]
        """ init weights to zero"""
        self._weights = np.zeros(n_features)
        self._bais = 1
        self.gradiant_decent(x,y)

    def roc_curve(self,x,y):
        y = y.to_numpy().ravel()
        thresholds = np.linspace(0,1,100)
        y_proba = self.predict_proba(x)
        tpr, fpr = [], []
        """ calculate the tpr and fpr per thershold"""
        for threshold in thresholds:
            y_pred = [1 if y_prob >threshold else -1 for y_prob in y_proba]
            tp = np.sum((y==y_pred) & (y==1))
            tn = np.sum((y == y_pred) & ( y == -1))
            fp = np.sum((y != y_pred) & (y == -1))
            fn = np.sum((y !=y_pred) & (y == 1))
            tpr.append(tp/(tp+fn))
            fpr.append((fp/(fp+tn)))
        best_thershold_arg = np.argmax(np.array(tpr)-np.array(fpr))
        best_treshold = thresholds[best_thershold_arg]
        print("question 3: best thresh hold: ",best_treshold)
        print ("explanation : ")
        """ go back"""
        return tpr,fpr

    def plot_roc_curve(self,tpr, fpr):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()




