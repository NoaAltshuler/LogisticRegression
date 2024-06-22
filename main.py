from logistic_regression import logistic_regression as lr
from text_vectorize import vectorize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
from oneVsrest import oneVsRest


def question2():
    df = pd.read_csv('spam_ham_dataset.csv', delimiter=',')
    x = vectorize(df)
    x[101] = np.ones(x.shape[0])
    y = df['label_num']
    y = y*2-1
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=35)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    q2=lr(l_rate=0.01,thresh=0.001,confidence=0.8)
    q2.fit(x_train_scaled,y_train)
    print(pd.DataFrame(q2._weights))
    print("train set score is: ",q2.score(x_train_scaled,y_train))
    print("test set score is: ",q2.score(x_test_scaled,y_test))
    print("")
    tpr,fpr = q2.roc_curve(x_test_scaled,y_test)
    q2.plot_roc_curve(tpr,fpr)
    print("question 3 explanation: spam is labeled as 1 i decided to measure the rate of the correct spam tagging.")
    print("the answer for the best threshold is the biggest diffrence between tpr (true spam label rate) and npr (false spam label rate) ")

def question4():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    x['bais'] = np.ones(x.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=37)
    q4 = oneVsRest()
    q4.fit(x_train,y_train)
    print("\nquestion 4: train score::", q4.score(x_train, y_train))
    print("question 4: test score::",q4.score(x_test,y_test))



question2()
question4()



