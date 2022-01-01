
import pandas as pd

data =pd.read_csv("diabetes.csv")
data.head()

data.describe() 
data.isnull()

data.columns
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]
X
y =data['Outcome']
y

y.value_counts()

from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(X ,y , test_size =0.2  , random_state =42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

k =10
knn =KNeighborsClassifier(k)
knn.fit(X_train ,y_train)

y_pred = knn.predict(X_test)
y_pred

print('The accuracy  is '.format((accuracy_score(y_test, y_pred)*100)))
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.2, solver='liblinear')
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
y_pred
print("The Acuracy is ". format(accuracy_score(y_test ,y_pred) *100))
from sklearn.tree import DecisionTreeClassifier
outcomeTree = DecisionTreeClassifier(criterion="entropy", max_depth =2)
outcomeTree.fit(X_train, y_train)
y_pred = outcomeTree.predict(X_test)
y_pred
print('The accuracy is'.format((accuracy_score(y_test, y_pred))*100))

