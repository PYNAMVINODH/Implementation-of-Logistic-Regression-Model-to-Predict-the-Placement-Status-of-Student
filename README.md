# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed libraries.
2. Read the Placement_data.csv file.And load the dataset.
3. Check the null values and duplicate values.
4. train and test the predicted value using logistic regression
5. calculate confusion_matrix,accuracy,classification_matrix and predict.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PYNAM VINODH
RegisterNumber:  212223240131
*/
```
## Head
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/4e2ad5dd-8e76-4ff1-843e-4351a71601e4)


## Copy
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/8befbb37-bc50-477b-b5a3-6cc293526cba)


## Null Duplicated
```
data1.isnull().sum()
data1.duplicated().sum()
```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/14f8ce8c-6878-4a88-ae0d-100f2df85a28)

## Label Encoder
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print(data1)
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/18e1c569-680c-4059-9b66-a76b1fe7dcd1)

## Dependent Value
```
y=data1["status"]
y
```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/7e382e98-4846-49f0-adb0-6966a9951a44)


## Logistic Regression,accuracy,confusion_matrix
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Array:\n",confusion)

```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/5e99e231-3070-41a4-80e5-93f68ce30564)

![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/18f3186d-613e-4700-b08b-ee8d55a092ee)

![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/cbcb8914-a880-4fef-a75e-5a0a31df06fb)

## Classification_Matrix,Lr:
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/PYNAMVINODH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742678/afc2cf70-e047-46a6-969e-57c56d408dce)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
