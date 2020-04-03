### Importing necessary modules
import os
try:
    import pandas as pd
except:
    os.system("sudo pip install pandas")
try:
    import seaborn as sns
except:
    os.system("sudo pip install seaborn")
try:
    import numpy as np
except:
    os.system("sudo pip install numpy")
try:
    import matplotlib.pyplot as plt
    import matplotlib.style 
except:
    os.system("sudo pip install matplotlib")
try:
    from sklearn.utils import shuffle
except:
    os.system("sudo pip install scikit-learn")

####

dataset=pd.read_csv("FlightDelays.csv")

## Checking for null values
total=dataset.isnull().sum().sort_values(ascending=False)
per=dataset.isnull().sum()/dataset.isnull().count()*100
per=per.sort_values(ascending=False)
print("Null Values in each feature:")
nullValues=pd.concat([total, per], axis=1, keys=['Total Null Values', 'Percentage'])
print(nullValues)
## 


## Random shuffling
dataset=shuffle(dataset)
##


## mapping categories
status={"ontime":0, "delayed":1}
carriers=dataset["CARRIER"].unique()
carrierDict={}
i=0
for carr in carriers:
    carrierDict[carr]=i
    i=i+1

origin=dataset["ORIGIN"].unique()
originDict={}
i=0
for org in origin:
    originDict[org]=i
    i=i+1

dest=dataset["DEST"].unique()
destDict={}
i=0
for des in dest:
    destDict[des]=i
    i=i+1

dataset["CARRIER"]=dataset["CARRIER"].map(carrierDict)
dataset["Flight Status"]=dataset["Flight Status"].map(status)
dataset["ORIGIN"]=dataset["ORIGIN"].map(originDict)
dataset["DEST"]=dataset["DEST"].map(destDict)
##

## Visualization
plt.figure(num=1, figsize=(15, 15))
plt.subplot(4,2,1)
sns.barplot(x='CARRIER', y='Flight Status', data=dataset)
plt.subplot(4,2,2)
sns.barplot(x='ORIGIN', y='Flight Status', data=dataset)
plt.subplot(4,2,3)
sns.barplot(x='DEST', y='Flight Status', data=dataset)
plt.subplot(4,2,4)
sns.barplot(x='Weather', y='Flight Status', data=dataset)
plt.subplot(4,2,5)
sns.barplot(x='DAY_WEEK', y='Flight Status', data=dataset)
plt.subplot(4,2,6)
sns.barplot(x='DAY_OF_MONTH', y='Flight Status', data=dataset)
plt.subplot(4,2,7)
sns.distplot(dataset["DEP_TIME"][dataset["Flight Status"]==1], bins=18)
sns.distplot(dataset["CRS_DEP_TIME"][dataset["Flight Status"]==1], bins=18)
plt.legend(["Departure time", "CRS Departure time"])
plt.subplot(4,2,8)
sns.distplot((dataset["DEP_TIME"]-dataset["CRS_DEP_TIME"])[dataset["Flight Status"]==1], bins=30)
plt.show()
##

## Creating a new feature
timediff=np.array(dataset["DEP_TIME"]-dataset["CRS_DEP_TIME"])
test=np.array(dataset["DEP_TIME"])
test=test//100
dataset["DEP_TIME"]=test
##

## Preparing feature vector

y=np.array(dataset["Flight Status"])
carrier=np.array(dataset["CARRIER"])
deptime=np.array(dataset["DEP_TIME"])
weekday=np.array(dataset["DAY_WEEK"])
weather=np.array(dataset["Weather"])
origin=np.array(dataset["ORIGIN"])
dest=np.array(dataset["DEST"])
x0=np.ones(np.size(y))


## Fitting Decision tree first time

X=np.vstack([ carrier, origin, dest, timediff, weekday, weather])
X=X.T

## Breaking dataset into 60:40 ratio
datasize=y.size
trainsize=int(datasize*0.6)
df_train=X[0:trainsize, :]
y_train=y[0:trainsize]
df_test=X[trainsize:datasize, :]
y_test=y[trainsize:datasize]
##


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

    

clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
acc=np.mean(y_pred==y_test)
print("Decision Tree with Initial Feature Vector")
print("Accuracy:" ,acc*100)
print("F1 score:", f1_score(y_test, y_pred, average='macro'))

## Fitting decision tree on best 4 features

X=np.vstack([carrier, dest, timediff, weekday])
X=X.T

## Breaking dataset into 60:40 ratio
datasize=y.size
trainsize=int(datasize*0.6)
df_train=X[0:trainsize, :]
y_train=y[0:trainsize]
df_test=X[trainsize:datasize, :]
y_test=y[trainsize:datasize]
##

clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
acc=np.mean(y_pred==y_test)
print("Decision Tree with 4 most Important Features")
print("Accuracy:" ,acc*100)
print("F1 score:", f1_score(y_test, y_pred, average='macro'))

## Fitting Logistic Regression on top 4 features

originVerbose=np.zeros((y.size, len(originDict)))
originVerbose[np.arange(y.size), origin]=1

carrierVerbose=np.zeros((y.size, len(carrierDict)))
carrierVerbose[np.arange(y.size), carrier]=1

destVerbose=np.zeros((y.size, len(destDict)))
destVerbose[np.arange(y.size), dest]=1

weekdayVerbose=np.zeros((y.size, 7))
weekdayVerbose[np.arange(y.size), weekday-1]=1

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
timediffScaled=scale.fit_transform(timediff.reshape(y.size, 1))

deptimeVerbose=np.zeros((y.size, deptime.max()-deptime.min()+1))
deptimeVerbose[np.arange(y.size), deptime]=1

timediffIntervals=timediff/25
timediffIntervals[timediffIntervals<=0]=0

timediffVerbose=np.zeros((y.size, timediffIntervals.max()-timediffIntervals.min()+1))
timediffVerbose[np.arange(y.size), timediffIntervals]=1

X=np.hstack((carrierVerbose, destVerbose, timediffVerbose, weekdayVerbose, weather.reshape(y.size, 1)))

## Breaking dataset into 60:40 ratio
datasize=y.size
trainsize=int(datasize*0.6)
df_train=X[0:trainsize, :]
y_train=y[0:trainsize]
df_test=X[trainsize:datasize, :]
y_test=y[trainsize:datasize]
##

from simpleLogistic import *
dim=X.shape[1]+1
classes=1
initialWeights=0.001*np.random.randn(dim, classes)
cost_history, weights=logistic_train(df_train, y_train, initialWeights, reg=0.065, learning_rate=6e-2, iterations=2000)
plt.plot(cost_history)
plt.xlabel("epochs")
plt.ylabel("Cost")
plt.show()
accu=logistic_accuracy(df_test, y_test, weights, threshold=0.5)
y_pred=logistic_predict(df_test, weights, threshold=0.5)
print("Logistic Regression")
print("Accuracy:", accu)
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(df_train, y_train)
predictions = logisticRegr.predict(df_test)
print("Logistic Regression (Sklearn Algorithm)")
print("Accuracy", np.mean(predictions==y_test))
print("F1 Score", f1_score(y_test, predictions, average='macro'))


## Fitting Linear SVM

from linear_svm import *
dim=X.shape[1]+1
classes=np.max(y)+1
initialWeights=0.001*np.random.randn(dim, classes)
initial=np.zeros(dim*classes).reshape(dim, classes)
cost_history, weights=svm_train(df_train, y_train, initialWeights, reg=0.105, learning_rate=6e-2, iterations=2000)
plt.plot(cost_history)
plt.xlabel("epochs")
plt.ylabel("Cost")
plt.show()
accu=svm_accuracy(df_test, y_test, weights)
y_pred=svm_predict(df_test, weights)
print("Linear SVM")
print("Accuracy:", accu)
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

from sklearn import svm

clf = svm.SVC(kernel='linear') 
clf.fit(df_train, y_train)
y_pred = clf.predict(df_test)
accu=np.mean(y_pred==y_test)
print("Linear SVM (Sklearn Algorithm)")
print("Accuracy", accu)
print("F1 Score", f1_score(y_test, y_pred, average='macro'))