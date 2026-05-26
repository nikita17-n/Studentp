import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

a = pd.read_csv("student_performance.csv")

# print(a.isnull().sum())

en = LabelEncoder()

a['grade'] = en.fit_transform(a['grade'])

s = StandardScaler()

# print(a)

X= a.drop(columns='grade')
y= a['grade']

X = s.fit_transform(X)


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

l = LogisticRegression(max_iter=10000)
m= l.fit(X_train,y_train)
n = m.predict(X_test)
print(n)

print("Accuracy Score Logistic Regression: ",accuracy_score(y_test,n))

# o = SVC()
# p = o.fit(X_train,y_train)
# q = p.predict(X_test)
# print("Accuracy Score : ",accuracy_score(y_test,q))

k = KNeighborsClassifier(n_neighbors=3)

r = k.fit(X_train,y_train)
s = r.predict(X_test)
print("Accuracy Score KNN : ",accuracy_score(y_test,s))

b = DecisionTreeClassifier()

c = b.fit(X_train,y_train)
d = c.predict(X_test)
print("Accuracy Score Decision Tree : ",accuracy_score(y_test,d))

e = RandomForestClassifier()

f = e.fit(X_train,y_train)
g = f.predict(X_test)
print("Accuracy Score Random Forest: ",accuracy_score(y_test,g))

h = GaussianNB()
# i = MultinomialNB()
j = BernoulliNB()

t = h.fit(X_train,y_train)
# u = i.fit(X_train,y_train)
v = j.fit(X_train,y_train)

w = t.predict(X_test)
# x = u.predict(X_test)
y = v.predict(X_test)

print("Accuracy Score GaussianNB : ",accuracy_score(y_test,w))
# print("Accuracy Score MultinomialNB: ",accuracy_score(y_test,x))
print("Accuracy Score BernoulliNB : ",accuracy_score(y_test,y))


