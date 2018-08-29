import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

gender_submission=pd.read_csv(r'C:\Users\yebod\Documents\GitHub\sys6018-competition-titanic\.git\all\gender_submission.csv')
test=pd.read_csv(r'C:\Users\yebod\Documents\GitHub\sys6018-competition-titanic\.git\all\test.csv')
train=pd.read_csv(r'C:\Users\yebod\Documents\GitHub\sys6018-competition-titanic\.git\all\train.csv')


gender_submission.tail()
test.head()
train.head()

#Drop useless columns
toDrop=['PassengerId','Name','Cabin','Ticket']
train_clean = train.drop(toDrop, axis=1)
train_clean['Sex']=train_clean['Sex']=='male'
train_dummy=pd.get_dummies(train_clean,columns=['Embarked','Pclass'])


X=train_dummy.iloc[:, 1:].values
y=train_dummy.iloc[:,0].values

X
#taking care of missing data
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,0:11])
X[:,0:11]=imputer.transform(X[:,0:11])

#train data using random forest
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,y)

#convert test file
test_clean = test.drop(toDrop, axis=1)
test_clean['Sex']=test_clean['Sex']=='male'
test_dummy=pd.get_dummies(test_clean,columns=['Embarked','Pclass'])
t=test_dummy.iloc[:,:].values

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(t[:,0:11])
t[:,0:11]=imputer.transform(t[:,0:11])
pred=clf.predict(t)
pid=list(range(892,1310))
result = pd.DataFrame(
    {'PassengerId': pid,
     'Survived': pred,
    })

result.to_csv('prediction.csv')





















