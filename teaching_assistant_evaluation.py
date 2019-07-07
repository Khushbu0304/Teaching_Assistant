
#importing libraries for the project teaching assistant dataset evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# names is the list consisting of all the column names or the features of the project

names=['english_speaker','course_instructor','course','semester','class_size','score']

# reading csv file "tae_data.csv"
#url='https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data'
tae_dataset=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data',names=names)

# printing the dataset of teaching assistant evaluation

print(tae_dataset)

# counting the total number of data available

print(tae_dataset.count()) # no missing enteries
print(tae_dataset.head())  # printing first five data entries
print(tae_dataset.tail())  # printing last five data entries

# objective 1
# feature building

from sklearn.preprocessing import StandardScaler

convert  = StandardScaler()
feature  = tae_dataset.drop(['course_instructor','course','score'],axis=1)
label    = tae_dataset['score']
feature  = convert.fit_transform(feature)


# objective 2
#train_test_split (80 & 20 percent) 

from sklearn.model_selection import train_test_split
f_train,f_test,l_train,l_test=train_test_split(feature,label,random_state=0,test_size=0.2)
print(f_train.shape)
print(f_test.shape)


#objective 3
#evaluate the accuracy usig logistic function

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0,multi_class='ovr')
model.fit(f_train,l_train)
y_predict=model.predict(f_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(l_test,y_predict))
confusion_matrix=confusion_matrix(l_test,y_predict)
print(confusion_matrix)


#objective 4
#find the best value of c

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix
score=[]
for i in range(-100,100):
    if(i<=0):
        print('C=%f'%(i)," is not valid for negative values of C")
    else:
        model=LogisticRegression(C=i,random_state=0,multi_class='ovr')
        model.fit(f_train,l_train)
        y_predict=model.predict(f_test)
        s=accuracy_score(l_test,y_predict)
        score.append(s)
        print('Accuracy at C=%f:%f'%(i,s))  
    
m=score.index(max(score))
print('Max Accuracy=',max(score)*100)
print('Max Accuracy at C=',m+1)

