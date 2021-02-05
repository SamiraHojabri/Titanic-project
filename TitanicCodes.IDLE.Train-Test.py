import pandas as pd
import numpy as np
import seaborn as sns

df= pd.read_csv(r"D:\python.samira\Projects\Titanic\Titanic train.csv")
df

df= df.fillna(method = 'pad')
df = df.drop(columns = ['PassengerId','Name','Ticket','Cabin'])
df

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()#az in dastoor baraye tabdile dadehaye gheire adadi be adadi
#komak migirim:
df['Sex'] = le.fit_transform(df['Sex'])#dar in mesal dadehaye jensiat tabdil be 0,1 mishavand
df['Embarked'] = le.fit_transform(df['Embarked'])

targets = df['Survived']
df = df.drop(columns = ['Survived'])
df


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df,targets,test_size=0.3)#inja miaym dadehaye train va taest ro joda mikonim
#be nesbat 30 be 70 ham taghsim mikonim

from sklearn import svm#inja mikhaym yek bar az algorythm svm pishbini ra anjam bedim
svm_a = svm.SVC()
svm_a.fit(x_train,y_train)# اينجا به الگوريتم ميگيم بيا از داده هاي ترين ما ياد بگير


from sklearn import metrics#hala inja mikhaym deghat algorythm ro ke az dadehaye ma yad gerefte besanjim
predicted = svm_a.predict(x_test)#ba in dastoor behesh migim hala ke yad gerefti moghe predict kardane
#va faghat x_test ro behesh midim , bayad y_test ro be ma bede,nabayad y_test ro behesh bedim
print(metrics.classification_report(y_test,predicted))# inja darsad deghat algorythm va ye seri data dige mide be ma


from sklearn.tree import DecisionTreeClassifier#hala az ye algorythm dige ham estefade mikonim,ba in dastoor algorythm decisionTree ro farakhani kardim
#shayad in algorythm deghat behtari bede
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)#inja migim bia algorythm ro fit kon ba x va y train,yani bia ba in dadehaye train yad begir

predicted1 = dt.predict(x_test)
print(metrics.classification_report(y_test,predicted1))


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train,y_train)

predicted2 = RF.predict(x_test)
print(metrics.classification_report(y_test,predicted2))

