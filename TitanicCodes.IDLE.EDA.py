import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')#in code miad nemoodarha ro be soorate khat keshi shode mide

df= pd.read_csv(r"D:\python.samira\Projects\Titanic\Titanic train.csv")
df

df.describe()
df.corr()#اين کد وابستگي فيچرها با هم رو نشون ميده در يک جدول

df.isna().sum()

sns.catplot(x= 'Sex', hue ='Survived', data= df,kind= 'count')

df.groupby(['Sex'])['Survived'].aggregate(lambda x:x.sum()/len(x))

group = df.groupby(['Survived','Pclass'])
#ba dastoore "groupby" migim sotoone Survived ro joda kon va bar asase sotoone Pclass biar too ye jadval 
data=group.size().unstack()
sns.heatmap(data,annot=True,fmt='d')


s1 = df[df['Sex'] == 'female']
s1
s1['Age'].plot(kind='hist')
s1['Age'].mean()

s2 = df[df['Sex'] == 'male']
s2
s2['Age'].plot(kind='hist')
s2['Age'].mean()

df[df['Fare']<1]
len(df[df['Fare']<1])

sns.catplot(x='Survived',hue='Embarked',data=df,kind='count')
sns.catplot(x='Survived',col='Embarked',data=df,kind='count')

#حالا براي چند فيچر ترکيبي کار ميکنيم
sns.catplot(x='Survived',hue='Sex',col='Pclass',data=df,kind='count')

sns.boxplot(x='Pclass',y='Age',data=df)
