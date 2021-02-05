import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

df = pd.read_csv(r"D:\python.samira\titanic train.csv")

df.head (10)#in dastoor 10 radif aval ro az jadval mide
print(df.head(10))

df.tail(6)#in dastoor 10 radif aval ro az jadval mide
print(df.tail(6))

df.describe()# yek did koli tahlili az jadvale morede nazar mide
print (df.describe())

df['Age'].median()# miangine seni ro az dadehaye mokhtalefe jadval mide, yek adad ast
print(df['Age'].median())

df['Survived'].unique()
print(df['Survived'].unique())#baraye in fichere khas yek araye:[0,1] midahad,dadehaye sotoone morede nazar
#bedoone tekrar ra barmigardanad

df['Pclass'].unique()
print(df['Pclass'].unique())

df['Fare'].unique()
print(df['Fare'].unique())

df['Sex'].unique()
print(df['Sex'].unique())

df['Age'].unique()
print(df['Age'].unique())

df['Cabin'].unique()
print(df['Cabin'].unique())

df['Embarked'].unique()
print(df['Embarked'].unique())















