import pandas as pd 
import numpy as np 
import sklearn 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 

def backTestPrices():

    df=pd.read_csv("sp500.csv")

    X= df[['Open','High','Low','Volume']]
    y= df['Close']


    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    regr=RandomForestRegressor(max_depth=5,n_estimators=100)
    regr.fit(X_train,y_train)
    pred=regr.predict(X_test)

    return regr.score(X_test,y_test)

# pred_price_score=backTestPrices()
# print(pred_price_score)
df=pd.read_csv("sp500.csv")

#

X= df[['Open','High','Low','Volume']]
y= df['Close']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

regr=RandomForestRegressor(max_depth=5,n_estimators=100)

regr.fit(X_train,y_train)
pred=regr.predict(X_test)
print(pred)