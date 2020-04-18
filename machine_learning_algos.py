import pandas as pd 
import numpy as np 
import sklearn 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 


# df=pd.read_csv("path/to/file")

"""
def DecisionTree():

    df=age_groups()

    X=df[['User_ID', 'Product_ID', 'Gender', 'Occupation', 'City_Category',
        'Marital_Status', 'Product_Category_1', 'Product_Category_2',
        'Product_Category_3']]

    y=df['Purchase']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    regr=RandomForestRegressor(max_depth=5,n_estimators=500)
    # for model importance 
    regr.feature_importances_

    regr.fit(X_train,y_train)
    predictions=regr.predict(X_test)

    return regr.score(X_test,y_test)
"""
# for back testing can add features seamlessly 

def backTestPrices():

    df=pd.read_csv("C:/Users/rpalacio/OneDrive - Capgemini/Desktop/crypto_price.csv")

    X=df['Time']

    y=str(df['Price'])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    regr=RandomForestRegressor(max_depth=5,n_estimators=100)
    regr.fit(X_train,y_train)
    pred=regr.predict(X_test)

    return regr.score(X_test,y_test)

# pred_price_score=backTestPrices()
# print(pred_price_score)
df=pd.read_csv("C:/Users/rpalacio/OneDrive - Capgemini/Desktop/crypto_price.csv")

X=df['Time'].astype('float')

y=df['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

regr=RandomForestRegressor(max_depth=5,n_estimators=100)

regr.fit(X_train,y_train)
pred=regr.predict(X_test)
print(pred)