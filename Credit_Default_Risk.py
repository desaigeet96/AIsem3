# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:38:10 2020

@author: Geet Desai
"""


import pandas as pd
import numpy as np

application_train=pd.read_csv("E:/Study/DS(Sem 3)/AI/application_train.csv")

application_train['AMT_ANNUITY'].fillna(application_train['AMT_ANNUITY'].mean(),inplace=True)
application_train['AMT_GOODS_PRICE'].fillna(application_train['AMT_GOODS_PRICE'].mean(),inplace=True)
application_train['NAME_TYPE_SUITE'].fillna(application_train['NAME_TYPE_SUITE'].mode()[0],inplace=True)
application_train['OWN_CAR_AGE'].fillna(application_train['OWN_CAR_AGE'].mean(),inplace=True)
application_train['OCCUPATION_TYPE'].fillna(application_train['OCCUPATION_TYPE'].mode()[0],inplace=True)
application_train['EXT_SOURCE_1'].fillna(application_train['EXT_SOURCE_1'].mean(),inplace=True)
application_train['EXT_SOURCE_2'].fillna(application_train['EXT_SOURCE_2'].mean(),inplace=True)
application_train['EXT_SOURCE_3'].fillna(application_train['EXT_SOURCE_3'].mean(),inplace=True)
application_train['APARTMENTS_AVG'].fillna(application_train['APARTMENTS_AVG'].mean(),inplace=True)
application_train['BASEMENTAREA_AVG'].fillna(application_train['BASEMENTAREA_AVG'].mean(),inplace=True)
application_train['YEARS_BEGINEXPLUATATION_AVG'].fillna(application_train['YEARS_BEGINEXPLUATATION_AVG'].mean(),inplace=True)
application_train['YEARS_BUILD_AVG'].fillna(application_train['YEARS_BUILD_AVG'].mean(),inplace=True)
application_train['COMMONAREA_AVG'].fillna(application_train['COMMONAREA_AVG'].mean(),inplace=True)
application_train['ELEVATORS_AVG'].fillna(application_train['ELEVATORS_AVG'].mean(),inplace=True)
application_train['ENTRANCES_AVG'].fillna(application_train['ENTRANCES_AVG'].mean(),inplace=True)
application_train['FLOORSMAX_AVG'].fillna(application_train['FLOORSMAX_AVG'].mean(),inplace=True)
application_train['FLOORSMIN_AVG'].fillna(application_train['FLOORSMIN_AVG'].mean(),inplace=True)
application_train['LANDAREA_AVG'].fillna(application_train['LANDAREA_AVG'].mean(),inplace=True)
application_train['LIVINGAPARTMENTS_AVG'].fillna(application_train['LIVINGAPARTMENTS_AVG'].mean(),inplace=True)
application_train['LIVINGAREA_AVG'].fillna(application_train['LIVINGAREA_AVG'].mean(),inplace=True)
application_train['NONLIVINGAPARTMENTS_AVG'].fillna(application_train['NONLIVINGAPARTMENTS_AVG'].mean(),inplace=True)
application_train['NONLIVINGAREA_AVG'].fillna(application_train['NONLIVINGAREA_AVG'].mean(),inplace=True)

application_train.drop(['APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI'],axis=1,inplace=True)

nvals=pd.DataFrame(application_train.isnull().sum())

application_train['FONDKAPREMONT_MODE'].fillna(application_train['FONDKAPREMONT_MODE'].mode()[0],inplace=True)
application_train['HOUSETYPE_MODE'].fillna(application_train['HOUSETYPE_MODE'].mode()[0],inplace=True)
application_train['TOTALAREA_MODE'].fillna(application_train['TOTALAREA_MODE'].mode()[0],inplace=True)
application_train['WALLSMATERIAL_MODE'].fillna(application_train['WALLSMATERIAL_MODE'].mode()[0],inplace=True)
application_train['EMERGENCYSTATE_MODE'].fillna(application_train['EMERGENCYSTATE_MODE'].mode()[0],inplace=True)
application_train['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(application_train['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0],inplace=True)
application_train['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(application_train['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0],inplace=True)
application_train['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(application_train['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0],inplace=True)
application_train['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(application_train['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0],inplace=True)
application_train['DAYS_LAST_PHONE_CHANGE'].fillna(application_train['DAYS_LAST_PHONE_CHANGE'].mode()[0],inplace=True)
application_train['CNT_FAM_MEMBERS'].fillna(application_train['CNT_FAM_MEMBERS'].mode()[0],inplace=True)


application_train.drop(['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR'],axis=1,inplace=True)
application_train.drop(['WEEKDAY_APPR_PROCESS_START'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
application_train['NAME_CONTRACT_TYPE']=le.fit_transform(application_train['NAME_CONTRACT_TYPE'])
application_train['CODE_GENDER']=le.fit_transform(application_train['CODE_GENDER'])
application_train['FLAG_OWN_CAR']=le.fit_transform(application_train['FLAG_OWN_CAR'])
application_train['FLAG_OWN_REALTY']=le.fit_transform(application_train['FLAG_OWN_REALTY'])
application_train['NAME_TYPE_SUITE']=le.fit_transform(application_train['NAME_TYPE_SUITE'])
application_train['NAME_INCOME_TYPE']=le.fit_transform(application_train['NAME_INCOME_TYPE'])
application_train['NAME_EDUCATION_TYPE']=le.fit_transform(application_train['NAME_EDUCATION_TYPE'])
application_train['NAME_FAMILY_STATUS']=le.fit_transform(application_train['NAME_FAMILY_STATUS'])
application_train['NAME_HOUSING_TYPE']=le.fit_transform(application_train['NAME_HOUSING_TYPE'])
application_train['OCCUPATION_TYPE']=le.fit_transform(application_train['OCCUPATION_TYPE'])
application_train['ORGANIZATION_TYPE']=le.fit_transform(application_train['ORGANIZATION_TYPE'])
application_train['FONDKAPREMONT_MODE']=le.fit_transform(application_train['FONDKAPREMONT_MODE'])
application_train['HOUSETYPE_MODE']=le.fit_transform(application_train['HOUSETYPE_MODE'])
application_train['WALLSMATERIAL_MODE']=le.fit_transform(application_train['WALLSMATERIAL_MODE'])
application_train['EMERGENCYSTATE_MODE']=le.fit_transform(application_train['EMERGENCYSTATE_MODE'])

X=application_train.drop(['SK_ID_CURR','TARGET'],axis=1)
y=application_train['TARGET']

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=300)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test)
accuracy_score(y_pred,y_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
confusion_matrix(y_pred,y_test)
accuracy_score(y_pred,y_test)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
confusion_matrix(y_pred,y_test)
accuracy_score(y_pred,y_test)
