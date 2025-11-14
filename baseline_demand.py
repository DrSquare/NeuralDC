# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:44:26 2023

@author: h_min
"""

# Import required libraries 
import pandas as pd
import numpy as np 

import os

import statsmodels.api as sm
import statsmodels.formula.api as smf
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


##### 1. Load the sampled POS dataset
os.chdir('C:/Users/h_min/OneDrive/Desktop/CS221/Project')
df = pd.read_csv("salty_snack_temp_0.05.csv")


##### 2. Data exploration and processing
### Check data
# Columns
print(df.columns)

# Shape: # of rows and columns
print(df.shape)

# Info for each column 
print(df.info)

# Null values for each column
print(df.isnull().sum(axis = 0))

# Summary statistics for data and firt 5 rows 
pd.set_option("display.max_columns", None)
print(df.describe())

# Sample data: First 5 rows 
df.head(5)

### Filter out stores not in the training set 
# Training set definition
df_original = df.copy()
Filter = df['week'] >= 285
df_train = df[~Filter]

df_train['iri_key'].nunique()
slist = df_train['iri_key'].unique()
Filter2 = df['iri_key'].isin(slist) 

df=df[Filter2]
print(df['iri_key'].nunique())
print(df_original['iri_key'].nunique())

# Check unique values for categorical columns
categorical_col = ["producttype", "package", "flavorscent", "fatcontent", "cookingmethod", "saltsodiumcontent", "typeofcut"]
# f (feature) and brand are also categorical, but not handling here. 
for col in categorical_col:
    print(df[col].unique())

# NaN handling
df["fatcontent"].replace({np.nan: "REGULAR"}, inplace=True)
df["cookingmethod"].replace({np.nan: "MISSING"}, inplace=True)
df["saltsodiumcontent"].replace({np.nan: "MISSING"}, inplace=True)

# Check unique values after NaN handling
categorical_col = ["producttype", "package", "flavorscent", "fatcontent", "cookingmethod", "saltsodiumcontent", "typeofcut"]

for col in categorical_col:
    print(df[col].unique())

### Create dummy variables 
# Product attributes
brand = pd.get_dummies(df['brand'], prefix='BRAND', drop_first=True)
pack = pd.get_dummies(df['package'], prefix='PKG', drop_first=True)
flavor = pd.get_dummies(df['flavorscent'], prefix='FLV', drop_first=True)
fat = pd.get_dummies(df['fatcontent'], prefix='FAT', drop_first=True)
cook = pd.get_dummies(df['cookingmethod'], prefix='CM', drop_first=True)
salt = pd.get_dummies(df['saltsodiumcontent'], prefix='SALT', drop_first=True)
cut = pd.get_dummies(df['typeofcut'], prefix='CUT', drop_first=True)

# Promotion
feature = pd.get_dummies(df['f'], prefix='F', drop_first=True)
display = pd.get_dummies(df['d'], prefix='D', drop_first=True)
promo = pd.get_dummies(df['pr'], prefix='PR', drop_first=True)

# Store and week fixed effect
store = pd.get_dummies(df['iri_key'], prefix='STORE', drop_first=True)
week = pd.get_dummies(df['numweek'], prefix='WK', drop_first=True)

# Take out relevant continuous variables 
logprice = df['logprice']
price = df['price']

voleq = df['vol_eq']

# Create product key column
product = df['colupc']

# Create intercept
intercept = pd.DataFrame(np.ones(df.shape[0]))

# Dimension checks
print(df.shape)
print(df_original.shape)
print(len(voleq))
print(len(logprice))
print(store.shape)
print(week.shape)
print(feature.shape)
print(display.shape)
print(promo.shape)
print(brand.shape)
print(pack.shape)
print(flavor.shape)
print(fat.shape)
print(cook.shape)
print(salt.shape)
print(cut.shape)
print(intercept.shape)

print(product.shape)


##### 3. Train/Test Split
Mask = df['week'] >= 285
print(Mask.shape)
Y = df['logunits']
print(Y.shape)

#X = pd.concat([intercept, logprice, store, week, feature, display, promo, brand, voleq, pack, flavor, fat, cook, salt, cut], axis=1)
X = pd.concat([logprice, store, week, feature, display, promo, brand, voleq, pack, flavor, fat, cook, salt, cut], axis=1)#X1 = pd.concat([intercept, X], axis=1)  # Why does this expand the row number to original? 

X_test = X[Mask]

Y_test = Y[Mask]
print(X_test.shape)
print(Y_test.shape)
X_train = X[~Mask]
Y_train = Y[~Mask]

# Product key column
product_test = product[Mask]
product_train = product[~Mask]

print(X_train.shape)
print(Y_train.shape)
print(X.shape)

# Check dimensions
assert(X.shape[0] == X_train.shape[0] + X_test.shape[0])
assert(Y.shape[0] == Y_train.shape[0] + Y_test.shape[0])


# Check columns of X_train
print(X_train.columns)


### Create share variable incluidng outside shares: market potential = 3 x max(total unit sales in X_train)
# (1) Calculate total equivalent units for each store/week combination
df['eq_units'] = df['units'] * df['vol_eq']
df_train = df[~Mask]
df_test = df[Mask]

totalunits = df.groupby(['iri_key', 'week'])['eq_units'].sum().reset_index()
totalunits.rename(columns={'eq_units':'total_eq_units'}, inplace=True)
totalunits.head(500)

# (2) Calcualte market potnetial 
df_train = df_train.merge(totalunits, how='left', on = ['iri_key', 'week'])
df_train.columns
m_potential = df_train.groupby(['iri_key'])['total_eq_units'].max().reset_index() 
m_potential['m_potential'] = 3 * m_potential['total_eq_units']
m_potential.head(5)
m_potential.drop("total_eq_units", axis=1, inplace=True)
m_potential.head(5)

# (3) Merge market potential and calculate shares (including outside shares)
df_new = df.merge(m_potential, how='left', on = ['iri_key'])

df_new['share'] = df_new['eq_units'] / df_new['m_potential']

df_new['share'].describe()

# (4) Merge total eqivalent units and calculate within market shares
df_new = df_new.merge(totalunits, how='left', on = ['iri_key', 'week'])
df_new['share_within'] = df_new['eq_units'] / df_new['total_eq_units']
print(df_new.columns)
df_new.head(5)

# (5) Create outside share, log(share), log(outside share) variables
df_new['outside_share'] = 1 - df_new['total_eq_units']/df_new['m_potential']
df_new['logshare'] = np.log(df_new['share'])
df_new['logoutsideshare'] = np.log(df_new['outside_share'])

# (6) Create target variable for logit regression: log(share) - log(outside share)
df_new['sharedp'] = df_new['logshare'] - df_new['logoutsideshare']
df_new.head(10)
mp = df_new['m_potential']

# (7) Create new training dataset for logit models 
# X: Including store fixed effect for now -> No changes
# Y2: share (Y: log(units))
Y2 = df_new['sharedp']

Mask = df_new['week'] >= 285

Y2_test = Y2[Mask]
mp_test = mp[Mask]
df_new_test = df_new[Mask]

print(Y2_test.shape)

Y2_train = Y2[~Mask]
print(Y2_train.shape)

# Check dimensions
assert(Y2.shape[0] == Y2_train.shape[0] + Y2_test.shape[0])


### Define functions to calculate Weighted MAPE, MAPE and MPE
def wmape_score(actual, forecast, weight):
  wmape = ((np.abs(forecast - actual)/np.abs(actual))*weight/weight.sum()).sum()
  return wmape

def mape_score(actual, forecast):
  mask = actual > 0
  mape = np.abs((actual - forecast)/actual)[mask].mean()
  return mape

def mpe_score(actual, forecast): 
  mpe = np.mean((actual - forecast)/actual)
  return mpe


def counterfactual_validity(forecast1, forecast2): 
    per_not_valid = np.mean((forecast1 - forecast2) > 0)
    return per_not_valid


##### 3. Model training and evaluation 
### 1. Linear model - OLS
# (1) Instantiate the model 
lr =  linear_model.LinearRegression(fit_intercept=True)
# (2) Fit the model with Training Data
lr.fit(X_train, Y_train)


# (3) Make prediction with test set
Y_pred = lr.predict(X_test)

#X_test['newlogprice'] = np.log(np.exp(X_test['logprice']) * 0.9)
X_test2 = X_test.copy()
X_test2['logprice'] = np.log(np.exp(X_test2['logprice']) * 0.9)

# (4) Model evaluation: In actual space (converting log prediction to actual volume)
lr_rmse = mean_squared_error(np.exp(Y_test), np.exp(Y_pred), squared=False)
lr_r2 = r2_score(np.exp(Y_test), np.exp(Y_pred))
lr_mape = mape_score(np.exp(Y_test), np.exp(Y_pred))
lr_wmape = wmape_score(np.exp(Y_test), np.exp(Y_pred), np.multiply(np.exp(Y_test),df_test['price']))
lr_mpe = mpe_score(np.exp(Y_test), np.exp(Y_pred))

lr_cfv = counterfactual_validity(np.exp(lr.predict(X_test)), np.exp(lr.predict(X_test2)))

print(lr_rmse)
print(lr_r2)
print(lr_mape)
print(lr_wmape)
print(lr_mpe)
print(lr_cfv)


### 2. Homogeneous logit (including outside share)
# (1) Instantiate the model 
hl =  linear_model.LinearRegression(fit_intercept=True)
# (2) Fit the model with Training Data
hl.fit(X_train, Y2_train)

# (3) Make prediction with test set
Y2_pred = hl.predict(X_test)

ShareRatio_pred = pd.DataFrame(np.exp(Y2_pred), columns =['ShareRatio_pred'])
ShareRatio_pred_comb = pd.concat([ShareRatio_pred, df_new_test['iri_key'], df_new_test['week']], axis=1)
ShareSum = ShareRatio_pred_comb.groupby(['iri_key', 'week'])['ShareRatio_pred'].sum().reset_index()
ShareSum.rename(columns={'ShareRatio_pred':'ShareRatio_sum'}, inplace=True)

ShareRatio_pred_comb = ShareRatio_pred_comb.merge(ShareSum, how='left', on=['iri_key', 'week'])
ShareRatio_pred_comb['Osh'] = 1 / (ShareRatio_pred_comb['ShareRatio_sum'] + 1)

Y_pred = ShareRatio_pred_comb['ShareRatio_pred'] * ShareRatio_pred_comb['Osh'] * mp_test 

# (4) Model evaluation
hl_rmse = mean_squared_error(np.exp(Y_test), Y_pred, squared=False)
hl_r2 = r2_score(np.exp(Y_test), Y_pred)
hl_mape = mape_score(np.exp(Y_test), Y_pred)
hl_wmape = wmape_score(np.exp(Y_test), Y_pred, np.multiply(np.exp(Y_test),df_test['price']))
hl_mpe = mpe_score(np.exp(Y_test), Y_pred)

hl_cfv = counterfactual_validity(np.exp(hl.predict(X_test)), np.exp(hl.predict(X_test2)))

print(hl_rmse)
print(hl_r2)
print(hl_mape)
print(hl_wmape)
print(hl_mpe)
print(hl_cfv)


### 3. XGB
# (1) Instantiate the model 
xgb = XGBRegressor()

# (2) Fit the model with Training Data
# param_grid = [{'learning_rate' : [0.01, 0.1], 'max_depth' : [3, 6, 9]}]
# gs = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# gs.fit(X_train, Y_train)
xgb.fit(X_train, Y_train)

# (3) Make prediction with test set
Y_pred_xgb = xgb.predict(X_test)
# Y_pred_xgb = gs.predict(X_test)


# (4) Model evaluation
xgb_rmse = mean_squared_error(np.exp(Y_test), np.exp(Y_pred_xgb), squared=False)
xgb_r2 = r2_score(np.exp(Y_test), np.exp(Y_pred_xgb))
xgb_mape = mape_score(np.exp(Y_test), np.exp(Y_pred_xgb))
xgb_wmape = wmape_score(np.exp(Y_test), np.exp(Y_pred_xgb), np.multiply(np.exp(Y_test),df_test['price']))
xgb_mpe = mpe_score(np.exp(Y_test), np.exp(Y_pred_xgb))

xgb_cfv = counterfactual_validity(np.exp(xgb.predict(X_test)), np.exp(xgb.predict(X_test2)))

print(xgb_rmse)
print(xgb_r2)
print(xgb_mape)
print(xgb_wmape)
print(xgb_mpe)
print(xgb_cfv)

### 4. Random Forests
# (1) Instantiate the model 
rf = RandomForestRegressor()

# (2) Fit the model with Training Data
rf.fit(X_train, Y_train)

# (3) Make prediction with test set
Y_pred_rf = rf.predict(X_test)

# (4) Model evaluation
rf_rmse = mean_squared_error(np.exp(Y_test), np.exp(Y_pred_rf), squared=False)
rf_r2 = r2_score(np.exp(Y_test), np.exp(Y_pred_rf))
rf_mape = mape_score(np.exp(Y_test), np.exp(Y_pred_rf))
rf_wmape = wmape_score(np.exp(Y_test), np.exp(Y_pred_rf), np.multiply(np.exp(Y_test),df_test['price']))
rf_mpe = mpe_score(np.exp(Y_test), np.exp(Y_pred_rf))

rf_cfv = counterfactual_validity(np.exp(rf.predict(X_test)), np.exp(rf.predict(X_test2)))

print(rf_rmse)
print(rf_r2)
print(rf_mape)
print(rf_wmape)
print(rf_mpe)
print(rf_cfv)