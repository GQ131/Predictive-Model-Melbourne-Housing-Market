# I import and read the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy import stats
from scipy.stats import norm, skew
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

#upload file onto data framw
Housing_df = pd.read_csv("Melbourne_housing.csv")
pd.set_option('display.max_columns', 25)
print(Housing_df.shape)

# I check the amount of missing values in the data set
def display_missing_perc(Housing_df):
    """
    This is a function that evaluates the percentage of NA values per column
    """
    for col in Housing_df.columns.tolist():          
        missing_value = 100*(Housing_df[col].isnull().sum()/len(Housing_df[col]))
        missing_num = Housing_df[col].isnull().sum()
        print(f'{col} column missing values: {missing_value} ; total missing: {missing_num}') # Here, I can also see the total number of missing values.
    print('\n')
display_missing_perc(Housing_df)


#I remove any unnamed columns
Housing_df = Housing_df.loc[:, ~Housing_df.columns.str.contains('^Unnamed')]
Housing_df.dtypes

#This is function to visualize the NA values.
#it is a heatmap to see how many NA values per column.
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"
dic_cols = {col:utils_recognize_type(Housing_df, col, max_cat=20) for col in Housing_df.columns}
heatmap = Housing_df.isnull()
for k,v in dic_cols.items():
    if v == "num":
        heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    else:
        heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
plt.show()
print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")


Housing_df.drop(['Address','Bedroom2','BuildingArea','YearBuilt', 'Lattitude', 'Longtitude'], axis=1, inplace=True)

# I display all the object columns:
Housing_df.select_dtypes(include = 'object')

# Display all the numeric columns (both float and int64):
Housing_df.select_dtypes(include=['float', 'int64'])


#With the 'Car'column, it's safe to assume that NaN values are 0, indicating that the property does not have a car spot.
Housing_df['Car'] = Housing_df['Car'].fillna(0)

Housing_df['Landsize'].value_counts()#this list all the unique values in our data set
#Housing_df['Landsize'].value_counts().idxmax() #this lists the most frequent values - the index of maximum values

# Column Landsize presents several missing values and 38 values as 0. I create a plot. 
# Frequency counts of the 'diagnosis' column
#Landsize_counts = Housing_df['Landsize'].value_counts()
# Create a bar plot
plt.figure(figsize=(10, 6))
sns.histplot(Housing_df['Landsize'], kde=True)
plt.title('Distribution of Landsize')
plt.xlabel('Landsize')
plt.ylabel('Frequency')
plt.show()

num_nans = Housing_df['Landsize'].isna().sum()
print(f"Number of NaNs in the 'Landsize' column: {num_nans}")


Housing_df.drop(['Landsize'], axis=1, inplace=True)
Housing_df.dtypes

Housing_df['Price'].value_counts()#this list all the unique values in our data set

#I check the number of missing values for the dependent variable
Housing_df['Price'].isnull().sum()


# Calculate the median price per suburb
median_price_Suburb = Housing_df.groupby('Suburb')['Price'].transform('median')

# Fill in missing values in the 'Price' column with the corresponding median price per suburb
Housing_df['Price'] = Housing_df['Price'].fillna(median_price_Suburb)


# Check the number of missing values in the 'Price' column after imputation
Housing_df['Price'].isnull().sum()

#Since there are some missing values, I use a broader category (e.g., 'Regionname') for imputation
median_price_Region = Housing_df.groupby('Regionname')['Price'].transform('median')
Housing_df['Price'] = Housing_df['Price'].fillna(median_price_Region)
Housing_df['Price'].isnull().sum()

median_bathroom_Region = Housing_df.groupby('Regionname')['Bathroom'].transform('median')
Housing_df['Bathroom'] = Housing_df['Bathroom'].fillna(median_price_Region)
Housing_df['Bathroom'].isnull().sum()

numeric_cols = Housing_df.select_dtypes(include=[np.number]).columns.tolist()
sns.pairplot(Housing_df, x_vars=[col for col in numeric_cols if col!='Price'], 
             y_vars='Price', kind='scatter')
plt.show()

# I check the distribution of the dependent variable
sns.distplot(Housing_df['Price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(Housing_df['Price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(Housing_df['Price'], plot=plt)
plt.show()

Housing_df["Price"] = np.log1p(Housing_df["Price"])

#Check the new distribution 
sns.distplot(Housing_df['Price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(Housing_df['Price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(Housing_df['Price'], plot=plt)
plt.show()

# I check the amount of missing values in the data set once again
def display_missing_perc(Housing_df):
    """
    This is a function that evaluates the percentage of NA values per column
    """
    for col in Housing_df.columns.tolist():          
        missing_value = 100*(Housing_df[col].isnull().sum()/len(Housing_df[col]))
        missing_num = Housing_df[col].isnull().sum()
        print(f'{col} column missing values: {missing_value} ; total missing: {missing_num}') # Here, I can also see the total number of missing values.
    print('\n')
display_missing_perc(Housing_df)

# I make sure the dates are in string format
Housing_df['Date'] = Housing_df['Date'].astype(str)

# Here, we are selecting which columns we cant to encode as "1, 2, 3.. etc"
from sklearn.preprocessing import LabelEncoder
cols = ['Suburb', 'Type', 'Method', 
        'SellerG', 'Date','CouncilArea', 'Regionname']
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(Housing_df[c].values)) #important: with this it will learn the number inside the column. 
    Housing_df[c] = lbl.transform(list(Housing_df[c].values))

# shape        
print('Shape all_data: {}'.format(Housing_df.shape))


print(Housing_df.head())

# this is the more "machine learning" way
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Housing_df = Housing_df.rename(columns={'Price':'Y'})
X=Housing_df.drop('Y', axis=1)
y=Housing_df['Y']
x=sm.add_constant(X) #under this way, I need to add a constant to the X value. 
results=sm.OLS(y,x).fit()
results.summary()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Splitting the dataset into training and testing sets
train, test = train_test_split(Housing_df, train_size=0.8, random_state=42)

# Define dependent and independent variables
X_train = train.drop('Y', axis=1)
y_train = train['Y']
X_test = test.drop('Y', axis=1)
y_test = test['Y']

#Scaling the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# Making predictions
preds = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Here, we undo the transformation we did to our dependent variable.
#y_pred_inverse = np.expm1(preds)  #this is the antilog of the vairable -1
#y_true_inverse = np.expm1(y_test)  # y_test are the true values

# Calculate MSE and R² on the original scale
#mse = mean_squared_error(y_true_inverse, y_pred_inverse)
#r2 = r2_score(y_true_inverse, y_pred_inverse)

#print("MSE:", mse)
#print("R^2:", r2)


from sklearn.linear_model import Lasso

#initialize the Lasso model
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

# Predict on training and testing data
train_preds_lasso = lasso.predict(X_train_scaled)
test_preds_lasso = lasso.predict(X_test_scaled)

# Evaluation
train_mse_lasso = mean_squared_error(y_train, train_preds_lasso)
test_mse_lasso = mean_squared_error(y_test, test_preds_lasso)
train_r2_lasso = r2_score(y_train, train_preds_lasso)
test_r2_lasso = r2_score(y_test, test_preds_lasso)

print(f"Training MSE with Lasso: {train_mse_lasso}")
print(f"Test MSE with Lasso: {test_mse_lasso}")
print(f"Training R-squared with Lasso: {train_r2_lasso}")
print(f"Test R-squared with Lasso: {test_r2_lasso}")

# Analyze the coefficients
print("Lasso Coefficients:", lasso.coef_)

# I reverse the logarithmic transformation for true values and predictions
y_train_inverse = np.expm1(y_train)
y_test_inverse = np.expm1(y_test)
train_preds_lasso_inverse = np.expm1(train_preds_lasso)
test_preds_lasso_inverse = np.expm1(test_preds_lasso)

# I evaluate my model on the original scale
train_mse_lasso_inverse = mean_squared_error(y_train_inverse, train_preds_lasso_inverse)
test_mse_lasso_inverse = mean_squared_error(y_test_inverse, test_preds_lasso_inverse)
train_r2_lasso_inverse = r2_score(y_train_inverse, train_preds_lasso_inverse)
test_r2_lasso_inverse = r2_score(y_test_inverse, test_preds_lasso_inverse)

print(f"Training MSE with Lasso (Original Scale): {train_mse_lasso_inverse}")
print(f"Test MSE with Lasso (Original Scale): {test_mse_lasso_inverse}")
print(f"Training R-squared with Lasso (Original Scale): {train_r2_lasso_inverse}")
print(f"Test R-squared with Lasso (Original Scale): {test_r2_lasso_inverse}")


train, val = train_test_split(Housing_df, train_size=0.8, random_state=4761)

# Define dependent and independent variables
X_train = train.drop('Y', axis=1)
y_train = train['Y']
X_test = test.drop('Y', axis=1)
y_test = test['Y']

#Scaling the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train_scaled, y_train)


import statsmodels.api as sm

linear_aic = results.aic
linear_bic = results.bic

print(f"Linear Regression AIC: {linear_aic}")
print(f"Linear Regression BIC: {linear_bic}")


# Making predictions
preds = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)  # Calculate RMSE by taking the square root of MSE
r2 = r2_score(y_test, preds)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")  # Print RMSE
print(f"R-squared: {r2}")


from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=None, cv=5, random_state=0) # when I say 'alphas=none' I let sklearn specify the amount of alphas. 

# Fit the model
lasso_cv.fit(X_train_scaled, y_train)

# Best alpha
print(f"Best alpha: {lasso_cv.alpha_}")

from sklearn.linear_model import LassoLarsIC
import time

# Model based on BIC
model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X_train_scaled, y_train)
alpha_bic_ = model_bic.alpha_

# Model based on AIC
model_aic = LassoLarsIC(criterion='aic')
t1 = time.time()
model_aic.fit(X_train_scaled, y_train)
alpha_aic_ = model_aic.alpha_

print(f"Alpha for BIC: {alpha_bic_}")
print(f"Alpha for AIC: {alpha_aic_}")


# AIC from LassoLarsIC
aic = model_aic.criterion_

# Number of parameters (non-zero coefficients in Lasso)
k = np.sum(lasso_cv.coef_ != 0)

# Sample size
n = X_train_scaled.shape[0]

# Calculate AICc
aicc = aic + (2*k*(k+1)/(n-k-1))

print(f"AIC: {aic}")
print(f"AICc: {aicc}")


# Predict on training and testing data
train_preds_lasso_CV = lasso_cv.predict(X_train_scaled)
test_preds_lasso_CV = lasso_cv.predict(X_test_scaled)

from sklearn.metrics import mean_squared_error, r2_score

# Calculate metrics
mse_train = mean_squared_error(y_train, train_preds_lasso_CV)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, train_preds_lasso_CV)

mse_test = mean_squared_error(y_test, test_preds_lasso_CV)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, test_preds_lasso_CV)

print(f"Training RMSE: {rmse_train}, R²: {r2_train}")
print(f"Test RMSE: {rmse_test}, R²: {r2_test}")

# Analyze the coefficients
print("Lasso Coefficients:", lasso.coef_)

# I reverse the logarithmic transformation for true values and predictions
y_train_inverse = np.expm1(y_train)
y_test_inverse = np.expm1(y_test)
train_preds_lassoCV_inverse = np.expm1(train_preds_lasso_CV)
test_preds_lassoCV_inverse = np.expm1(test_preds_lasso_CV)

# I evaluate my model on the original scale
train_mse_lassoCV_inverse = mean_squared_error(y_train_inverse, train_preds_lassoCV_inverse)
test_mse_lassoCV_inverse = mean_squared_error(y_test_inverse, test_preds_lassoCV_inverse)
train_r2_lassoCV_inverse = r2_score(y_train_inverse, train_preds_lassoCV_inverse)
test_r2_lassoCV_inverse = r2_score(y_test_inverse, test_preds_lassoCV_inverse)

print(f"Training MSE with Lasso (Original Scale): {train_mse_lassoCV_inverse}")
print(f"Test MSE with Lasso (Original Scale): {test_mse_lassoCV_inverse}")
print(f"Training R-squared with Lasso (Original Scale): {train_r2_lassoCV_inverse}")
print(f"Test R-squared with Lasso (Original Scale): {test_r2_lassoCV_inverse}")


from sklearn.model_selection import cross_val_score

# For Linear Regression
linear_cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

# For Lasso Regression
lasso_cv_scores = cross_val_score(lasso_cv, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"Linear Regression 5-fold CV: {-linear_cv_scores.mean()}")
print(f"Lasso Regression 5-fold CV: {-lasso_cv_scores.mean()}")


# Calculate MSE for the predictions on test data for both models
linear_test_mse = mean_squared_error(y_test, preds)  # preds from Linear Regression
lasso_test_mse = mean_squared_error(y_test, test_preds_lasso)  # test_preds_lasso from Lasso

print(f"Linear Regression Test MSE: {linear_test_mse}")
print(f"Lasso Regression Test MSE: {lasso_test_mse}")


