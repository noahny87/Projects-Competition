import pandas as pd 
import numpy as np 
import sklearn as sk 
import datetime as dt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib as plt
import lightgbm as lgb
import xgboost as xgb
import csv
import matplotlib.pyplot as plt
 

def format_data(train,test):
   #preprocess data 
    missing_cols_in_test = [
    'shutdown',
    'mini_shutdown',
    'blackout',
    'mov_change',
    'frankfurt_shutdown',
    'precipitation',
    'snow',
    'user_activity_1',
    'user_activity_2'
      ]

    train = train.drop(missing_cols_in_test, axis=1, errors='ignore')
    train = train.sort_values("date").reset_index(drop=True)
    train.shape, test.shape

    for df in [train, test]:
     df['holiday_name'] = df['holiday_name'].fillna('').astype('category')
     df['warehouse'] = df['warehouse'].astype('category')
    
     df['date'] = pd.to_datetime(df['date'])
     df['year'] = df['date'].dt.year
     df['month'] = df['date'].dt.month
     df['day'] = df['date'].dt.day
     df['day_of_week'] = df['date'].dt.dayofweek
     df['day_of_year'] = df['date'].dt.dayofyear
     df['week'] = df['date'].dt.isocalendar().week.astype(int)
     df['quarter'] = df['date'].dt.quarter
     df['season'] = (df['month'] % 12 + 3) // 3
     df['season'] = df['season'].astype(int)
     df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    
     df['group'] = (df['year']-2020)*48+df['month']*4+df['day']//7
    
     df['day_before_holiday'] = df['holiday'].shift().fillna(0)
     df['day_after_holiday'] = df['holiday'].shift(-1).fillna(0)
     df['day_before_school_holiday'] = df['school_holidays'].shift().fillna(0)
     df['day_after_school_holiday'] = df['school_holidays'].shift(-1).fillna(0)
     df['day_before_winter_school_holiday'] = df['winter_school_holidays'].shift().fillna(0)
     df['day_after_winter_school_holiday'] = df['winter_school_holidays'].shift(-1).fillna(0)
     df['holiday_and_shops_closed'] = df['holiday'] * df['shops_closed']
     df['week_before_holiday'] = df['holiday'].shift(-7).rolling(window=7, min_periods=1).sum().shift(1).fillna(0).apply(lambda x: 1 if x > 0 else 0)


    cat_features = [col for col in train.columns if train[col].dtype.name == 'category']

    train.shape, test.shape
    return train,test


def df_processing(train,test): 
   #to determine what columns to keep we can perform a correlation test on teh variables against each other mainly looking at our target variable orders 
   #create corr heat map 
   train1 = train.drop(['date','id','warehouse','holiday_name'],axis = 1)
   train1 = pd.DataFrame(train1)
   #in order for test data to predict properly with our model they need to have same predictor columns and values so we need to remove all columns that are not the same in both
   #this is because a model can only predict based on features found in the test set 
   train_pred = train1['orders']
   #format test data i.e. only keep encoded columns 
   test_var = test.drop(['warehouse', 'date', 'holiday_name'], axis =1)
   test_var = pd.DataFrame(test_var)
   
   return train1,train_pred,test_var


def split(X, y):
   #split training data 
   X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.9)
   return  X_train,X_test,y_train,y_test

def run_model1(Xtr,Xte,ytr,yte,pred):
   #heres where the LightGBM model will run 
   # defining parameters 
   
   lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.9547441579422116,
    "learning_rate": 0.04406871820205758,
    "min_child_samples": 8,
    "min_child_weight": 0.5163384032966228,
    "min_split_gain": 0.18807271127987724,
    "n_estimators": 600,
    "n_jobs": -1,
    "num_leaves": 48,
    "random_state": 123,
    "reg_alpha": 5.4660768249665646,
    "reg_lambda": 5.974261145462608,
    "subsample": 0.039552743069246055,
    "verbose": -1
}
   
   train = lgb.Dataset(data=Xtr,label=ytr)
   test = lgb.Dataset(data=Xte,reference=train)
   
   params_lgbm = {
    'metric': ['l2'],
    'learning_rate': [.05,.1, .15,.2],
    'max_depth': [3,5,7],
    'num_leaves': [31,35,40],
    'n_estimators': [300,500,700],
    'verbose':[-1],
   }
   
   # Initialize your model
   model1 = lgb.LGBMRegressor(**lgbm_params)
   model1 = model1.fit(Xtr,ytr)
   # Set up GridSearchCV
   model2 = lgb.LGBMRegressor(boosting_type='gbdt',reg_alpha=5.53451745697,min_child_samples=8,min_split_gain= 0.200727, reg_lambda=5.95423562434,n_jobs=-1,colsample_bytree=.96,subsample=.0392663746,min_child_weight  = 0.5263384032966228, random_state=123)
   grid_search = GridSearchCV(model2, params_lgbm, scoring='neg_mean_absolute_percentage_error', cv=5)

# Fit GridSearchCV to the training data
   g = grid_search.fit(Xtr, ytr)

   # Use the best estimator to make predictions on the test data
   bst = grid_search.best_estimator_.predict(Xte)

# Calculate the MAPE using the true values and predictions
   bstmape = mean_absolute_percentage_error(yte, bst)
   #predict future vals using grid search
   Tru_pred2 = grid_search.best_estimator_.predict(pred)
   bst_param = grid_search.best_params_
   # Continue with your code for prediction and evaluation
   model_pred = model1.predict(Xte)
   mape = mean_absolute_percentage_error(yte, model_pred)
   Tru_pred = model1.predict(pred)

   # Print outcome
   print(f"The best parameters for each warehouse:{bst_param}","\n")
   print(f'The MAPE of the grid search is:{bstmape*100}',"\n")
   print(f"The MAPE of the (non-grid search) lgbm1 is: {mape*100}%")
   #train and test sets are already filtered by warehouse so need an if statement comparing mape's of each model 
   #then return the predictions of the best model
   
   if bstmape < mape :
      return Tru_pred2
   else:
      return Tru_pred



 #create csv and append the data to it 
def create_csv(pred):
   pred.to_csv("Submission.csv",index=False)
   print('file created successfully!')
      
from sklearn.model_selection import train_test_split

def by_warehouse(train, test):
    # Preprocess the data
    train1 = train.drop(['date', 'id', 'holiday_name'], axis=1)
    train_pred1 = train1['orders']
    test_var = test.drop(['date', 'holiday_name'], axis=1)
    
    # Initialize final DataFrame
    final_df = pd.DataFrame(columns=['id', 'Predicted Orders'])
    
    # Loop through each warehouse
    for warehouse in test_var['warehouse'].unique():
        print(f"Warehouse: {warehouse}")
        
        # Filter data for the current warehouse
        train_ware = train1[train1['warehouse'] == warehouse]
        train_pred = train_ware['orders']
        test_ware = test_var[test_var['warehouse'] == warehouse]
        
        # Prepare data for model training
        train_ware = train_ware.drop(['warehouse', 'orders'], axis=1)
        test_ware2 = test_ware.drop(['warehouse', 'id'], axis=1)
        
        # Split the data
        Xtr, xte, ytr, yte = train_test_split(train_ware, train_pred, train_size=0.8,random_state=1024)
        
        # Train the model and get predictions (replace 'run_model1' with your actual model training and prediction code)
        m1_predictions = run_model1(Xtr, xte, ytr, yte, test_ware2)
        # Create a temporary DataFrame with the current warehouse's predictions
        temp_df = pd.DataFrame({
            'id': test_ware['id'].values,
            'Predicted Orders': m1_predictions
        })
        
        # Append the temporary DataFrame to the final DataFrame
        final_df = pd.concat([final_df, temp_df], ignore_index=True)
    
    return final_df

# Make sure to define the 'run_model1' function to return predictions


      

def main(): 
  #load data 
  train = pd.read_csv("C:\\Users\\noahn\\OneDrive\\Documents\\Pythonprojects\\rohlik-orders-forecasting-challenge\\train.csv")
  test = pd.read_csv("C:\\Users\\noahn\\OneDrive\\Documents\\Pythonprojects\\rohlik-orders-forecasting-challenge\\test.csv")
  #create instances storing the data formatted data
  train_f,test_f = format_data(train,test)
  #initialize data for splitting 
  train_var,train_pred,test_var = df_processing(train_f,test_f)
  #split data 
  X_train,X_test,y_train,y_test = split(train_var,train_pred)
  #remove ID column from test data and add back after 
  test_var = pd.DataFrame(test_var)
  test2 = test_var.drop('id',axis =1)
  print(train_f.head())
  #run lightGBM model on data
  #results = (run_model1(X_train,X_test,y_train,y_test,test2))
  #results2 = (run_model2(X_train,X_test,y_train,y_test,test2))
  #append results back to test var data since no random state was applied they can be appended back as they were taken out
  #res = pd.DataFrame()
  #call new functions 
  b = by_warehouse(train_f,test_f)
  print(b)
  #create the csv 
  create_csv(b)

  #plot predicted orders 
  #x=b['Predicted Orders'].plot()
  #plt.show()
  
if __name__ == "__main__":
   main()