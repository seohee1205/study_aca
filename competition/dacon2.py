import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold,HalvingRandomSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel as C
import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
path = 'd:/study_data/_data/dacon/'
path_save = 'd:/study_data/_save/dacon/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# Encode categorical features
encoder = LabelEncoder()

train_csv['Weight_Status'] = encoder.fit_transform(train_csv['Weight_Status'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Weight_Status'] = encoder.fit_transform(test_csv['Weight_Status'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [GaussianProcessRegressor(), XGBRegressor()]    #LGBMRegressor()

param_r = [{"kernel":[ C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
                          C(1.0, (1e-3, 1e3)) * RBF(20, (1e-2, 1e2)) + WhiteKernel(noise_level=1.2, noise_level_bounds=(1e-10, 1e+1)),
                          C(1.5, (1e-3, 1e3)) * RBF(15, (1e-2, 1e2)) + WhiteKernel(noise_level=1.5, noise_level_bounds=(1e-10, 1e+1)),],
               "n_restarts_optimizer": [5, 9, 12],"alpha": [1e-10, 1e-5],}]

param_d = [{'iterations':[1000,2000,1500],'learning_rate':[0.03,0.05,0.01]},{'depth':[6,5,2],'loss_function':['RMSE'],'task_type':['CPU']}]
# Split data into training and testing sets
x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 1412)

for k in range(10000):
    x_train, x_test,y_train, y_test = train_test_split(x,
                            y,test_size=0.8, shuffle=True, random_state=k)
    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test_csv = scaler.transform(test_csv)
        
        for j in range(len(model_list)):
            if j==0:
                param = param_r
            elif j==1:
                param = param_d
            model = RandomizedSearchCV(model_list[j], param, cv=10, verbose=1)
            model.fit(x_train, y_train)

            loss = model.score(x_test, y_test)
            print('loss : ', loss)
            print('test RMSE : ', RMSE(y_test, model.predict(x_test)))
            
            if RMSE(y_test, model.predict(x_test))<0.5:
                submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
                submit_csv['Calories_Burned'] = model.predict(test_csv)
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(path_save + 'Calories' + date + '.csv')
                break