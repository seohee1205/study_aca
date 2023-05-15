import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier
import time
# Load data
train = pd.read_csv('d:/study_data/_data/dacon_airplane/train.csv')
test = pd.read_csv('d:/study_data/_data/dacon_airplane/test.csv')
sample_submission = pd.read_csv('d:/study_data/_data/dacon_airplane/sample_submission.csv', index_col=0)

#print(train)
# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
NaN = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']

for col in NaN:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)

    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])

    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
print('Done.')

# Remove unlabeled data
train = train.dropna()

column4 = {}
for i, column in enumerate(sample_submission.columns):
    column4[column] = i

def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column4))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

pf = PolynomialFeatures(degree=2)
train_x = pf.fit_transform(train_x)

for k in range(42, 70):

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.3, random_state=k, shuffle=True)
    
    model = XGBClassifier(tree_method= 'gpu_hist',
                          gpu_id = 0,
                          predictor = 'gpu_predictor',
                          learning_rate = 0.015,
                          max_depth = 6,
                          n_estimators=649,
                          # n_samples = 1
                          )
    model.fit(train_x, train_y)
    
    # Model evaluation
    val_y_pred = model.predict(val_x)

    acc = accuracy_score(val_y, val_y_pred)
    f1 = f1_score(val_y, val_y_pred, average='weighted')
    pre = precision_score(val_y, val_y_pred, average='weighted')
    recall = recall_score(val_y, val_y_pred, average='weighted')
    log = log_loss(val_y, val_y_pred)

    print('Accuracy_score:',acc)
    print('F1 Score:f1',f1)
    print('logloss : ', log)
    y_pred = model.predict_proba(test_x)
    print(y_pred[:,0])
    print('not_delayave : ', y_pred[:,0].mean())
    print('delayavr : ', y_pred[:,1].mean())
    
    if y_pred[:,0].mean()<0.5:
        submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
    
        import datetime
        date = datetime.datetime.now()
        date = date.strftime("%m%d_%H%M")

        save_path = 'd:/study_data/_save/dacon_airplane/'    
        
        submission.to_csv(save_path + date + f'_sample_submission_{k}.csv', float_format='%.3f')


