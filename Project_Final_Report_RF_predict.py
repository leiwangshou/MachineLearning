import pandas
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
import time

time_start = time.clock()

data_path_w1 = 'E:/VRWarehouse/MachineLearning/ciu_w1.csv'
data_path_w2 = 'E:/VRWarehouse/MachineLearning/ciu_w2.csv'


data_file_w1 = open(data_path_w1, 'r')
data_file_w2 = open(data_path_w2, 'r')


data_w1 = pandas.read_csv(data_file_w1, index_col = False)
data_w2 = pandas.read_csv(data_file_w2, index_col = False)


#Down sample or Up sample
pos_data_w1 = data_w1.loc[data_w1['d7_buy'] == 1]
neg_data_w1 = data_w1.loc[data_w1['d7_buy'] == 0]

#data = pos_data.append(neg_data.loc[0:200000],ignore_index = True)

# Over-sampling  
data_w1 = data_w1.append([pos_data_w1]*90,ignore_index = True)

data_w1 = data_w1.sample(frac=1)
# data_length = data.shape[0]
# data_train_length = int(data_length*0.8)
feature_w1 = data_w1.drop(['user_id','item_id','d7_buy'], axis=1)
label_w1 = data_w1.d7_buy
feature_w2 = data_w2.drop(['user_id','item_id','d7_buy'], axis=1)
label_w2 = data_w2.d7_buy

feature_train = feature_w1
label_train = label_w1
feature_test = feature_w2
label_test = label_w2

#Feature_train, Feature_test, Label_train, Label_test = train_test_split(Feature, Label, test_size = 0.2, random_state = 200, stratify = Label)


rf_claf = RandomForestClassifier(max_depth = 5, n_estimators = 100, max_features = 'sqrt', verbose=False)
rf_claf.fit(feature_train,label_train)

joblib.dump(rf_claf, 'E:/VRWarehouse/MachineLearning/RF__predict.pkl') 

# validation and evaluation
Prob_pred = rf_claf.predict_proba(feature_test)
Label_pred = rf_claf.predict(feature_test)

f1_score = float(metrics.f1_score(label_test, Label_pred))


print("F1 Score: " ,end = '') 
print(f1_score)


data_file_w1.close()
data_file_w2.close()
elapsed = (time.clock() - time_start)
print("Time used:",elapsed)
