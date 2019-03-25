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
data_path_w4 = 'E:/VRWarehouse/MachineLearning/ciu_w4.csv'

data_file_w1 = open(data_path_w1, 'r')
data_file_w2 = open(data_path_w2, 'r')
data_file_w4 = open(data_path_w4, 'r')

data_w1 = pandas.read_csv(data_file_w1, index_col = False)
data_w2 = pandas.read_csv(data_file_w2, index_col = False)
data_w4 = pandas.read_csv(data_file_w4, index_col = False)


data_w12 = data_w1.append(data_w2,ignore_index = True)

#Down sample or Up sample
pos_data_w12 = data_w12.loc[data_w12['d7_buy'] == 1]
neg_data_w12 = data_w12.loc[data_w12['d7_buy'] == 0]

#data = pos_data.append(neg_data.loc[0:200000],ignore_index = True)

# Over-sampling  
data_w12 = data_w12.append([pos_data_w12]*90,ignore_index = True)

data_w12 = data_w12.sample(frac=1)
# data_length = data.shape[0]
# data_train_length = int(data_length*0.8)
feature_w12 = data_w12.drop(['user_id','item_id','d7_buy'], axis=1)
label_w12 = data_w12.d7_buy
feature_w4 = data_w4.drop(['user_id','item_id'], axis=1)

feature_train = feature_w12
label_train = label_w12
feature_test = feature_w4

#Feature_train, Feature_test, Label_train, Label_test = train_test_split(Feature, Label, test_size = 0.2, random_state = 200, stratify = Label)


rf_claf = RandomForestClassifier(max_depth = 5, n_estimators = 100, max_features = 'sqrt', verbose=False)
rf_claf.fit(feature_train,label_train)

joblib.dump(rf_claf, 'E:/VRWarehouse/MachineLearning/RF_w1w2_w4.pkl') 

#gbdt_clf = joblib.load('/Users/lancy/Documents/sqlite/predict/RF_w1w2_w4.pkl')


# validation and evaluation
prob_pred = rf_claf.predict_proba(feature_test)

prob_pred_1 = [x[1] for x in prob_pred]

label_pred = rf_claf.predict(feature_test)

label_pred = label_pred.tolist()

prob_pred_1 = pandas.DataFrame({'Probrablity':prob_pred_1})

output = data_w4[['user_id','item_id']]
pos_output = output.iloc[[i for i in range(len(label_pred)) if(label_pred[i] == 1)]] 
prob_output = output.join(prob_pred_1)

pos_output.to_csv('E:/VRWarehouse/MachineLearning/RF_w1w2_w4.csv') 
prob_output.to_csv('E:/VRWarehouse/MachineLearning/RF_w1w2_w4_prob.csv') 


#print("F1 Score: " ,end = '') 
#print(f1_score)


data_file_w1.close()
data_file_w2.close()
data_file_w4.close()
elapsed = (time.clock() - time_start)
print("Time used:",elapsed)