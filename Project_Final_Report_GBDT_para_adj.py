import pandas
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import time


# def para_adj(parameter, value, default_value):
#     for val in value:
#         if(parameter == "learning_rate"):
#             p1 = val
#             p2 = default_value[1]
#             p3 = default_value[2]
#         elif(parameter == "n_estimators"):
#             p1 = default_value[0]
#             p2 = val
#             p3 = default_value[2]
#         elif(parameter == "max_depth"):
#             p1 = default_value[0]
#             p2 = default_value[1]
#             p3 = val
#         else:
#             print("Invalid input!")
#             return
#         gbdt_clf = GradientBoostingClassifier(  learning_rate = p1,
#                                                 n_estimators = p2,
#                                                 max_depth = p3,  
#                                                 subsample = default_value[3],
#                                                 max_features = default_value[4],
#                                                 verbose = [5])


time_start = time.clock()

data_path_w1 = 'E:/VRWarehouse/MachineLearning/ciu_w1.csv'
data_path_w2 = 'E:/VRWarehouse/MachineLearning/ciu_w2.csv'
data_path_w4 = 'E:/VRWarehouse/MachineLearning/ciu_w4.csv'

data_file_w1 = open(data_path_w1, 'r')
data_file_w2 = open(data_path_w2, 'r')
data_file_w4 = open(data_path_w4, 'r')

data_w1 = pandas.read_csv(data_file_w1, index_col = False)
data_w2 = pandas.read_csv(data_file_w2, index_col = False)
#data_w4 = pandas.read_csv(data_file_w4, index_col = False)

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

f1_score = []

# f1_score = []
# for value in [0.01,0.05,0.1]:
#     gbdt_clf = GradientBoostingClassifier(learning_rate = value,
#                                             n_estimators=150,
#                                             max_depth=5,  
#                                             subsample=0.8,
#                                             max_features="sqrt",
#                                             verbose=False)
#     gbdt_clf.fit(feature_train, label_train)

#     # validation and evaluation
#     Prob_pred = gbdt_clf.predict_proba(feature_test)
#     Label_pred = gbdt_clf.predict(feature_test)

#     f1_score.append(float(metrics.f1_score(label_test, Label_pred)))
# print("F1 Score: " ,end = '') 
# print(f1_score)

f1_score = []
for value in [50,70,90]:
    gbdt_clf = GradientBoostingClassifier(learning_rate=0.01,
                                            n_estimators = value,
                                            max_depth=7,  
                                            subsample=0.8,
                                            max_features="sqrt",
                                            verbose=False)
    gbdt_clf.fit(feature_train, label_train)

    # validation and evaluation
    Prob_pred = gbdt_clf.predict_proba(feature_test)
    Label_pred = gbdt_clf.predict(feature_test)

    f1_score.append(float(metrics.f1_score(label_test, Label_pred)))
print("F1 Score: " ,end = '') 
print(f1_score)


# f1_score = []
# for value in [3,5,7,10,12]:
#     gbdt_clf = GradientBoostingClassifier(learning_rate=0.05,
#                                             n_estimators = 150,
#                                             max_depth=value,  
#                                             subsample=0.8,
#                                             max_features="sqrt",
#                                             verbose=False)
#     gbdt_clf.fit(feature_train, label_train)

#     # validation and evaluation
#     Prob_pred = gbdt_clf.predict_proba(feature_test)
#     Label_pred = gbdt_clf.predict(feature_test)

#     f1_score.append(float(metrics.f1_score(label_test, Label_pred)))
# print("F1 Score: " ,end = '') 
# print(f1_score)




#print(Label_test.values.tolist())
#print(len(label_test))
print("F1 Score: " ,end = '') 
print(f1_score)
#print(Prob_pred)

data_file_w1.close()
data_file_w2.close()
data_file_w4.close()
elapsed = (time.clock() - time_start)
print("Time used:",elapsed)