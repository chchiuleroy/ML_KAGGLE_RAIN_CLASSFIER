# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:45:57 2018

@author: roy
"""

import numpy as np; import pandas as pd; import math
from joblib import Parallel, delayed; import multiprocessing

data_temp = pd.read_csv('C:/Users/roy/Documents/z_dataset/seattleWeather_1948-2017.csv')

str = data_temp.iloc[:, 0]
d1 = []
for j in range(len(str)):
 d1.append(str[j].split('-'))

col = ['year', 'month', 'day']
d1 = pd.DataFrame(d1, columns = col)

data = pd.concat([d1, data_temp.iloc[:, range(1, 5)]], axis = 1)

def process(data):
    from sklearn import preprocessing, linear_model, svm, metrics, ensemble, cross_validation, neighbors
    from sklearn.metrics import confusion_matrix
    level, fre = np.unique(data['RAIN'].dropna(axis = 0, how = 'any'), return_counts = True)
    idd0 = np.matrix(np.where(data["RAIN"].isnull())).T
    for i in range(np.size(idd0)):
        data.iloc[idd0[i], 6] = np.random.choice(range(int(level[1])+1), size = 1, replace = True, p = fre/sum(fre))
    data['RAIN'] = data['RAIN'].astype('int')
    
    prcp_new = data['PRCP'].dropna(axis = 0, how = 'any')
    hist, bin_edges = np.histogram(prcp_new, density = True)
    idd1 = np.matrix(np.where(data["PRCP"].isnull())).T
    bin_p = np.random.choice(range(len(bin_edges)-1), size = 1, p = hist * np.diff(bin_edges))
    for j in range(np.size(idd1)):
        data.iloc[idd1[j], 3] = np.random.uniform(bin_edges[bin_p], bin_edges[bin_p+1], size = len(bin_p))
    
    ratio = 0.7; size = data.shape[0]
    
    spl = int(ratio*size)
    idx0 = np.random.choice(range(size), spl, replace = False)
    idx1 = list(set(range(size)).difference(set(idx0)))
    train = data.iloc[idx0, :]; test = data.iloc[idx1, :]
    idx2 = data.shape[1]-1
    train_y = train.iloc[:, idx2]; test_y = test.iloc[:, idx2]
    train_x = train.iloc[:, range(idx2)]; test_x = test.iloc[:, range(idx2)]
        
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    forest.fit(train_x, train_y)
    forest_p_train = forest.predict(train_x); forest_p_test = forest.predict(test_x)
        
    bag = ensemble.BaggingClassifier(n_estimators = 100)
    bag.fit(train_x, train_y)
    bag_p_train = bag.predict(train_x); bag_p_test = bag.predict(test_x)
        
    boost = ensemble.AdaBoostClassifier(n_estimators = 100)
    boost.fit(train_x, train_y)
    boost_p_train = boost.predict(train_x); boost_p_test = boost.predict(test_x)
        
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_x, train_y)
    knn_p_train = knn.predict(train_x); knn_p_test = knn.predict(test_x)
        
    svc = svm.SVC()
    svc.fit(train_x, train_y)
    svc_p_train = svc.predict(train_x); svc_p_test = svc.predict(test_x)
        
    predict_train = np.matrix([forest_p_train, bag_p_train, boost_p_train, knn_p_train, svc_p_train]).T
    predict_test = np.matrix([forest_p_test, bag_p_test, boost_p_test, knn_p_test, svc_p_test]).T
    train_merge = np.mean(predict_train, axis = 1).round()
    test_merge = np.mean(predict_test, axis = 1).round()
        
    acc_train = metrics.accuracy_score(train_y, train_merge)
    acc_test = metrics.accuracy_score(test_y, test_merge)
    
    acc = [acc_train, acc_test]
    
    return acc 

process(data)

run = 100 
cores_n = multiprocessing.cpu_count(); n = cores_n - 4
results = Parallel(n_jobs = n, backend = "threading")(delayed(process)(data) for l in range(run))

