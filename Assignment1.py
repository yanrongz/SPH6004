# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:47:16 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np
import mifs
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import svm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,auc
"""
Imputation
"""
data=pd.read_csv("Assignment_1_data.csv",sep=',',na_values=np.nan,index_col=False)
na_count=pd.DataFrame(data.isnull().sum())
data_new=data[na_count[na_count[0]<10000].index]
data_new=data_new.fillna(data_new.mean())

"""
Normalization
"""
disc_features=[1]
num_features=[2,3,5,6,8,9,11,12,17,18,27,4,7,10,13,14,15,16,19,20,21,22,23,24,25,26]
bin_features=[0]
cat_features=[28,29,30,31,32]

scaler=preprocessing.StandardScaler()
data_new.iloc[:,num_features]=scaler.fit_transform(data_new.iloc[:,num_features])
discretizer=preprocessing.KBinsDiscretizer(n_bins=10,strategy='uniform',encode='onehot')
discretizer.fit(np.array(data.age).reshape(-1,1)).bin_edges_
data_discre=pd.DataFrame(discretizer.transform(np.array(data.age).reshape(-1,1)).toarray())
data_discre.columns=['age_1','age_2','age_3','age_4','age_5','age_6','age_7','age_8','age_9','age_10']
data_new.iloc[:,bin_features]=data_new.iloc[:,bin_features].replace(['M','F'],[0,1])
oh_enc=preprocessing.OneHotEncoder(categories='auto',handle_unknown='ignore')
data_cat=pd.DataFrame(oh_enc.fit_transform(pd.DataFrame(data_new.iloc[:,cat_features],dtype=np.int8)).toarray())

"""
Split the dataset
"""
data_newest=data_new.iloc[:,bin_features].join(data_discre).join(data_new.iloc[:,num_features]).join(data_cat)
label=data_new['outcome'].replace([True,False],[1,0])
x_train,x_test,y_train,y_test=train_test_split(data_newest,label,test_size=0.2,shuffle=True,random_state=5) 
#random_state fixes division of training dataset and testing dataset.

"""
Balance the dataset
"""
smo=SMOTE(random_state=0)
x_smo,y_smo=smo.fit_resample(x_train,y_train)
data_smo=x_smo.join(y_smo)
#data_smo.to_csv('data_smo.csv',index=False,header=False) #Prepare the data for matlab to calculate feature scores.

def model_metrics(y_pred,y):
    result={'ACC':accuracy_score(y,y_pred),
            'recall':recall_score(y,y_pred,average='macro'),
            'precision':precision_score(y,y_pred,average='macro'),
            'F1_score':f1_score(y,y_pred,average='macro'),
            'AUC':roc_auc_score(y,y_pred,average='macro')}
    return result 

"""
Feature selection: Filtered 不依赖机器学习的方法（所以每个方法都针对不同的分类器跑一遍）
"""
#######################################--Relief--#######################################

def dist(a,b):
    '''
    #
    :param a: a vector
    :param b: a vector/matrix
    '''
    a=np.array(a)
    b=np.array(b)
    if len(a) != b.shape[1]:
        print('error: Two samples have different columns number.')
    else:
        '''
        Calculate the Euclidean distance between vectors and matrices
        '''
        distance=np.zeros(shape=(1,b.shape[0]), dtype=float)
        for i in range(b.shape[0]):
            distance[0,i]=(sum((a-b[i,:])**2))**(1/2)      
    return distance

def find_this_matrix(a,b):
    '''
    #
    :param a: a mactrix
    :param b: a vector
    :function find_this_matrix: to find a row in a which equals to b
    '''
    a=np.array(a)
    b=np.array(b)
    index=[]
    for j in range(a.shape[0]):
        if (a[j,:]==b).all():
            index.append(j)
    #index=index.T
    return index

def relief(data0,t,k):
    '''
    #
    :param data: Normalized data with label
    :param t: threshold
    :param k: K features you want to selection
    '''
    data=np.array(data0)
    (data_row,data_col)=data.shape  
    all_distance=np.empty((3,data_row),dtype=object)
    for j in range(data_row):
        all_distance[0,j]=data[j,0:data_col-1]
        #store the nearest neighbor sample with same category
        middle_same=data[(data[:,data_col-1]==data[j,data_col-1]),0:data_col-1]
        j_find=find_this_matrix(middle_same,data[j,0:data_col-1])
        middle_same=np.delete(middle_same,j_find,axis=0)
        distance=dist(data[j,0:data_col-1],middle_same)
        min_index=np.where(np.isin(distance, min(distance)))[0].tolist()
        if len(min_index)>1:
            u=min_index[0]
        else:
            u=min_index
        all_distance[1,j]=middle_same[u,:]
        #store the nearest neighbor sample with different category
        middle_same=data[(data[:,data_col-1]!=data[j,data_col-1]),0:data_col-1]
        distance=dist(data[j,0:data_col-1],middle_same)
        min_index=np.where(np.isin(distance, min(distance)))[0].tolist()
        if len(min_index)>1:
            u=min_index[0]
        else:
            u=min_index
        all_distance[2,j]=middle_same[u,:]
                                
    score_matrix=np.zeros((data_row,data_col-1),dtype=float)
    for g in range(data_row):
        score_matrix[g,:]=(all_distance[1,j]-all_distance[3,j])**2-(all_distance[1,j]-all_distance[2,j])**2
        score=np.sum(score_matrix,axis=0)
        index=list(i for i in range(data_col-1))
        index=np.array(index)
        score_index=np.vstack((index,score))
        score_index=score_index.T
        sort_score_index=score_index[np.lexsort(-score_index.T)]
        
        effect_k=sort_score_index[0:k,0]
        effect_k=effect_k.tolist()[0]
        data_matrix_k=data[:,effect_k]
        
        effect_index=np.where(sort_score_index[:,1]>=t)[0]
        effect_t=sort_score_index[effect_index,0]
        effect_t=effect_t.tolist()[0]
        data_matrix_t=data[:,effect_t]
        
    return data_matrix_t,data_matrix_k

#Relief algorithm takes a long time and large space to run in python, the score for every feature was calculate from matlab.
sort_score_index_relief=pd.read_csv("sort_score_index_R.csv",sep=',',index_col=False,header=None)

def MatrixAfterScore(data0,sort_score_index,k):
    data=np.array(data0)
    sort_score_index=np.array(sort_score_index)
    sort_score_index=sort_score_index.T
    effect_k=sort_score_index[0:k,0]
    effect_k=np.array(effect_k,dtype=int)
    effect_k=effect_k-1
    data_matrix_k=data[:,effect_k]
    return data_matrix_k
'''
#svm
#for k in range(10,80,10):
for k in [70]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_relief,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]-1]
    train_label=y_smo
    params={'C':[1，10,100,1000],
            'gamma':[0,0.1,0.01,0.001,0001]}
    model=GridSearchCV(svm.SVC(kernel='rbf'),params,cv=5,scoring='accuracy')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_relief,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    print(result)
'''    

#randomforest
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_relief,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'max_depth':[3,5],
            'min_samples_split':[50,100,150],
            'min_samples_leaf':[20,30]
            }
    model=GridSearchCV(RandomForestClassifier(n_estimators=120),
                   params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_relief,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    print(result)

#adaboost
#since tuning parameters in decision tree takes a very long time, 
#we only tune learning rate and n_estimators to see the trend of learning rate.
record=[]
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_relief,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'learning_rate':[0.05,0.06,0.07,0.08,0.09,0.1],
            'n_estimators':[100,110,120],
            }
    model=GridSearchCV(AdaBoostClassifier(tree.DecisionTreeClassifier()),
                       params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_relief,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    record.append([result])
    print(result)

#lightgbm
record=[]
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_relief,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'max_depth':[3,4,5],
            'num_leaves':[30,50,70,90,110,130,150],
            'min_data_in_leaf':[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
            }
    model=GridSearchCV(lgb.LGBMClassifier(learning_rate=0.1,n_estimators=120,feature_fraction=0.8,lambda_l1=1,lambda_l2=0.01,objective='binary'),
                   params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_relief,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    record.append([result])
    print(result)
    
#LR
record=[]
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_relief,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'C':[0.1,1,10,100],
            'max_iter':[1,10,100,500],
            }
    model=GridSearchCV(LogisticRegression(),
                   params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_relief,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    record.append([result])
    print(result)


'''

#######################################--Fscore--#######################################
#Fscore algorithm was implement by matlab.
sort_score_index_fscore=pd.read_csv("sort_score_index_F.csv",sep=',',index_col=False,header=None)

 ''' 
#######################################--MRMR--#######################################

feat_selector=mifs.MutualInformationFeatureSelector(method='MRMR',k=x_smo.shape[1])
feat_selector.fit(x_smo,y_smo)
sort_score_index=feat_selector.ranking_

'''
Due to python limited memory, MRMR was implemented by matlab.
'''
#Import the result gained for matlab.
sort_score_index_mrmr=pd.read_csv("sort_score_index_mrmr.csv",sep=',',index_col=False,header=None)

#randonforest
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_mrmr,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'max_depth':[3,5],
            'min_samples_split':[50,100,150],
            'min_samples_leaf':[20,30]
            }
    model=GridSearchCV(RandomForestClassifier(n_estimators=120),
                   params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_mrmr,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    print(result)

#adaboost
#since tuning parameters in decision tree takes a very long time, 
#we only tune learning rate and n_estimators to see the trend of learning rate.
record=[]
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_mrmr,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'learning_rate':[0.05,0.06,0.07,0.08,0.09,0.1],
            'n_estimators':[100,110,120],
            }
    model=GridSearchCV(AdaBoostClassifier(tree.DecisionTreeClassifier()),
                       params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_mrmr,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    record.append([result])
    print(result)


#lightgbm
record=[]
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_mrmr,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'max_depth':[3,4,5],
            'num_leaves':[30,50,70,90,110,130,150],
            'min_data_in_leaf':[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
            }
    model=GridSearchCV(lgb.LGBMClassifier(learning_rate=0.1,n_estimators=120,feature_fraction=0.8,lambda_l1=1,lambda_l2=0.01,objective='binary'),
                   params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_mrmr,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    record.append([result])
    print(result)
    
#LR
record=[]
for k in [10,20,30,40,50,60,70,78]:
    data_matrix_k=MatrixAfterScore(data_smo,sort_score_index_mrmr,k)
    train_data=data_matrix_k[:,0:data_matrix_k.shape[1]]
    train_label=y_smo
    params={'C':[0.1,1,10,100],
            'max_iter':[1,10,100,500],
            }
    model=GridSearchCV(LogisticRegression(),
                   params,return_train_score=True,cv=5,scoring='recall')
    model.fit(train_data,train_label)
    print(f"The best params for {k} features: {model.best_params_}")
    test_data=MatrixAfterScore(x_test,sort_score_index_mrmr,k)
    test_label=y_test
    pred=model.predict(test_data)
    result=model_metrics(pred,test_label)
    record.append([result])
    print(result)

"""
Feature selection: Wrapper
"""
#######################################--RFE--#######################################

#################################------------rf------------#################################
'''
#
:params a represents max_depth
:params b represents min_samples_split
:params c represents min_samples_leaf
'''
features=[70]
max_depth_=[3,5]
min_samples_split_=[50,100,150]
min_samples_leaf_=[20,30]
record=[]
for k in features:
    for a in max_depth_:
        for b in min_samples_split_:
            for c in min_samples_leaf_:
                estimator=RandomForestClassifier(max_depth=a,
                                                 min_samples_split=b,
                                                 min_samples_leaf=c,
                                                 n_estimators=120)
                selector=RFE(estimator,n_features_to_select=k,step=1).fit(x_smo,y_smo)
                select_rf=selector.get_support()
                train_data=selector.transform(x_smo)
                train_label=y_smo
                scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
                record.append([k,a,b,c,scores.mean()])

best_k,best_a,best_b,best_c=70,3,100,20

model=RandomForestClassifier(max_depth=best_a,
                             min_samples_split=best_b,
                             min_samples_leaf=best_c,
                             n_estimators=120)
selector=RFE(model,n_features_to_select=best_k,step=1).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_rf=selector.get_support()
test_data=x_test.loc[:,select_rf]
test_label=y_test
pred=predictor.predict(test_data)
result=model_metrics(pred,test_label)
print(result)

#################################------------Adaboost------------#################################
'''
#
:params a represents learning_rate
:params b represents n_estimators
'''
features=[70]
learning_rate_=[0.05,0.07,0.1]
n_estimators_=[100,110,120]
record=[]
for k in features:
    for a in learning_rate_:
        for b in n_estimators_:
            estimator=AdaBoostClassifier(tree.DecisionTreeClassifier(),learning_rate=a,n_estimators=b)
            selector=RFE(estimator,n_features_to_select=k,step=1).fit(x_smo,y_smo)
            train_data=selector.transform(x_smo)
            train_label=y_smo
            scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
            record.append([k,a,b,scores.mean()])

best_k,best_a,best_b=70,0.05,120

model=AdaBoostClassifier(tree.DecisionTreeClassifier(),learning_rate=best_a,n_estimators=best_b)
selector=RFE(model,n_features_to_select=k,step=1).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_ab=selector.get_support()
test_data=x_test.loc[:,select_ab]
pred=predictor.predict(test_data)
test_label=y_test
result=model_metrics(pred,test_label)
print(result)

#################################------------LR------------#################################
'''
#
:params a represents C
:params b represents max_iter
'''

features=[70]
C_=[0.1,1,10,100]
max_iter_=[1,10,100]
record=[]
for k in features:
    for a in C_:
        for b in max_iter_:
            estimator=LogisticRegression(C=a,max_iter=b)
            selector=RFE(estimator,n_features_to_select=k,step=1).fit(x_smo,y_smo)
            select_rf=selector.get_support()
            train_data=selector.transform(x_smo)
            train_label=y_smo
            scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
            record.append([k,a,b,scores.mean()])

best_k,best_a,best_b=70,10,10

model=LogisticRegression(C=best_a,max_iter=best_b)
selector=RFE(model,n_features_to_select=best_k,step=1).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_rf=selector.get_support()
test_data=x_test.loc[:,select_rf]
test_label=y_test
pred=predictor.predict(test_data)
result=model_metrics(pred,test_label)
print(result)

"""
Feature selection: Embedded
"""

#######################################--Select from model--#######################################
'''
#random forest
selector=SelectFromModel(RandomForestClassifier()).fit(x_smo,y_smo)
select_rf=selector.get_support()
test_data=x_test[select_rf]
'''
#################################------------rf------------#################################
'''
#
:params a represents max_depth
:params b represents min_samples_split
:params c represents min_samples_leaf
'''
features=[70]
max_depth_=[3,5]
min_samples_split_=[50,100,150]
min_samples_leaf_=[20,30]
record=[]
for k in features:
    for a in max_depth_:
        for b in min_samples_split_:
            for c in min_samples_leaf_:
                estimator=RandomForestClassifier(max_depth=a,
                                                 min_samples_split=b,
                                                 min_samples_leaf=c,
                                                 n_estimators=120)
                selector=SelectFromModel(estimator,max_features=k,threshold=0).fit(x_smo,y_smo)
                select_rf=selector.get_support()
                train_data=selector.transform(x_smo)
                train_label=y_smo
                scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
                record.append([k,a,b,c,scores.mean()])


best_k,best_a,best_b,best_c=70,3,150,20

model=RandomForestClassifier(max_depth=best_a,
                             min_samples_split=best_b,
                             min_samples_leaf=best_c,
                             n_estimators=120)
selector=SelectFromModel(model,max_features=best_k,threshold=0).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_rf=selector.get_support()
test_data=x_test.loc[:,select_rf]
test_label=y_test
pred=predictor.predict(test_data)
result=model_metrics(pred,test_label)
print(result)


#################################------------Adaboost------------#################################

'''
#
:params a represents learning_rate
:params b represents n_estimators
'''
features=[10]
learning_rate_=[0.05,0.07,0.1]
n_estimators_=[100,110,120]
record=[]
for k in features:
    for a in learning_rate_:
        for b in n_estimators_:
            estimator=AdaBoostClassifier(tree.DecisionTreeClassifier(),learning_rate=a,n_estimators=b)
            selector=SelectFromModel(estimator,max_features=k,threshold=0).fit(x_smo,y_smo)
            train_data=selector.transform(x_smo)
            train_label=y_smo
            scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
            record.append([k,a,b,scores.mean()])

best_k,best_a,best_b=10,0.1,120

model=AdaBoostClassifier(tree.DecisionTreeClassifier(),learning_rate=best_a,n_estimators=best_b)
selector=SelectFromModel(model,max_features=best_k,threshold=0).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_ab=selector.get_support()
test_data=x_test.loc[:,select_ab]
pred=predictor.predict(test_data)
test_label=y_test
result=model_metrics(pred,test_label)
print(result)


"""
#################################------------Gradientboosting------------#################################
selector=SelectFromModel(GradientBoostingClassifier()).fit(x_smo,y_smo)
select_gbdt=selector.get_support()
test_data=x_test[select_gbdt]

'''
#
:params a represents max_depth
:params b represents min_samples_split
:params c represents min_samples_leaf
'''
features=[70]
max_depth_=[5]#[3,5]
min_samples_split_=[150]#[100,200]
min_samples_leaf_=[60]#[50,70]
record=[]
for k in features:
    for a in max_depth_:
        for b in min_samples_split_:
            for c in min_samples_leaf_:
                estimator=GradientBoostingClassifier(max_depth=a,
                                                     min_samples_split=b,
                                                     min_samples_leaf=c,
                                                     learning_rate=0.1,
                                                     n_estimators=120,
                                                     subsample=0.8)
                selector=SelectFromModel(estimator,max_features=best_k,threshold=0).fit(x_smo,y_smo)
                train_data=selector.transform(x_smo)
                train_label=y_smo
                scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
                record.append([k,a,b,c,scores.mean()])

best_k,best_a,best_b,best_c=70,5,150,60

model=GradientBoostingClassifier(max_depth=best_a,
                                 min_samples_split=best_b,
                                 min_samples_leaf=best_c,
                                 learning_rate=0.1,
                                 n_estimators=120,
                                 subsample=0.8)
selector=SelectFromModel(model,max_features=best_k,threshold=0).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_lgb=selector.get_support()
test_data=x_test.loc[:,select_lgb]
pred=predictor.predict(test_data)
result=model_metrics(pred,test_label)
print(result)
"""
#################################------------lightgbm------------#################################
    
'''
#
:params a represents max_depth
:params b represents num_leaves
:params c represents min_data_in_leaf
'''
features=[10]
max_depth_=[5,6,7]
num_leaves_=[30,40,50,60,70]
min_data_in_leaf_=[10,20,30,40,50]
record=[]
for k in features:
    for a in max_depth_:
        for b in num_leaves_:
            for c in min_data_in_leaf_:
                estimator=lgb.LGBMClassifier(max_depth=a,
                                             num_leaves=b,
                                             min_data_in_leaf=c,
                                             learning_rate=0.1,
                                             n_estimators=120,
                                             feature_fraction=0.8,
                                             lambda_l1=1,
                                             lambda_l2=0.01,
                                             objective='binary')
                selector=SelectFromModel(estimator,max_features=k,threshold=0).fit(x_smo,y_smo)
                select_lgb=selector.get_support()
                train_data=selector.transform(x_smo)
                train_label=y_smo
                scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
                record.append([k,a,b,c,scores.mean()])

best_k,best_a,best_b,best_c=10,7,30,20

model=lgb.LGBMClassifier(max_depth=best_a,
                         num_leaves=best_b,
                         min_data_in_leaf=best_c,
                         learning_rate=0.1,
                         n_estimators=120,
                         feature_fraction=0.8,
                         lambda_l1=1,
                         lambda_l2=0.01,
                         objective='binary')
selector=SelectFromModel(model,max_features=k,threshold=0).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_lgb=selector.get_support()
test_data=x_test.loc[:,select_lgb]
pred=predictor.predict(test_data)
result=model_metrics(pred,test_label)
print(result)

#################################------------LR------------#################################
'''
#
:params a represents C
:params b represents max_iter
'''

features=[70]
C_=[0.1,1,10,100]
max_iter_=[1,10,100]
record=[]
for k in features:
    for a in C_:
        for b in max_iter_:
            estimator=LogisticRegression(C=a,max_iter=b)
            selector=SelectFromModel(estimator,max_features=k,threshold=0).fit(x_smo,y_smo)
            select_rf=selector.get_support()
            train_data=selector.transform(x_smo)
            train_label=y_smo
            scores=cross_val_score(estimator,train_data,train_label,cv=5,scoring='recall')
            record.append([k,a,b,scores.mean()])

best_k,best_a,best_b=70,1,100

model=LogisticRegression(C=best_a,max_iter=best_b)
selector=SelectFromModel(model,max_features=k,threshold=0).fit(x_smo,y_smo)
train_data=selector.transform(x_smo)
train_label=y_smo
predictor=model.fit(train_data,train_label)
select_rf=selector.get_support()
test_data=x_test.loc[:,select_rf]
test_label=y_test
pred=predictor.predict(test_data)
result=model_metrics(pred,test_label)
print(result)


#################################------------plot------------#################################

plot_data1=pd.read_csv("Feature selection comparision new.csv",sep=',',na_values=np.nan,index_col=False)
sns.set_style('white')
p1=sns.lineplot(x='dim1',y='AUC1',hue='selector1',data=plot_data1)
plt.legend(title="Feature selection",loc='lower right')
plt.xlabel('Dimension')
plt.ylabel('AUC')
plt.title('Random Forest')

p2=sns.lineplot(x='dim2',y='AUC2',hue='selector2',data=plot_data1)
plt.legend(title="Feature selection",loc='lower right')
plt.xlabel('Dimension')
plt.ylabel('AUC')
plt.title('Adaboost')

p3=sns.lineplot(x='dim3',y='AUC3',hue='selector3',data=plot_data1)
plt.legend(title="Feature selection",loc='lower right')
plt.xlabel('Dimension')
plt.ylabel('AUC')
plt.title('Logistics Regression')


data_matrix_k1=MatrixAfterScore(data_smo,sort_score_index_relief,20)
data_matrix_k2=MatrixAfterScore(data_smo,sort_score_index_relief,50)
data_matrix_k3=MatrixAfterScore(data_smo,sort_score_index_relief,60)
train_data1=data_matrix_k1[:,0:data_matrix_k1.shape[1]]
train_data2=data_matrix_k2[:,0:data_matrix_k2.shape[1]]
train_data3=data_matrix_k3[:,0:data_matrix_k3.shape[1]]
train_label=y_smo
test_data1=MatrixAfterScore(x_test,sort_score_index_relief,20)
test_data2=MatrixAfterScore(x_test,sort_score_index_relief,50)
test_data3=MatrixAfterScore(x_test,sort_score_index_relief,60)

model1=RandomForestClassifier(max_depth=3,min_samples_split=30,min_samples_leaf=50,n_estimators=120)
model1.fit(train_data1,train_label)
pred1=model1.predict(test_data1)

model2=AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=110,learning_rate=0.1)
model2.fit(train_data2,train_label)
pred2=model2.predict(test_data2)

model3=LogisticRegression(C=100,max_iter=100)
model3.fit(train_data3,train_label)
pred3=model3.predict(test_data3)

fpr1,tpr1,threshold1=roc_curve(y_test,pred1)
roc_auc1=auc(fpr1,tpr1)

fpr2,tpr2,threshold2=roc_curve(y_test,pred2)
roc_auc2=auc(fpr2,tpr2)

fpr3,tpr3,threshold3=roc_curve(y_test,pred3)
roc_auc3=auc(fpr3,tpr3)

plt.title("ROC curve")
plt.plot(fpr1,tpr1,'b',label="Random Forest AUC=%0.3f" % roc_auc1)
plt.plot(fpr2,tpr2,'r',label="Adaboost=%0.3f" % roc_auc2)
plt.plot(fpr3,tpr3,'g',label="Logistics Regression=%0.3f" % roc_auc3)
plt.legend(loc="lower right")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show














