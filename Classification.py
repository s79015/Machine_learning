#!/usr/bin/env python
# coding: utf-8

# # CS909 2020 Assignment 1: Classification

# # Yerzhan Apsattarov ID(1990463)

# 

# # Question No. 1: (Showing data) [5 Marks]
# Load the training and test data files and answer the following questions:

# # i. How many training and test examples are there? You can use np.loadtxt for this purpose. Show some objects of each class using plt.matshow

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

Xtrain = np.loadtxt('Xtrain.csv') 
Ytrain= np.loadtxt('Ytrain.csv')

Xtest = np.loadtxt('Xtest.csv')
print("Answer:")
print("There are",len(Xtrain),"examples in training data and each example has",len(Xtrain[0]),'features')
print("There are",len(Xtest),"examples in testing data and each example has",len(Xtest[0]),'features')
plt.matshow(Xtrain[:30])
plt.show()

def samplemat(dims):
    
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] =Ytrain[i]
    return aa


# Display matrix
plt.matshow(samplemat((30, 30)))
plt.colorbar()


# Answer:
# 
# There are 3000 examples in training data and each example has 784 features.
# 
# There are 5000 examples in testing data and each example has 784 features.

# In[8]:


print(Xtrain[0])
print(Xtrain[1])


# Answer:
# From Xtrain examples we can see that the features are mixed quite well. For example, first feature starts from 1,0,9... and second feature begins with 5,11,1... Hence, we can not use the hyperparameter "shuffle" in Stratified cross validation.

# # ii. How many positive and negative examples are there in the training dataset?

# In[4]:


neg=0
pos=0
for i in range(len(Ytrain)):
    if Ytrain[i]>0:
        pos+=1
    if Ytrain[i]<0:
        neg+=1   
print("Answer:")
print("There are",pos,"positive examples in training dataset.")   
print("There are",neg,"negative examples in training dataset.")

classes=np.unique(Ytrain)
print("Ytrain has two labels", classes)
        


# Answer:
# There are 1179 (39.3 % of total) positive examples in training dataset. 
# There are 1821 (60.7 % of total) negative examples in training dataset.
# Ytrain has two labels [-1.  1.]

# # iii. Which performance metric (accuracy, AUC-ROC and AUC-PR) should be used? Give your reasoning. 

# Answer: 
# Accuracy:
# This works well only if there is an equal number of samples belonging to each class.
# For example, suppose that in our training set, 98% of class A samples and 2% of class B samples. Then our model can easily get 98% training accuracy by simply predicting each training sample belonging to class A.
# When the same model is tested on a test kit with 60% of Class A samples and 40% of Class B samples, then the test accuracy drops to 60%. The classification accuracy is great, but it gives us a false sense of achieving high accuracy. From the previous answer(Q1(iii)) we know that our data class is not divided equally. Therofore it is not good choice for our dataset.
# 
# AUC-ROC:
# A ROC curve is constructed by plotting the true positive rate (TPR) against the false positive rate (FPR).
# Area under the curve (AUC) is one of the most widely used indicators for estimation. Used for the binary classification problem. The true positive rate is the proportion of observations that were correctly predicted to be positive out of all positive observations (TP/(TP + FN)). Similarly, the false positive rate is the proportion of observations that are incorrectly predicted to be positive out of all negative observations (FP/(TN + FP)). For example, in medical testing, the true positive rate is the rate in which people are correctly identified to test positive for the disease in question.
# AUC-ROC is to calculate the area under the ROC curve. Considering the AUC-ROC we can compare different classification methods and can find best classification.
# If we care equally about the positive and negative class or our dataset is quite balanced, then going with ROC AUC is a best choice for this dataset.
# 
# AUC-PR:
# The precision-recall plot is a model-wide measure for evaluating binary classifiers and closely related to the ROC plot.
# A PR curve is a graph with Precision values on the x-axis and Recall values on the y-axis. The formula for the Precision is TP/(TP+FP) and for Recall is TP/(TP+FN). A precision-recall curve is a great metric for demonstrating the tradeoff between precision and recall for unbalanced datasets. In an unbalanced dataset, one class is substantially over-represented compared to the other. Our dataset is fairly balanced, so a precision-recall curve isn’t the most appropriate metric.
# The precision recall area under curve (AUC-PR) is just the area under the PR curve. The higher it is, the better the model is. 
# 
# Taking everything into account, I think AUC-ROC will be best choice, because our dataset quite balanced and we care equally positive and negative class.

# # iv. What is the expected accuracy of a random classifier (one that generates random labels for a given example) for this problem over the training and test datasets? Demonstrate why this would be the case.

# Answer: The classification accuracy equation for a random classifier (Random Assumption) is as follows:
# 
# Accuracy = 1 / k (here k is the number of classes). In our case, the value of Ytrain has two labels (-1,1), therefore k=2.
# 
# Thus, the classification expected accuracy of a random classifier in our case for training and test datasets is 1/2 = 50%

# # v. What is the AUC-ROC and AUC-PR of a random classifier for this problem over the training and test datasets? Demonstrate why this would be the case.

# Answer: 
# AUC-ROC of a random classifier:
# A classifier with a random level of performance always shows a straight line from the origin (0; 0) to the upper right corner (1; 1). Two areas separated by this ROC curve indicate a simple estimate of the level of performance. ROC curves in the area with the upper left corner (0; 1) indicate good performance levels, while ROC curves in the other area with the lower right corner (1; 0) indicate low performance levels.
# 
# A random classifier will have an AUC close to 0.5. This is easy to understand: for every correct prediction, the next prediction will be incorrect. So the true positive rate and the false positive rate are the same.
# ![a-roc-curve-of-a-random-classifier.png](attachment:a-roc-curve-of-a-random-classifier.png)
# 
# AUC-PR of a random classifier:
# The classifier with a random level of performance shows a straight line. This line divides the callback space into two areas. The divided area above the line is the area of good performance levels. The other area under the line is the area of poor performance.![random-precision-recall-curve1.png](attachment:random-precision-recall-curve1.png)

# # Question No. 2: (Nearest Neighbor Classifier) [5 Marks]
# Perform 5-fold stratified cross-validation (https://scikitlearn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) over the training
# dataset using the k = 1 nearest neighbour classifier and answer the following questions:

# # i. What is the prediction accuracy, AUC-ROC and AUC-PR for each fold using this classifier?
# Show code to demonstrate the results.

# In[4]:


# First way to find the Accuracy, AUC-ROC, AUC-PR (using cross_val_score)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

for i in range(5):
    print("Fold",i+1, "accuracy:", cv_accuracy[i])

for i in range(5):
    print("Fold",i+1,'AUC-ROC :',cv_auc_roc[i] )
    
for i in range(5):
    print("Fold",i+1,'AUC-PR  :',cv_auc_pr[i] )    
    

print("Cross-validation accuracy (5-fold) :", cv_accuracy)
print('Cross-validation (AUC-ROC) (5-fold):',cv_auc_roc )
print('Cross-validation (AUC-PR) (5-fold) :',cv_auc_pr )


# In[7]:


# Second way to find the Accuracy, AUC-ROC, AUC-PR (using roc_auc_score and average_precision_score)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)

def get_accuracy(model, Xtrain,Ytrain,Xtest_cv,Ytest_cv):
    model.fit(Xtrain,Ytrain)
    return model.score(Xtest_cv,Ytest_cv)

def get_auc_roc(model,Xtrain,Ytrain,Xtest_cv,Ytest_cv):
    classifier= model.fit(Xtrain,Ytrain)
    y_score=classifier.predict(Xtest_cv)
    # Compute ROC curve and ROC area for each class
    roc_auc = roc_auc_score(Ytest_cv, y_score)
    return roc_auc

def get_auc_pr(model,Xtrain,Ytrain,Xtest_cv,Ytest_cv):
    classifier= model.fit(Xtrain,Ytrain)
    y_score=classifier.predict(Xtest_cv)
    # Compute ROC curve and ROC area for each class
    roc_pr = average_precision_score(Ytest_cv, y_score)
    return roc_pr


# run
accuracy_list=[]
for i,(tr_idx, v_idx) in enumerate(skf.split(Xtrain,Ytrain)):
    Xtr, Xv = Xtrain[tr_idx], Xtrain[v_idx]
    ytr, yv = Ytrain[tr_idx], Ytrain[v_idx]
    accuracy= get_accuracy(knn,Xtr,ytr,Xv,yv)
    print('Fold',i+1,'accuracy:',accuracy)
    accuracy_list.append(accuracy)

auc_roc_list=[]
for i,(tr_idx, v_idx) in enumerate(skf.split(Xtrain,Ytrain)):
    Xtr, Xv = Xtrain[tr_idx], Xtrain[v_idx]
    ytr, yv = Ytrain[tr_idx], Ytrain[v_idx]
    auc_roc= get_auc_roc(knn,Xtr,ytr,Xv,yv)
    print('Fold',i+1,' AUC_ROC:',auc_roc)
    auc_roc_list.append(auc_roc)

auc_pr_list=[]
for i,(tr_idx, v_idx) in enumerate(skf.split(Xtrain,Ytrain)):
    Xtr, Xv = Xtrain[tr_idx], Xtrain[v_idx]
    ytr, yv = Ytrain[tr_idx], Ytrain[v_idx]
    auc_PR= get_auc_pr(knn,Xtr,ytr,Xv,yv)
    print('Fold',i+1,'  AUC_PR:',auc_PR)
    auc_pr_list.append(auc_PR)    

print("Cross-validation accuracy (5-fold):", accuracy_list)
print('Cross-validation (AUC-ROC) (5-fold):',auc_roc_list )
print('Cross-validation (AUC-PR) (5-fold):',auc_pr_list )


# In[13]:


# Third way to find the Accuracy, AUC-ROC, AUC-PR (using metrics library)

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==-1:
           TN += 1
        if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP,FP,TN,FN)

def get_accuracy_1(TP,FP,TN,FN):
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return ACC

def get_auc_roc_1(y_actual, y_hat):
    FPR, TPR, thresholds = metrics.roc_curve(y_actual, y_hat)
    
    return metrics.auc(FPR, TPR)

def get_auc_pr_1(model,Xtrain,Ytrain,Xtest_cv,Ytest_cv):
    classifier= model.fit(Xtrain,Ytrain)
    y_score=classifier.predict(Xtest_cv)
    # Compute ROC curve and ROC area for each class
    roc_pr = average_precision_score(Ytest_cv, y_score)
    return roc_pr

#Run
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
#Accuracy
accuracy_list=[]
for i,(tr_idx, v_idx) in enumerate(skf.split(Xtrain,Ytrain)):
    Xtr, Xv = Xtrain[tr_idx], Xtrain[v_idx]
    ytr, yv = Ytrain[tr_idx], Ytrain[v_idx]
    knn.fit(Xtr, ytr)
    y_pred = knn.predict(Xv)
    TP,FP,TN,FN=perf_measure(yv, y_pred)
    accuracy= get_accuracy_1(TP,FP,TN,FN)
    print('Fold',i+1,'accuracy:',accuracy)
    accuracy_list.append(accuracy)

#AUC-ROC
auc_roc_list=[]
for i,(tr_idx, v_idx) in enumerate(skf.split(Xtrain,Ytrain)):
    Xtr, Xv = Xtrain[tr_idx], Xtrain[v_idx]
    ytr, yv = Ytrain[tr_idx], Ytrain[v_idx]
    knn.fit(Xtr, ytr)
    y_pred = knn.predict(Xv)
       
    auc_roc= get_auc_roc_1(yv,y_pred)
    print('Fold',i+1,' AUC_ROC:',auc_roc)
    auc_roc_list.append(auc_roc)

#AUC-PR   
auc_pr_list=[]
for i,(tr_idx, v_idx) in enumerate(skf.split(Xtrain,Ytrain)):
    Xtr, Xv = Xtrain[tr_idx], Xtrain[v_idx]
    ytr, yv = Ytrain[tr_idx], Ytrain[v_idx]
    auc_PR= get_auc_pr_1(knn,Xtr,ytr,Xv,yv)
    print('Fold',i+1,'  AUC_PR:',auc_PR)
    auc_pr_list.append(auc_PR)    

print("Cross-validation accuracy (5-fold):", accuracy_list)
print('Cross-validation (AUC-ROC) (5-fold):',auc_roc_list )
print('Cross-validation (AUC-PR) (5-fold):',auc_pr_list )


# Answer:
# Cross-validation accuracy (5-fold): [0.7770382695507487, 0.7533333333333333, 0.7416666666666667, 0.7583333333333333, 0.7378964941569283]
# Cross-validation (AUC-ROC) (5-fold): [0.7685163687021127, 0.7400819519463588, 0.7394067796610169, 0.7576131495623021, 0.727788169277531]
# Cross-validation (AUC-PR) (5-fold): [0.624488904540898, 0.5942294954217806, 0.5833051921548409, 0.6032864300181217, 0.5753581448243409]

# ii. What is the mean and standard deviation of each performance metric (accuracy, AUC-ROC and AUC-PR)across all the folds for this classifier? Show code to demonstrate the results.

# In[17]:


print('Mean of cross-validation accuracy (5-fold):          {:.3f}'.format(np.mean(accuracy_list)))
print('St. deviation of cross-validation accuracy (5-fold): {:.3f}'.format(np.std(accuracy_list)))

print('Mean of cross-validation AUC-ROC (5-fold):           {:.3f}'.format(np.mean(auc_roc_list)))
print('St. deviation of cross-validation AUC-ROC (5-fold):  {:.3f}'.format(np.std(auc_roc_list)))

print('Mean of cross-validation AUC-PR (5-fold):            {:.3f}'.format(np.mean(auc_pr_list)))
print('St. deviation of cross-validation AUC-PR (5-fold):   {:.3f}'.format(np.std(auc_pr_list)))


# Answer:
# Mean cross-validation accuracy (5-fold): 0.754
# St. deviation cross-validation accuracy (5-fold): 0.014
# Mean cross-validation AUC-ROC (5-fold): 0.747
# St. deviation cross-validation AUC-ROC (5-fold): 0.014
# Mean cross-validation AUC-PR (5-fold): 0.596
# St. deviation cross-validation AUC-PR (5-fold): 0.017

# # iii. What is the impact of various forms of pre-processing (https://scikit-learn.org/stable/modules/preprocessing.html ) on the cross-validation performance? Show code to demonstrate the results.

# In[6]:


# Pre-processing "Scale"
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

Xtrain_scaled = preprocessing.scale(Xtrain)
knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'average_precision')
print('Scale. Accuracy:', cv_accuracy)
print('Scale. AUC-ROC :',cv_auc_roc )
print('Scale. AUC-PR  :', cv_auc_pr)

print('Scale. Accuracy. Mean: {:.3f}; Mean without preprocessing: 0.754; Differ:{:.3f}'.format(np.mean(cv_accuracy),np.mean(cv_accuracy)-0.754))
print('Scale. Accuracy. Std : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_accuracy),np.std(cv_accuracy)-0.014))

print('Scale. AUC-ROC. Mean : {:.3f}; Mean without preprocessing: 0.747; Differ:{:.3f}'.format(np.mean(cv_auc_roc),np.mean(cv_auc_roc)-0.747))
print('Scale. AUC-ROC. Std  : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_auc_roc),np.std(cv_auc_roc)-0.014))

print('Scale. AUC-PR. Mean  : {:.3f}; Mean without preprocessing: 0.596; Differ:{:.3f}'.format(np.mean(cv_auc_pr),np.mean(cv_auc_pr)-0.596))
print('Scale. AUC-PR. Std   : {:.3f}; Std without preprocessing : 0.017; Differ:{:.3f}'.format(np.std(cv_auc_pr),np.std(cv_auc_pr)-0.017))


# 

# Answer:
# Pre-processing "Scale".
# Scaling the data brings all values onto one scale eliminating the sparsity and it follows the same concept of Normalization and Standardization. 
# The "scale" have impact on the cross-validation performance. It is improved the mean of Accuracy, AUC-ROC and AUC-PR. However, the increase was not so significant.

# In[7]:


# Pre-processing "Minmax_scale"
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
Xtrain_scaled = preprocessing.minmax_scale(Xtrain)
knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'average_precision')
print('Minmax_scale. Accuracy:', cv_accuracy)
print('Minmax_scale. AUC-ROC :',cv_auc_roc )
print('Minmax_scale. AUC-PR  :', cv_auc_pr)

print('Minmax_scale. Accuracy. Mean: {:.3f}; Mean without preprocessing: 0.754; Differ:{:.3f}'.format(np.mean(cv_accuracy),np.mean(cv_accuracy)-0.754))
print('Minmax_scale. Accuracy. Std : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_accuracy),np.std(cv_accuracy)-0.014))

print('Minmax_scale. AUC-ROC. Mean : {:.3f}; Mean without preprocessing: 0.747; Differ:{:.3f}'.format(np.mean(cv_auc_roc),np.mean(cv_auc_roc)-0.747))
print('Minmax_scale. AUC-ROC. Std  : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_auc_roc),np.std(cv_auc_roc)-0.014))

print('Minmax_scale. AUC-PR. Mean  : {:.3f}; Mean without preprocessing: 0.596; Differ:{:.3f}'.format(np.mean(cv_auc_pr),np.mean(cv_auc_pr)-0.596))
print('Minmax_scale. AUC-RR. Std   : {:.3f}; Std without preprocessing : 0.017; Differ:{:.3f}'.format(np.std(cv_auc_pr),np.std(cv_auc_pr)-0.017))

Answer:
Pre-processing "Minmax_scale".
Transform features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.


The "Minmax_scale" decreased the mean and std of Accuracy, AUC-ROC and AUC-PR. However, the decrease was also not so significant.
# In[8]:


# Pre-processing "RobustScaler"
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(Xtrain)

Xtrain_scaled = transformer.transform(Xtrain)
knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'average_precision')
print('RobustScaler. Accuracy:', cv_accuracy)
print('RobustScaler. AUC-ROC :', cv_auc_roc )
print('RobustScaler. AUC-PR  :', cv_auc_pr)

print('RobustScaler. Accuracy. Mean: {:.3f}; Mean without preprocessing: 0.754; Differ:{:.3f}'.format(np.mean(cv_accuracy),np.mean(cv_accuracy)-0.754))
print('RobustScaler. Accuracy. Std : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_accuracy),np.std(cv_accuracy)-0.014))

print('RobustScaler. AUC-ROC. Mean : {:.3f}; Mean without preprocessing: 0.747; Differ:{:.3f}'.format(np.mean(cv_auc_roc),np.mean(cv_auc_roc)-0.747))
print('RobustScaler. AUC-ROC. Std  : {:.3f}; Std without preprocessing : 0.014 and differ:{:.3f}'.format(np.std(cv_auc_roc),np.std(cv_auc_roc)-0.014))

print('RobustScaler. AUC-PR. Mean  : {:.3f}; Mean without preprocessing: 0.596; Differ:{:.3f}'.format(np.mean(cv_auc_pr),np.mean(cv_auc_pr)-0.596))
print('RobustScaler. AUC-PR. Std   : {:.3f}; Std without preprocessing : 0.017; Differ:{:.3f}'.format(np.std(cv_auc_pr),np.std(cv_auc_pr)-0.017))


# Answer:
# Pre-processing "RobustScaler".
# Scale features using statistics that are robust to outliers.
# This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). 
# The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
# 
# 
# The "RobustScaler" also decreased the mean of Accuracy, AUC-ROC and AUC-PR. Besides, the decrease was higher than "Minmax-scale".

# In[9]:


# Pre-processing "Binarizer"
from sklearn import preprocessing
Xtrain_scaled = preprocessing.binarize(Xtrain)
knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'average_precision')
print('Binarizer. Accuracy:', cv_accuracy)
print('Binarizer. AUC-ROC :',cv_auc_roc )
print('Binarizer. AUC-PR  :', cv_auc_pr)

print('Binarizer. Accuracy. Mean: {:.3f}; Mean without preprocessing: 0.754; Differ:{:.3f}'.format(np.mean(cv_accuracy),np.mean(cv_accuracy)-0.754))
print('Binarizer. Accuracy. Std : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_accuracy),np.std(cv_accuracy)-0.014))

print('Binarizer. AUC-ROC. Mean : {:.3f}; Mean without preprocessing: 0.747; Differ:{:.3f}'.format(np.mean(cv_auc_roc),np.mean(cv_auc_roc)-0.747))
print('Binarizer. AUC-ROC. Std  : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_auc_roc),np.std(cv_auc_roc)-0.014))

print('Binarizer. AUC-PR. Mean  : {:.3f}; Mean without preprocessing: 0.596; Differ:{:.3f}'.format(np.mean(cv_auc_pr),np.mean(cv_auc_pr)-0.596))
print('Binarizer. AUC-PR. Std   : {:.3f}; Std without preprocessing : 0.017; Differ:{:.3f}'.format(np.std(cv_auc_pr),np.std(cv_auc_pr)-0.017))


# Answer:
# Pre-processing "Binarizer".
# Binarize data (set feature values to 0 or 1) according to a threshold
# Values greater than the threshold map to 1, while values less than or equal to the threshold map to 0. With the default threshold of 0, only positive values map to 1.
# 
# The "Binarizer" decreased noticeably than previous 2 preprocessing methods.

# In[18]:


#standard scaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Xtrain_scaled = StandardScaler().fit(Xtrain).transform(Xtrain) 

knn = KNeighborsClassifier(n_neighbors = 1)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain_scaled,Ytrain, cv=skf, scoring = 'average_precision')
print('Binarizer. Accuracy:', cv_accuracy)
print('Binarizer. AUC-ROC :',cv_auc_roc )
print('Binarizer. AUC-PR  :', cv_auc_pr)

print('Binarizer. Accuracy. Mean: {:.3f}; Mean without preprocessing: 0.754; Differ:{:.3f}'.format(np.mean(cv_accuracy),np.mean(cv_accuracy)-0.754))
print('Binarizer. Accuracy. Std : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_accuracy),np.std(cv_accuracy)-0.014))

print('Binarizer. AUC-ROC. Mean : {:.3f}; Mean without preprocessing: 0.747; Differ:{:.3f}'.format(np.mean(cv_auc_roc),np.mean(cv_auc_roc)-0.747))
print('Binarizer. AUC-ROC. Std  : {:.3f}; Std without preprocessing : 0.014; Differ:{:.3f}'.format(np.std(cv_auc_roc),np.std(cv_auc_roc)-0.014))

print('Binarizer. AUC-PR. Mean  : {:.3f}; Mean without preprocessing: 0.596; Differ:{:.3f}'.format(np.mean(cv_auc_pr),np.mean(cv_auc_pr)-0.596))
print('Binarizer. AUC-PR. Std   : {:.3f}; Std without preprocessing : 0.017; Differ:{:.3f}'.format(np.std(cv_auc_pr),np.std(cv_auc_pr)-0.017))


# The answer:
# 
# Considering all 5 preprocessing methods, we can conclude that "Preprocessing scale" and "StandardScaler"show a unnoticeable increase for Accuracy, AUC-ROC and AUC-PR. Other 3 preprocessing results decreased the Accuracy, AUC-PR and AUC-ROC. Hence, we can say that preprocessing can not help for KNN classifier(n_neighbours=1) so much.

# # iv. Use 5-fold cross-validation over training data to calculate the optimal value of k for the k-Nearest neighbour classifier. What is the optimal value of k and what are the cross-validation accuracy, AUC-ROC and AUC-PR? Show code to demonstrate the results.

# In[2]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# choose k between 1 to 101
k_range = range(1, 101)
# Empty lists for "Accuracy", "AUC-ROC" and "AUC-PR"
acc_scores = []
roc_scores = []
pr_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    skf = StratifiedKFold(n_splits=5)
    scores_acc = cross_val_score(knn, Xtrain, Ytrain, cv=skf, scoring='accuracy')
    scores_roc = cross_val_score(knn, Xtrain, Ytrain, cv=skf, scoring='roc_auc')
    scores_pr = cross_val_score(knn, Xtrain, Ytrain, cv=skf, scoring='average_precision')
    acc_scores.append(scores_acc.mean())
    roc_scores.append(scores_roc.mean())
    pr_scores.append(scores_pr.mean())
# plot accuracy to see clearly
plt.plot(k_range, acc_scores, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label="Accuracy")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.show()
#plot roc to see clearly
plt.plot(k_range, roc_scores,marker='o', markerfacecolor='blue',color='red', linewidth=2, label="AUC-ROC")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated AUC-ROC')
plt.legend()
plt.show()
#plot pr to see clearly
plt.plot(k_range, pr_scores,color='green',marker='o', markerfacecolor='blue', linewidth=2, linestyle='dashed', label="AUC-PR")
plt.xlabel('Value of K for KNN')
plt.ylabel("Cross-Validated AUC-PR")
plt.legend()
plt.show()


# In[3]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# choose k between 1 to 35
k_range = range(1, 35)
# Empty lists for "Accuracy", "AUC-ROC" and "AUC-PR"
acc_scores = []
roc_scores = []
pr_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    skf = StratifiedKFold(n_splits=5)
    scores_acc = cross_val_score(knn, Xtrain, Ytrain, cv=skf, scoring='accuracy')
    scores_roc = cross_val_score(knn, Xtrain, Ytrain, cv=skf, scoring='roc_auc')
    scores_pr = cross_val_score(knn, Xtrain, Ytrain, cv=skf, scoring='average_precision')
    acc_scores.append(scores_acc.mean())
    roc_scores.append(scores_roc.mean())
    pr_scores.append(scores_pr.mean())
# plot accuracy to see clearly
plt.plot(k_range, acc_scores, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label="Accuracy")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.show()
#plot roc to see clearly
plt.plot(k_range, roc_scores,marker='o', markerfacecolor='blue',color='red', linewidth=2, label="AUC-ROC")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated AUC-ROC')
plt.legend()
plt.show()
#plot pr to see clearly
plt.plot(k_range, pr_scores,color='green',marker='o', markerfacecolor='blue', linewidth=2, linestyle='dashed', label="AUC-PR")
plt.xlabel('Value of K for KNN')
plt.ylabel("Cross-Validated AUC-PR")
plt.legend()
plt.show()


# Answer: 
# Considering the graphs we can conclude that the KNN classifier with number of neighbours:
# 
# -from K=15 to k= 20 shows best results of "Accuracy";
# 
# -from K=15 to k= 35 shows best results of "AUC-ROC";
# 
# -from K=20 to k= 40 shows best results of "AUC-PR". 
# 
# In order to find precise values of KNN for each of them we will investigate further.

# In[17]:


#n_neighbors=15
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
knn = KNeighborsClassifier(n_neighbors=15)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('Mean cross-validation accuracy k=15 Xtrain (5-fold): {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean cross-validation AUC-ROC k=15 Xtrain (5-fold) : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean cross-validation AUC-PR k=15 Xtrain (5-fold)  : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[18]:


#n_neighbors=17
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
knn = KNeighborsClassifier(n_neighbors=17)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('Mean cross-validation accuracy k=17 Xtrain (5-fold): {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean cross-validation AUC-ROC k=17 Xtrain (5-fold) : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean cross-validation AUC-PR k=17 Xtrain (5-fold)  : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[19]:


#n_neighbors=18
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
knn = KNeighborsClassifier(n_neighbors=18)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('Mean cross-validation accuracy k=18 Xtrain (5-fold): {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean cross-validation AUC-ROC k=18 Xtrain (5-fold) : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean cross-validation AUC-PR k=18 Xtrain (5-fold)  : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[22]:


#grid search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold

k_range = list(range(1, 41))
param_grid = dict(n_neighbors=k_range)
skf = StratifiedKFold(n_splits=5)

#Accuracy
grid = GridSearchCV(knn, param_grid, cv=skf, scoring='accuracy')
grid.fit(Xtrain, Ytrain)

print("The best classifier is: ", grid.best_estimator_, "and best score of 'Accuracy'",grid.best_score_)

# AUC_ROC
grid = GridSearchCV(knn, param_grid, cv=skf, scoring='roc_auc')
grid.fit(Xtrain, Ytrain)

print("The best classifier is: ", grid.best_estimator_, "and best score of 'AUC-ROC'",grid.best_score_)

#AUC_PR
grid = GridSearchCV(knn, param_grid, cv=skf, scoring='average_precision')
grid.fit(Xtrain, Ytrain)

print("The best classifier is: ", grid.best_estimator_, "and best score of 'AUC-PR'",grid.best_score_)


# In[16]:


#n_neighbors=16
knn = KNeighborsClassifier(n_neighbors=16)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('Mean cross-validation accuracy k=16 Xtrain (5-fold): {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean cross-validation AUC-ROC k=16 Xtrain (5-fold) : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean cross-validation AUC-PR k=16 Xtrain (5-fold)  : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[26]:


#n_neighbors=25
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
knn = KNeighborsClassifier(n_neighbors=25)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('Mean cross-validation accuracy k=25 Xtrain (5-fold): {}'.format(np.mean(cv_accuracy)))
print('Mean cross-validation AUC-ROC k=25 Xtrain (5-fold) : {}'.format(np.mean(cv_auc_roc)))
print('Mean cross-validation AUC-PR k=25 Xtrain (5-fold)  : {}'.format(np.mean(cv_auc_pr)))


# In[27]:


#n_neighbors=31
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
knn = KNeighborsClassifier(n_neighbors=31)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('Mean cross-validation accuracy k=31 Xtrain (5-fold): {}'.format(np.mean(cv_accuracy)))
print('Mean cross-validation AUC-ROC k=31 Xtrain (5-fold) : {}'.format(np.mean(cv_auc_roc)))
print('Mean cross-validation AUC-PR k=31 Xtrain (5-fold)  : {}'.format(np.mean(cv_auc_pr)))


# In[28]:


from IPython.display import HTML, display
import tabulate
table = [["" ,"KNeighbors (n_neigh=1)","KNeighbors (n_neigh=15)","KNeighbors (n_neigh=16)","KNeighbors (n_neigh=17)","KNeighbors (n_neigh=18)","KNeighbors (n_neigh=25)","KNeighbors (n_neigh=31)"],
        ['Accuracy',0.754,0.764,0.774,0.769,0.764,0.7716569749730786,0.767663630547122],
        ['AUC-ROC',0.747,0.851,0.851,0.852,0.850,0.8534422243365316,0.8526846953366745],
        ['AUC-PR',0.596,0.752,0.753,0.755,0.754,0.7616813279019118,0.7624389642278805],
        ]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# Answer: 
# As can be seen in the table the KNN classifier with number of neighbours:
# 
# -k= 16 shows best results of "Accuracy" (0.774);
# 
# -k= 25 shows best results of "AUC-ROC" (0.85344);
# 
# -k= 31 shows best results of "AUC-PR" (0.76243). 

# # Question No. 3: [5 Marks]
# Use 5-fold stratified cross-validation over training data to choose an optimal classifier between: k-nearest neighbour, Perceptron, Naïve Bayes Classifier, Logistic regression, Linear SVM and Kernelized SVM. Be sure to tune the hyperparameters of each classifier type (k for k-nearest neighbour, C and kernel type and parameters for SVM and so on). Report the cross validation results (mean and standard deviation of accuracy, AUC-ROC and AUC-PR across fold) of your best model. You may look into grid search as well as ways of pre-processing data. Show code to demonstrate the results. Also show the comparison of these classifiers using a single table.

# Answer:
# K-nearest neighbour classifier:
# From question 2.iv we know that the best "accuracy", "AUC-ROC" and "AUC-PR" for K-nearest neighbour classifier:
# 
# -k_neighbour = 16 shows best results of "Accuracy" (0.774);
# 
# -k_neighbour = 25 shows best results of "AUC-ROC" (0.85344);
# 
# -k_neighbour = 31 shows best results of "AUC-PR" (0.76243).
# 
# Also we know mean of Accuracy, AUC-ROC and AUC-PR for K-nearest neighbour classifier with k=1.

# In[19]:


#Perceptron classifier (max_iter=60, eta0=0.15, random_state=0) and (max_iter=100, eta0=0.01, random_state=0)
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=60, eta0=0.15, random_state=0)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(ppn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(ppn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(ppn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".  Perceptron(max_iter=60, eta0=0.15, random_state=0). : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   Perceptron(max_iter=60, eta0=0.15, random_state=0). : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    Perceptron(max_iter=60, eta0=0.15, random_state=0). : {:.3f}'.format(np.mean(cv_auc_pr)))

ppn_2 = Perceptron(max_iter=100, eta0=0.01, random_state=0)
cv_accuracy = cross_val_score(ppn_2,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(ppn_2,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(ppn_2,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print("")
print('Mean of "Accuracy".  Perceptron(max_iter=100, eta0=0.01, random_state=0).: {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   Perceptron(max_iter=100, eta0=0.01, random_state=0).: {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    Perceptron(max_iter=100, eta0=0.01, random_state=0).: {:.3f}'.format(np.mean(cv_auc_pr)))


# Perceptron classifier results:
# Mean of "Accuracy".  Perceptron classifier(max_iter=60, eta0=0.15, random_state=0). Cross-validation Xtrain (5-fold): 0.652
# Mean of "AUC-ROC". Perceptron classifier(max_iter=60, eta0=0.15, random_state=0). Cross-validation Xtrain (5-fold): 0.686
# Mean of "AUC-PR".  Perceptron classifier(max_iter=60, eta0=0.15, random_state=0). Cross-validation Xtrain (5-fold): 0.552
# 
# Mean of "Accuracy".  Perceptron classifier(max_iter=100, eta0=0.01, random_state=0). Cross-validation Xtrain (5-fold): 0.652
# Mean of "AUC-ROC". Perceptron classifier(max_iter=100, eta0=0.01, random_state=0). Cross-validation Xtrain (5-fold): 0.686
# Mean of "AUC-PR".  Perceptron classifier(max_iter=100, eta0=0.01, random_state=0). Cross-validation Xtrain (5-fold): 0.552   
# 
# 
# Considering the results of perceptron classifier with different hyperparameters, we can conclude the outputs were the same.

# In[20]:


#Gaussian Naive bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(gnb,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(gnb,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(gnb,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".  Gaussian Naive Bayes. : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   Gaussian Naive Bayes. : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    Gaussian Naive Bayes. : {:.3f}'.format(np.mean(cv_auc_pr)))


# Gaussian Naive Bayes classifier results:
# Mean of "Accuracy".  Gaussian Naive Bayes classifier. Cross-validation Xtrain (5-fold): 0.665
# Mean of "AUC-ROC". Gaussian Naive Bayes classifier. Cross-validation Xtrain (5-fold): 0.719
# Mean of "AUC-PR".  Gaussian Naive Bayes classifier. Cross-validation Xtrain (5-fold): 0.554   

# In[5]:


#Bernoulli Naive Bayes classifier binarize=0.0
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

model = BernoulliNB(binarize=0.0)
skf = StratifiedKFold(n_splits=5)

cv_accuracy = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".  Bernoulli Naive Bayes(binarize=0.0). : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   Bernoulli Naive Bayes(binarize=0.0). : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    Bernoulli Naive Bayes(binarize=0.0). : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[6]:


#Bernoulli Naive Bayes classifier binarize=1.0
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

model = BernoulliNB(binarize=1.0)
skf = StratifiedKFold(n_splits=5)

cv_accuracy = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".  Bernoulli Naive Bayes(binarize=1.0). : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   Bernoulli Naive Bayes(binarize=1.0). : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    Bernoulli Naive Bayes(binarize=1.0). : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[9]:


# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

model = MultinomialNB()
skf = StratifiedKFold(n_splits=5)

cv_accuracy = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".  MultinomialNB. : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   MultinomialNB. : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    MultinomialNB. : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[21]:


#Logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
logisticRegr =  LogisticRegression()
skf = StratifiedKFold(n_splits=5)
#logisticRegr.fit(Xtrain,Ytrain)
cv_accuracy = cross_val_score(logisticRegr,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(logisticRegr,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(logisticRegr,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".  Logistic regression classifier. : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".   Logistic regression classifier. : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".    Logistic regression classifier. : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[25]:


#Linear SVM (C=0.001)
from sklearn.model_selection import cross_val_score
from sklearn import svm
skf = StratifiedKFold(n_splits=5)
lin_svm = svm.SVC(kernel='linear', C = 0.001)
cv_accuracy = cross_val_score(lin_svm,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(lin_svm,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(lin_svm,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".Linear SVM (C=0.001). : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC". Linear SVM (C=0.001). : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".  Linear SVM (C=0.001). : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[4]:


#Linear SVM (C=0.0001)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
skf = StratifiedKFold(n_splits=5)
lin_svm = svm.SVC(kernel='linear', C = 0.0001)
cv_accuracy = cross_val_score(lin_svm,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(lin_svm,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(lin_svm,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".Linear SVM (C=0.0001). : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC". Linear SVM (C=0.0001). : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".  Linear SVM (C=0.0001). : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[6]:


#Kernel='rbf'
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
kernel_rbf = SVC(kernel='rbf')
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".Gaussian Kernel. : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC". Gaussian Kernel. : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".  Gaussian Kernel. : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[7]:


# Sigmoid Kernel
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
kernel_rbf = SVC(kernel='sigmoid')
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy".Sigmoid Kernel. : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC". Sigmoid Kernel. : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".  Sigmoid Kernel. : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[4]:


# Using grid search in order to find the best SVM classifier

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

# Parameters of SVM for Grid search
C_range = 10. ** np.arange(-4, 2)
gamma_range = 10. ** np.arange(-4, 2)
param_grid = dict(gamma=gamma_range, C=C_range)

#stratified K fold with n_splits=5
skf = StratifiedKFold(n_splits=5)

# AUC_ROC
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skf, scoring='roc_auc')
grid.fit(Xtrain, Ytrain)
print("The best classifier is: ", grid.best_estimator_, "and best score of 'AUC-ROC'",grid.best_score_)


# In[3]:


#Kernel RBF with optimized hyperparametres
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
kernel_rbf = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(kernel_rbf,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('Mean of "Accuracy". Kernel "RBF". : {:.3f}'.format(np.mean(cv_accuracy)))
print('Mean of "AUC-ROC".  Kernel "RBF". : {:.3f}'.format(np.mean(cv_auc_roc)))
print('Mean of "AUC-PR".   Kernel "RBF". : {:.3f}'.format(np.mean(cv_auc_pr)))


# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
C_param_range = [0.001,0.01,0.1,10]
for i in C_param_range:
    model = LogisticRegression(penalty = 'l2', C = i,random_state = 0)
    skf = StratifiedKFold(n_splits=5)
    cv_accuracy = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring='accuracy')
    cv_auc_roc = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
    cv_auc_pr = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
    
    print('Mean of "Accuracy".  LogisticRegression (C={}). : {:.3f}'.format(i,np.mean(cv_accuracy)))
    print('Mean of "AUC-ROC".   LogisticRegression (C={}). : {:.3f}'.format(i,np.mean(cv_auc_roc)))
    print('Mean of "AUC-PR".    LogisticRegression (C={}). : {:.3f}'.format(i,np.mean(cv_auc_pr)))


# In[7]:


#create the first table to compare the results

from IPython.display import HTML, display
import tabulate
table = [["" ,"KNeighbors (n_neigh=1)","KNeighbors (n_neigh=16)","KNeighbors (n_neigh=25)","KNeighbors (n_neigh=31)","Perceptron","Gaussian Naive Bayes","Logistic regression","Linear SVM (C=0.001)","Linear SVM (C=0.0001)","Gaussian Kernel", "Sigmoid Kernel", "SVM, kernel=RBF, C=10.0, gamma=0.0001"],
        ['Accuracy',0.754,0.774,0.771656,0.767663,0.652,0.665,0.652,0.662,0.674,0.617,0.607,0.617],
        ['AUC-ROC',0.747,0.851,0.853442,0.852684,0.686,0.719,0.692,0.697,0.772,0.529,0.500,0.529],
        ['AUC-PR',0.596,0.753,0.761681,0.762438,0.552,0.554,0.554,0.557,0.582,0.417,0.393,0.417],
        ]
display(HTML(tabulate.tabulate(table, tablefmt='html')))



# In[16]:


#create the second table to compare the results

from IPython.display import HTML, display
import tabulate
table = [["" ,"Bernoulli Naive Bayes(binarize=0.0)","Bernoulli Naive Bayes(binarize=1.0)","Multinomial Naive Bayes","LogisticRegression (C=0.001)","LogisticRegression (C=0.1)","LogisticRegression (C=10)"],
        ['Accuracy',0.653,0.652,0.681,0.658,0.648,0.651],
        ['AUC-ROC',0.742,0.744,0.720,0.702,0.689,0.692],
        ['AUC-PR',0.574,0.578,0.554,0.562,0.552,0.554],
        ]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# Answers:
# According to the above 2 tables one can conclude the next:
# 
# Kneighbors classifier with n_neighbors=16 shows best results in "Accuracy".
# 
# Kneighbors classifier with n_neighbors=25 shows best results in "AUC-ROC".
# 
# Kneighbors classifier with n_neighbors=31 shows best results in "AUC-PR" .
# 
# As we know from Q1, for our dataset I decided to use for comparison the "AUC-ROC". Therefore the best model is the Kneighbors classifier with n_neighbors=25.
# 

# In[8]:


#Results (standard deviation and mean) of Best model Kneighbors classifier with n_neighbors=25.
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
knn = KNeighborsClassifier(n_neighbors=25)
skf = StratifiedKFold(n_splits=5)
#Xtrain_scaled = preprocessing.scale(Xtrain)
cv_accuracy = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(knn,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')
print('KNeighborsClassifier(n_neighbors=25). Accuracy. Mean: {:.3f}'.format(np.mean(cv_accuracy)))
print('KNeighborsClassifier(n_neighbors=25). Accuracy. Std : {:.3f}'.format(np.std(cv_accuracy)))

print('KNeighborsClassifier(n_neighbors=25). AUC-ROC.  Mean: {:.3f}'.format(np.mean(cv_auc_roc)))
print('KNeighborsClassifier(n_neighbors=25). AUC-ROC.  Std : {:.3f}'.format(np.std(cv_auc_roc)))

print('KNeighborsClassifier(n_neighbors=25). AUC-PR.   Mean: {:.3f}'.format(np.mean(cv_auc_pr)))
print('KNeighborsClassifier(n_neighbors=25). AUC-PR.   Std : {:.3f}'.format(np.std(cv_auc_pr)))


# Answer:
# 
# The best model results:
# 
# KNeighborsClassifier(n_neighbors=25). Accuracy. Mean: 0.772
# 
# KNeighborsClassifier(n_neighbors=25). Accuracy. Std : 0.014
# 
# KNeighborsClassifier(n_neighbors=25). AUC-ROC.  Mean: 0.853
# 
# KNeighborsClassifier(n_neighbors=25). AUC-ROC.  Std : 0.009
# 
# KNeighborsClassifier(n_neighbors=25). AUC-PR.   Mean: 0.762
# 
# KNeighborsClassifier(n_neighbors=25). AUC-PR.   Std : 0.014

# # Question No. 4 [5 Marks]

# # i. Reduce the number of dimensions of the data using PCA to 2 and plot a scatter plot of the training data. What are your observations about the data based on data?

# In[9]:


#PCA without preprocessing
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(Xtrain)
X_train_pca = pca.transform(Xtrain) #regular PCA
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c = Ytrain)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('The training data PCA (n_components = 2) without preprocessing')
plt.show()


# In[29]:


#PCA with preprocessing
#### Using PCA to find the first two principal components of the training data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
Xtrain_scaled = preprocessing.scale(Xtrain)

# Number of PCA components=2
pca = PCA(n_components = 2).fit(Xtrain_scaled)

# Train PCA
Xtrain_pca = pca.transform(Xtrain_scaled)
print(Xtrain.shape, Xtrain_pca.shape, Ytrain.shape)

# Using PCA to plot the first two principal components of the training data
import matplotlib.pyplot as plt

colors = ['r', 'g']
plt.scatter(Xtrain_pca[:,0], Xtrain_pca[:,1],c = Ytrain)

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('The training data PCA (n_components = 2) with preprocessing')

plt.show()


# Answer: 
# 
# Principal component analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to make data easy to explore and visualize.
# 
# Comparing the graphs, we can see that after preprocessing the scale of axes diminished. Also, we can see in both pictures that First principal component can not divide quite clear. The reason might be that when we decrease the number of dimensions of Training data from 784 to 2, then we can lose a significant amount of data.
# 
# On the other hand, we can see that most of the blue points are below Zero in the First principal component axis and yellow points above of point 0 of First principal component axis. If we create a vertical line at 0 in the First principal component axis, I can assume that the established line can give us the value of accuracy around 60-70 per cent.

# # ii. Plot the scree graph of PCA and find the number of dimensions that explain 95% variance in the training set.

# In[11]:


from sklearn.decomposition import PCA
pca115 = PCA(n_components=115)

#training PCA
pca115.fit(Xtrain) 

#projecting the data onto Principal components
projected = pca115.transform(Xtrain)

#print shapes
print(Xtrain.shape)
print(projected.shape)

#plot results
plt.plot(np.arange(len(pca115.explained_variance_ratio_))+1,np.cumsum(pca115.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca115.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph')
plt.grid()
plt.show()


# In[14]:


from sklearn import preprocessing
from sklearn.decomposition import PCA

for i in range (110,115):
    pca = PCA(n_components=i)
    #training PCA
    pca.fit(Xtrain) 
#projecting the data onto Principal components
    projected = pca.transform(Xtrain)
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))


# Answer: According to the results we need to use at least 112 dimensions in order to explain 95% variance in the training set.

# # iii. Reduce the number of dimensions of the data using PCA and perform classification. What is the (optimal) cross-validation performance of a Kernelized SVM and XGBoost classification with PCA? Remember to perform hyperparameter optimization!

# In[16]:


# cross-validation performance of a Kernelized SVM with PCA=115
from sklearn.decomposition import PCA

pca = PCA(n_components=112).fit(Xtrain)
X_train_pca = pca.transform(Xtrain)

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

# Parameters of SVM for Grid search
C_range = 10. ** np.arange(-4, 8)
gamma_range = 10. ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)

#stratified K fold with n_splits=5
skf = StratifiedKFold(n_splits=5)

# For scoring we will use 'Auc-ROC', because from Q 1.3 we know for our dataset is the result of 'AUC-ROC' is most suitable 

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skf,scoring = 'roc_auc')

grid.fit(X_train_pca, Ytrain)

print("The best classifier is: ", grid.best_estimator_, "and best score of 'AUC-ROC'",grid.best_score_)


# The best classifier in SVM with these hyperparametres:  SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False) and best score of 'AUC-ROC' 0.7964734906565261

# In[3]:


#cross-validation performance of a XGBoost classification with PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

#PCA=112
from sklearn.decomposition import PCA
pca = PCA(n_components=112).fit(Xtrain)
X_train_pca = pca.transform(Xtrain)

# Grid search and SKF
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier


# Parameters of XGBoost for Grid search
param_grid ={
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

#stratified K fold with n_splits=5
skf = StratifiedKFold(n_splits=5)

# For scoring we will use 'Auc-ROC', because from Q 1.3 we know for our dataset is the result of 'AUC-ROC' is most suitable
grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=skf,scoring = 'roc_auc')

grid.fit(X_train_pca, Ytrain)

print("The best classifier is: ", grid.best_estimator_, "and best score of 'AUC-ROC'",grid.best_score_)


# The best XGBClassifier with these hyperparametres:  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.6, gamma=1,
#               learning_rate=0.1, max_delta_step=0, max_depth=5,
#               min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=1) and best score of 'AUC-ROC' 0.8818673426491669

#  Answer: 
#  
#  XGBoost classification:
#  
# XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. 
# 
# Kernelized SVM:
# 
# Kernel Support Vector Machine (SVM) is useful to deal with nonlinear classification based on a linear discriminant function in a high-dimensional (kernel) space. It employs kernel trick which permits us to work in the input space instead of dealing with a potentially high-dimensional, even theoretically infinite dimensional, kernel (feature) space. Also kernel trick has become so popular that it is used in a variety of other pattern recognition and machine learning algorithms.
# 
#  In order to find  the (optimal) cross-validation performance of a Kernelized SVM and XGBoost classification with PCA I used Grid search method from SKLearn.
#  
#  Comparing the obtained results of Kernelized SVM and XGBoost classification, we can conclude that for our dataset the  XGBoost classification with optimized hyperparametres showed higher value for 'AUC-ROC' than Kernelized SVM with optimized hyperparametres. 

# # Question No. 5 [5 Marks]

# Develop an optimal pipeline for classification based on your analysis (Q1-Q4). You are free to use any tools at your disposal. However, no external data sources may be used. Describe your pipeline and report your results over the test data set. (You are required to submit your prediction file together with the assignment in a zip folder). Your prediction file should be a single column file containing the prediction score of the corresponding example in Xtest (be sure to have the same order!). Your prediction file should be named by your student ID, e.g., u100011.csv.

# # Which classifier need I use?

# I will try to find the best classifier according to AUC-ROC results. And in order to test classifiers, I will use classifiers, which have the optimized hyperparameters. These optimised hyperparameters I will find via Grid Search. However, I will skip in this file the codes of Grid Search, because Jupiter notebook is becoming too large. In the final I will create a single table with results of each classifier.

# In[11]:


from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=1,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.8, verbosity=1)

skf = StratifiedKFold(n_splits=5)
# results on cross-validation
cv_accuracy = cross_val_score(XGB,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(XGB,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(XGB,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('XGBClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('XGBClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('XGBClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# In[12]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

RFC = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features=3, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=1000,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

skf = StratifiedKFold(n_splits=5)
# results on cross-validation
cv_accuracy = cross_val_score(RFC,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(RFC,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(RFC,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('RandomForestClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('RandomForestClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('RandomForestClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# In[20]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#stratified K fold with n_splits=5
skf = StratifiedKFold(n_splits=5)
parameters = {'n_estimators': (1, 2),
                  'base_estimator__C': (1, 2)}
 
# AUC_ROC
grid=GridSearchCV(BaggingClassifier(SVC()), parameters,cv=skf, scoring="roc_auc").fit(Xtrain, Ytrain)

print("The best classifier is: ", grid.best_estimator_)


# In[21]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold

BAGC = BaggingClassifier(base_estimator=SVC(C=1, cache_size=200, class_weight=None,
                                     coef0=0.0, decision_function_shape='ovr',
                                     degree=3, gamma='auto_deprecated',
                                     kernel='rbf', max_iter=-1,
                                     probability=False, random_state=None,
                                     shrinking=True, tol=0.001, verbose=False),
                  bootstrap=True, bootstrap_features=False, max_features=1.0,
                  max_samples=1.0, n_estimators=2, n_jobs=None, oob_score=False,
                  random_state=None, verbose=0, warm_start=False)

skf = StratifiedKFold(n_splits=5)
# results on cross-validation
cv_accuracy = cross_val_score(BAGC,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(BAGC,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(BAGC,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('BaggingClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('BaggingClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('BaggingClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# In[28]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection


# Number of PCA components=112
pca = PCA(n_components = 112).fit(Xtrain)
Xtrain_pca = pca.transform(Xtrain)

skf = StratifiedKFold(n_splits=5)


n_estimators = 150
max_features= 112
min_samples_split= 2

model = ExtraTreesClassifier(n_estimators=n_estimators,
                                     max_features= max_features,
                                     criterion= 'entropy',
                                     min_samples_split= min_samples_split,
                                     max_depth= max_features,
                                     min_samples_leaf= min_samples_split, ## the default is 1 (note we use more than one sample for the split)
                                     class_weight='balanced_subsample',
                                     random_state=1,
                                     verbose=1)

cv_accuracy = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('ExtraTreesClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('ExtraTreesClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('ExtraTreesClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# In[36]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection


# Number of PCA components=112
pca = PCA(n_components = 112).fit(Xtrain)
Xtrain_pca = pca.transform(Xtrain)

skf = StratifiedKFold(n_splits=5)


n_estimators = 150
max_features= 112
min_samples_split= 2

model = ExtraTreesClassifier(n_estimators=n_estimators,
                                     max_features= max_features,
                                     criterion= 'entropy',
                                     min_samples_split= min_samples_split,
                                     max_depth= max_features,
                                     min_samples_leaf= min_samples_split, ## the default is 1 (note we use more than one sample for the split)
                                     class_weight='balanced_subsample',
                                     random_state=1,
                                     verbose=1)

cv_accuracy = cross_val_score(model,Xtrain_pca,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain_pca,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain_pca,Ytrain, cv=skf, scoring = 'average_precision')

print('ExtraTreesClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('ExtraTreesClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('ExtraTreesClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# In[31]:


from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Number of PCA components=112
pca = PCA(n_components = 112).fit(Xtrain)
Xtrain_pca = pca.transform(Xtrain)

skf = StratifiedKFold(n_splits=5)


num_trees = 150
max_features = 112

model = GradientBoostingClassifier(n_estimators=num_trees,max_features= max_features)

cv_accuracy = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain,Ytrain, cv=skf, scoring = 'average_precision')

print('GradientBoostingClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('GradientBoostingClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('GradientBoostingClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# In[37]:


#create the table to compare the results

from IPython.display import HTML, display
import tabulate
table = [["" ,"KNeighbors (n_neigh=25)","XGBClassifier","RandomForest","BaggingClassifier","ExtraTreesClassifier","GradientBoostingClassifier"],
        ['Accuracy',0.772,0.796,0.786, 0.610,0.808,0.793],
        ['AUC-ROC',0.853,0.876,0.874,0.524,0.886,0.871],
        ['AUC-PR',0.762,0.804,0.805,0.413,0.824,0.790],
        ]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# Answer:
# 
# Comparing all results of different classifiers (with optimized hyperparamres)  I found that the AUC-ROC of ExtraTreesClassifier with optimized hyperparameters shows the highest value (0.886). Therefore, for pipeline I will use this type of classifier in the pipeline.

# # Search the optimal component of PCA

# From the scree graph of PCA 115 (Q4,ii) we can see that signifiant increase was in 10 and 15 dimensions 
# and after that the line is not increased noticeably. In order to see the exact number of PCA components in the elbow I will create new scree graph for Number components with range from 1 to 20. 

# In[4]:


#creating new scree graph
pca20 = PCA(n_components=20)
#training PCA
pca20.fit(Xtrain)
#projecting the data onto Principal components
projected = pca20.transform(Xtrain) 
print(Xtrain.shape)
print(projected.shape)
plt.plot(np.arange(len(pca20.explained_variance_ratio_))+1,np.cumsum(pca20.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca20.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph')
plt.grid()
plt.show()


# In[6]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

for i in (10,15,111,112):
    pca = PCA(n_components=i).fit(Xtrain)
    X_train_pca = pca.transform(Xtrain)
    
    n_estimators = 150
    max_features= i
    min_samples_split= 2
    
    model = ExtraTreesClassifier(n_estimators=n_estimators,
                                     max_features= max_features,
                                     criterion= 'entropy',
                                     min_samples_split= min_samples_split,
                                     max_depth= max_features,
                                     min_samples_leaf= min_samples_split, ## the default is 1 (note we use more than one sample for the split)
                                     class_weight='balanced_subsample',
                                     random_state=1,
                                     verbose=1)
    skf = StratifiedKFold(n_splits=5)
    # results on cross-validation
    cv_accuracy = cross_val_score(model,X_train_pca,Ytrain, cv=skf, scoring='accuracy')
    cv_auc_roc = cross_val_score(model,X_train_pca,Ytrain, cv=skf, scoring = 'roc_auc')
    cv_auc_pr = cross_val_score(model,X_train_pca,Ytrain, cv=skf, scoring = 'average_precision')
    print('PCA=',i,'ExtraTreesClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
    print('PCA=',i,'ExtraTreesClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
    print('PCA=',i,'ExtraTreesClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))
    print()


# According to results, we can see that PCA with 111 dimensions shows a higher value of "AUC-ROC" (0.885) than other PCA dimensions. However, before when we will use PCA=112 the value value of "AUC-ROC" was equal to 0.886, but now it is equal to (0.883). I believe that the slight decrease was due to computer or softaware. Therefore I will use PCA 112 in our pipeline.

# # Need I use preprocessing in the pipeline?

# In[39]:


# with preprocessing "scale" 
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

#preprocessing

Xtrain_normalized=preprocessing.scale(Xtrain)

# Number of PCA components=112
pca = PCA(n_components = 112).fit(Xtrain_normalized)

# Train PCA
Xtrain_pca = pca.transform(Xtrain_normalized)

n_estimators = 150
max_features= 112
min_samples_split= 2

# classifier
model = ExtraTreesClassifier(n_estimators=n_estimators,
                                     max_features= max_features,
                                     criterion= 'entropy',
                                     min_samples_split= min_samples_split,
                                     max_depth= max_features,
                                     min_samples_leaf= min_samples_split, ## the default is 1 (note we use more than one sample for the split)
                                     class_weight='balanced_subsample',
                                     random_state=1,
                                     verbose=1)

skf = StratifiedKFold(n_splits=5)
# results on cross-validation
cv_accuracy = cross_val_score(model,Xtrain_pca,Ytrain, cv=skf, scoring='accuracy')
cv_auc_roc = cross_val_score(model,Xtrain_pca,Ytrain, cv=skf, scoring = 'roc_auc')
cv_auc_pr = cross_val_score(model,Xtrain_pca,Ytrain, cv=skf, scoring = 'average_precision')

print('Scale.PCA=112.ExtraTreesClassifier. Accuracy: {}'.format(np.mean(cv_accuracy)))
print('Scale.PCA=112.ExtraTreesClassifier.  AUC-ROC: {}'.format(np.mean(cv_auc_roc)))
print('Scale.PCA=112.ExtraTreesClassifier.  AUC-PR : {}'.format(np.mean(cv_auc_pr)))


# Analys:
# With preprocessing before PCA:
# 
# Scale.PCA=112.ExtraTreesClassifier. Accuracy: 0.7883319833295833
# 
# Scale.PCA=112.ExtraTreesClassifier.  AUC-ROC: 0.877477807033577
# 
# Scale.PCA=112.ExtraTreesClassifier.  AUC-PR : 0.8179683972152525
# 
# Without preprocessing before PCA:
# 
# ExtraTreesClassifier. Accuracy: 0.8083347713002906
# 
# ExtraTreesClassifier.  AUC-ROC: 0.8863586711232738
# 
# ExtraTreesClassifier.  AUC-PR : 0.8241216059592091
# 
# Comparing the results of ExtraTreesClassifier we can conclude next: The result of "Accuracy","AUC-ROC" and "AUC-PR" decreased after preprocessing. Therefore I will not use the preprocessing method in the pipeline.
# 

# Pipeline!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#creating the pipeline



# First step of pipeline the pipeline will reduce the dimensions from 784 to 15, because this number of dimensions represented good result compariong others.

# The second step of th pipeline will use the classification methods. From Q4 we know that XGBClassifier found the best results of  "Auc-ROC" than KNN(n=25).

#Hyperparameter optimisation
n_estimators = 150
max_features= 112
min_samples_split= 2

EXT = make_pipeline(PCA(n_components=112), ExtraTreesClassifier(n_estimators=n_estimators,
                                     max_features= max_features,
                                     criterion= 'entropy',
                                     min_samples_split= min_samples_split,
                                     max_depth= max_features,
                                     min_samples_leaf= min_samples_split, 
                                     class_weight='balanced_subsample',
                                     random_state=1,
                                     verbose=1))

EXT.fit(Xtrain, Ytrain)

pred = EXT.predict(Xtest)


#save results in the csv file
np.savetxt('u1990463.csv', pred, fmt='%d')
print('The file is created')


# Answer:
# 
# Before Q4 the best classifier for 'AUC-ROC'  was the KNeighborsClassifier(n_neighbors=25). However, as we know from Q4 that the result of ExtraTreesClassifier for "AUC-ROC" represented higher value than  KNeighborsClassifier. Therefore,  we will use ExtraTreesClassifier in the pipeline.
# 
# We can use the preprocessing process as a first step in the pipeline. However, as we know the ExtraTreesClassifier classifier with preprocessing showed not good result than without preprocessing. Therefore we will skip the preprocessing step in the pipeline.
# 
# The first step of the pipeline will reduce the dimensions from 784 to 112 because from Q4 and Q5, and we know that the 112 number of dimensions represented the best result comparing others.
# 
# The second step of the pipeline will use the ExtraTreesClassifier with optimized hyperparameters. From the table we know that ExtraTreesClassifier found the best results. 
# 
# The final line of the code will save results to the csv file.
# 

# In[ ]:




