import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, Normalizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

input_file = "./Code/Data/final.csv"
data = pd.read_csv(input_file)

X = data[["subscribe_cnt", "Danceability"]]

# All
'''
X = data[["subscribe_cnt","Replay_Gain","BPM","Dissonance","Pitch_Salience","Gender","Danceability",
            "Genre_Alternative","Genre_Blues","Genre_Electronic","Genre_FolkCountry","Genre_FunkSoulRNB",
            "Genre_Jazz","Genre_Pop","Genre_RapHippop","Genre_Rock","Mood_Acoustic","Mood_Aggressive",
            "Mood_Electronic","Mood_Happy","Mood_Party","Mood_Relaxed","Mood_Sad",
            "Mood_MirexC1","Mood_MirexC2","Mood_MirexC3","Mood_MirexC4","Mood_MirexC5","Atonal","Instrumental","Bright_Timbre"]]
'''

'''
X = data[["subscribe_cnt","Dissonance","Pitch_Salience","Gender","Danceability",
            "Genre_Alternative","Genre_Blues","Genre_Electronic","Genre_FolkCountry","Genre_FunkSoulRNB",
            "Genre_Jazz","Genre_Pop","Genre_RapHippop","Genre_Rock","Mood_Acoustic","Mood_Aggressive",
            "Mood_Electronic","Mood_Happy","Mood_Party","Mood_Relaxed","Mood_Sad",
            "Mood_MirexC1","Mood_MirexC2","Mood_MirexC3","Mood_MirexC4","Mood_MirexC5","Atonal","Instrumental","Bright_Timbre"]]
'''

X = X.apply(pd.to_numeric, errors='coerce')
Y = data[["love_class"]]
Y = Y.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
Y.fillna(0, inplace=True)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=222)
    '''
    svc = SVC()
    y_pred = svc.fit(x_train, y_train.values.ravel())
    y_pred = svc.predict(x_test)
    print("*" * 80)
    print('準確率: %.2f' % accuracy_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("TN: %d, FP: %d, FN: %d, TP: %d" %(tn, fp, fn, tp))
    print(classification_report(y_test, y_pred))
    '''
    '''
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}  
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    grid.fit(x_train, y_train.values.ravel()) 
    print(grid.best_params_) 
    '''

    print("*" * 80)
    
    svc = SVC(C=1000, gamma=0.001, kernel='rbf')
    y_pred = svc.fit(x_train, y_train.values.ravel())
    y_pred = svc.predict(x_test)
    print('準確率: %.2f' % accuracy_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("TN: %d, FP: %d, FN: %d, TP: %d" %(tn, fp, fn, tp))
    print(classification_report(y_test, y_pred))

    print("*" * 80)
    
    '''
    merged = pd.concat([x_test, y_test], axis = 1)
    merged.to_csv("tmp.csv", index=False)
    
    merged = pd.DataFrame(y_pred, columns=["pred"])
    merged.to_csv("ans.csv", index=False)
    '''
    
    print("10-fold CV Logistic Regression")
    log_reg = linear_model.LogisticRegression(solver='liblinear', C=0.1, penalty='l1')
    cv_10_results = cross_val_score(log_reg, X, Y.values.ravel(), cv=10, scoring='accuracy')
    print("Accuracy: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(log_reg, X, Y.values.ravel(), cv=10, scoring='precision')
    print("Precision: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(log_reg, X, Y.values.ravel(), cv=10, scoring='recall')
    print("Recall: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(log_reg, X, Y.values.ravel(), cv=10, scoring='f1')
    print("F1 score: %f" % cv_10_results.mean())
    print("*" * 80)

    print("10-fold CV KNN")
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    cv_10_results = cross_val_score(knn, X, Y.values.ravel(), cv=10, scoring='accuracy')
    print("Accuracy: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(knn, X, Y.values.ravel(), cv=10, scoring='precision')
    print("Precision: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(knn, X, Y.values.ravel(), cv=10, scoring='recall')
    print("Recall: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(knn, X, Y.values.ravel(), cv=10, scoring='f1')
    print("F1 score: %f" % cv_10_results.mean())
    print("*" * 80)

    print("10-fold CV SVM")
    cv_10_results = cross_val_score(svc, X, Y.values.ravel(), cv=10, scoring='accuracy')
    print("Accuracy: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(svc, X, Y.values.ravel(), cv=10, scoring='precision')
    print("Precision: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(svc, X, Y.values.ravel(), cv=10, scoring='recall')
    print("Recall: %f" % cv_10_results.mean())
    cv_10_results = cross_val_score(svc, X, Y.values.ravel(), cv=10, scoring='f1')
    print("F1 score: %f" % cv_10_results.mean())
    print("*" * 80)