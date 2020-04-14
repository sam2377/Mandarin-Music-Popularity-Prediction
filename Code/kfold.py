import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

input_file = "./Code/Data/trim1218.csv"
data = pd.read_csv(input_file)

if __name__ == "__main__":

    '''
    X = data[["subscribe_cnt", "Dissonance", 
                 "Genre_Jazz", "Genre_Rock",
                "Mood_Acoustic", "Mood_Happy", "Mood_Relaxed",
                "Instrumental", "Bright_Timbre"]]
    '''
    
    #X = data[["subscribe_cnt", "dislike_cnt"]]
    #X = data[["subscribe_cnt"]]

    ''' # All
    X = data[["subscribe_cnt","Replay_Gain","BPM","Dissonance","Pitch_Salience","Gender","Danceability",
                "Genre_Alternative","Genre_Blues","Genre_Electronic","Genre_FolkCountry","Genre_FunkSoulRNB",
                "Genre_Jazz","Genre_Pop","Genre_RapHippop","Genre_Rock","Mood_Acoustic","Mood_Aggressive",
                "Mood_Electronic","Mood_Happy","Mood_Party","Mood_Relaxed","Mood_Sad",
                "Mood_MirexC1","Mood_MirexC2","Mood_MirexC3","Mood_MirexC4","Mood_MirexC5","Atonal","Instrumental","Bright_Timbre"]]
    '''
    
    '''
    X = data[["subscribe_cnt", "Genre_Alternative", "Genre_Blues", "Mood_Aggressive", "Mood_Electronic", "Mood_Party", "Bright_Timbre"]]
    '''

    X = data[["subscribe_cnt", "Genre_Alternative","Genre_Blues","Genre_Electronic","Genre_FolkCountry","Genre_FunkSoulRNB",
                "Genre_Jazz","Genre_Pop","Genre_RapHippop","Genre_Rock","Mood_Acoustic","Mood_Aggressive",
                "Mood_Electronic","Mood_Happy","Mood_Party","Mood_Relaxed","Mood_Sad",
                "Mood_MirexC1","Mood_MirexC2","Mood_MirexC3","Mood_MirexC4","Mood_MirexC5"]]

    X = X.apply(pd.to_numeric, errors='coerce')
    Y = data[["view_cnt"]]
    Y = Y.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)
    Y.fillna(0, inplace=True)

    X_ = StandardScaler().fit_transform(X)
    linreg = linear_model.LinearRegression()

    print("*" * 80)
    print("10-fold Linear regression")
    cv_10_results = cross_val_score(linreg, X_, Y, cv=10, scoring="neg_mean_squared_error")
    print("MSE: %f" % (abs(cv_10_results.mean())))
    cv_10_results = cross_val_score(linreg, X_, Y, cv=10, scoring="neg_mean_squared_error")
    print("RMSE: %f" % np.sqrt((abs(cv_10_results.mean()))))
    cv_10_results = cross_val_score(linreg, X_, Y, cv=10, scoring="neg_mean_absolute_error")
    print("MAE: %f" % (abs(cv_10_results.mean())))
    cv_10_results = cross_val_score(linreg, X_, Y, cv=10, scoring="r2")
    print("R^2: %f" % ((cv_10_results.mean())))
    
    print("*" * 80)
    print("10-fold Support vector regression")
    clf = SVR(kernel='linear', C=100000, gamma=1e-7, cache_size=2000, epsilon=0.5)
    cv_10_results = cross_val_score(clf, X_, Y.values.ravel(), cv=10, scoring="neg_mean_squared_error")
    print("MSE: %f" % (abs(cv_10_results.mean())))
    cv_10_results = cross_val_score(clf, X_, Y.values.ravel(), cv=10, scoring="neg_mean_squared_error")
    print("RMSE: %f" % np.sqrt((abs(cv_10_results.mean()))))
    cv_10_results = cross_val_score(clf, X_, Y.values.ravel(), cv=10, scoring="neg_mean_absolute_error")
    print("MAE: %f" % (abs(cv_10_results.mean())))
    cv_10_results = cross_val_score(clf, X_, Y.values.ravel(), cv=10, scoring="r2")
    print("R^2: %f" % ((cv_10_results.mean())))
    print("*" * 80)
    #print(cv_10_results)
    #print(y_test)
    #print("-" * 80)
    #print(y_pred)
    '''
    print("MSE: %f" % metrics.mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("MAE: %f" % metrics.mean_absolute_error(y_test, y_pred))
    print("R^2: %f" % linreg.score(x_test, y_test))
    
    '''
    print("YA")