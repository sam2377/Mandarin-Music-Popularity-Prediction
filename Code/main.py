import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

input_file = "./Code/Data/trim.csv"
data = pd.read_csv(input_file)

if __name__ == "__main__":

    '''
    X = data[["subscribe_cnt", "Dissonance", 
                 "Genre_Jazz", "Genre_Rock",
                "Mood_Acoustic", "Mood_Happy", "Mood_Relaxed",
                 "Instrumental", "Bright_Timbre"]]
    '''

    #X = data[["subscribe_cnt"]]
    #X = data[["subscribe_cnt"]]

    ''' # All
    X = data[["subscribe_cnt","Replay_Gain","BPM","Dissonance","Pitch_Salience","Gender","Danceability",
                "Genre_Alternative","Genre_Blues","Genre_Electronic","Genre_FolkCountry","Genre_FunkSoulRNB",
                "Genre_Jazz","Genre_Pop","Genre_RapHippop","Genre_Rock","Mood_Acoustic","Mood_Aggressive",
                "Mood_Electronic","Mood_Happy","Mood_Party","Mood_Relaxed","Mood_Sad",
                "Mood_MirexC1","Mood_MirexC2","Mood_MirexC3","Mood_MirexC4","Mood_MirexC5","Atonal","Instrumental","Bright_Timbre"]]
    '''
    '''
    X = data[["subscribe_cnt","Replay_Gain","BPM","Dissonance","Pitch_Salience","Gender","Danceability",
                "Genre_Alternative","Genre_Blues","Genre_Electronic","Genre_FolkCountry","Genre_FunkSoulRNB",
                "Genre_Jazz","Genre_Pop","Genre_RapHippop","Genre_Rock","Mood_Acoustic","Mood_Aggressive",
                "Mood_Electronic","Mood_Happy","Mood_Party","Mood_Relaxed","Mood_Sad",
                "Mood_MirexC1","Mood_MirexC2","Mood_MirexC3","Mood_MirexC4","Mood_MirexC5","Atonal","Instrumental","Bright_Timbre"]]
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

    x_train, x_test, y_train, y_test = train_test_split(X_, Y, test_size = 0.1, random_state=2747)

    #print(data.shape)
    #print(x_train.shape)
    #print(x_test.shape)
    #print(y_train.shape)
    #print(x_train)
    
    linreg = linear_model.LinearRegression()
    linreg.fit(x_train, y_train)
    y_pred = linreg.predict(x_test)

    #print(y_test)
    #print("-" * 80)
    #print(y_pred)
    print("*" * 80)
    print("Linear Regression")
    print("MSE: %f" % metrics.mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("MAE: %f" % metrics.mean_absolute_error(y_test, y_pred))
    print("R^2: %f" % linreg.score(x_test, y_test))

    #print(metrics.r2_score(y_test, y_pred)) 
    #print(linreg.intercept_)
    #print(linreg.coef_)
    #print(y_pred.mean())
    #print(y_test.mean())
    SS_tot = (y_test.mean() - y_test) * (y_test.mean() - y_test)
    SS_res = (y_pred - y_test.values.ravel()) * (y_pred - y_test.values.ravel())
    abs_ = abs(y_test.mean() - y_test)
    #print("SS_res: %f" % SS_res.sum())
    #print("SS_tot: %f" % SS_tot.sum())
    r2 = 1 - SS_res.sum()/SS_tot.sum()
    #print(r2)
    print("*" * 80)
    print("Linear Regression Constant Mean Prediction")
    print("RMSE: %f" % np.sqrt(SS_tot.mean()))
    print("MAE: %f" % abs_.mean())
    print("*" * 80)
    
    print("Support Vector Regression")
    x_train, x_test, y_train, y_test = train_test_split(X_, Y, test_size = 0.1, random_state=1725)
    clf = SVR(kernel='linear', C=100000, gamma=1e-7, cache_size=2000, epsilon=0.6)
    #clf = SVR(kernel='poly', degree=3, C=1000, gamma=0.1, cache_size=2000)
    clf.fit(x_train, y_train.values.ravel())
    y_pred = clf.predict(x_test)
    print("MSE: %f" % metrics.mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("MAE: %f" % metrics.mean_absolute_error(y_test, y_pred))
    print("R^2: %f" % clf.score(x_test, y_test))
    SS_tot = (y_test.mean() - y_test) * (y_test.mean() - y_test)
    SS_res = (y_pred - y_test.values.ravel()) * (y_pred - y_test.values.ravel())
    abs_ = abs(y_test.mean() - y_test)
    #print("SS_res: %f" % SS_res.sum())
    #print("SS_tot: %f" % SS_tot.sum())
    r2 = 1 - SS_res.sum()/SS_tot.sum()
    #print(r2)
    print("*" * 80)
    print("SVR Constant Mean Prediction")
    print("RMSE: %f" % np.sqrt(SS_tot.mean()))
    print("MAE: %f" % abs_.mean())
    print("*" * 80)

    '''    
    parameters = {'kernel': ['linear', 'rbf', 'poly'], 'C':[1.5, 10, 100, 1000, 10000, 100000],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.3,0.4,0.5,0.6]}
    svr = SVR()
    clf = GridSearchCV(svr, parameters)
    clf.fit(x_train, y_train.values.ravel())
    print(clf.best_params_)
    '''

    print("YA")