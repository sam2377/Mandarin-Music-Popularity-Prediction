import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./Code/Data/trim1218.csv")
    #plt.hist(df["view_cnt"], bins=1000) 
    plt.plot(df["subscribe_cnt"], df["view_cnt"], "o")
    plt.show()
    #print(df['view_cnt'].mean())
    #data = df[["love_ratio", "subscribe_cnt"]]
    #correlation = df.corr(method='pearson')
    #correlation.to_csv("trim_tmp.csv", index=False)
    print("YA")