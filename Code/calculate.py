import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./Code/Data/output1218.csv")
    df.insert(5, "love_ratio", (df["like_cnt"]*100/(df["like_cnt"] + df["dislike_cnt"])))
    new_data = pd.Series([])
    for i in range(len(df)):
        if df["view_cnt"][i] > 160000 :
            new_data[i] = 1
        else :
            new_data[i] = 0
    df.insert(5, "class", new_data)
    #print(df["love_ratio"])
    #print(df.info())
    df.to_csv("./Code/Data/tmp.csv", index=False)

