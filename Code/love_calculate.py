import pandas as pd

if __name__ == "__main__":
    
    df = pd.read_csv("./Code/Data/trim1218.csv")
    new_data = pd.Series([])
    for i in range(len(df)):
        if df["love_ratio"][i] > 97.857 :
            new_data[i] = 1
        else :
            new_data[i] = 0
    df.insert(6, "love_class", new_data)
    #print(df["love_ratio"])
    #print(df.info())
    df.to_csv("./Code/Data/final.csv", index=False)

