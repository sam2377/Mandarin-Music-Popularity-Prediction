import pandas as pd

if __name__ == "__main__":
    print("YA")
    a = pd.read_csv("./Code/Data/data1218.csv")
    b = pd.read_csv("./Code/Data/audio.csv")
    merged = pd.concat([a, b], axis = 1)
    merged.to_csv("output1215.csv", index=False)
    pass