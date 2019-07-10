import pandas as pd
split_rate=0.1

if __name__=="__main__":
    img=pd.read_csv("D:/project/ShuffleNet/csv/img8.csv")
    l=len(img)
    x=int(l*(1-split_rate))
    train=img[:x]
    val=img[x:]
    train.to_csv("D:/project/ShuffleNet/csv/train8.csv", index=False, encoding='utf-8')
    val.to_csv("D:/project/ShuffleNet/csv/val8.csv", index=False, encoding='utf-8')
    print(l)
    print(train[:10])
    print(val[:10])
    print("finish")