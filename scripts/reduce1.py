import pandas as pd
import shutil, os
from random import shuffle
def copy_cat():
    data = pd.read_csv(r"D:\project\ShuffleNet\csv\train_complex\cat11.csv")
    data = data.loc[data["pred1"] > 0.8]
    data = data.sort_index(by=["pred1"], ascending=True)
    #print(data)
    img_list = list(data["filename"])
    print(img_list)
    img_list = img_list[:30000]
    root = "D:/train3/Cat"
    for i in range(len(img_list)):
        img_path = img_list[i]
        img_name = img_path.split("\\")[-1]
        copy_path = os.path.join(root, img_name)
        print(img_path, copy_path)
        shutil.copy(img_path, copy_path)
    print("finish")

def copy_dog():
    data = pd.read_csv(r"D:\project\ShuffleNet\csv\train_complex\dog11.csv")
    data = data.loc[data["pred1"] > 0.8]
    data = data.sort_index(by=["pred1"], ascending=True)
    #print(data)
    img_list = list(data["filename"])
    print(img_list)
    img_list = img_list[:30000]
    root = "D:/train3/Dog"
    for i in range(len(img_list)):
        img_path = img_list[i]
        img_name = img_path.split("\\")[-1]
        copy_path = os.path.join(root, img_name)
        print(img_path, copy_path,i)
        shutil.copy(img_path, copy_path)
    print("finish")

def copy_other():
    data_dir = "D:/train/Other"
    img_list = os.listdir(data_dir)
    shuffle(img_list)
    img_list = img_list[:30000]
    root = "D:/train3/Other"
    for i in range(len(img_list)):
        img_path = os.path.join(data_dir, img_list[i])
        copy_path = os.path.join(root, img_list[i])
        print(img_path, copy_path, i)
        shutil.copy(img_path,copy_path)
    print("finish")

if __name__=="__main__":
    #copy_cat()
    copy_dog()
    #copy_other()