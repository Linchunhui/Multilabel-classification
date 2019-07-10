import pandas as pd
import os
import shutil
def a():
    data=pd.read_csv('D:/project/ShuffleNet/csv/train_error.csv')
    dir="D:/dataset/picauto_roadmap/Cat"
    dele_dir="D:/train/Dog"
    list=os.listdir(dir)
    data=data[~data['name'].isin(list)]
    data.to_csv('D:/project/ShuffleNet/csv/train_error1.csv',index=False)
    print(data,len(data))

def b():
    data=pd.read_csv('D:/project/ShuffleNet/csv/train_error1.csv')
    img_list=data['name']
    save_dir="D:/train_error2"
    src_dir="D:/train"
    path1='Cat'
    path2='Dog'
    path3='Other'
    for i in range(len(img_list)):
        img_name=img_list[i]
        #img_name=img_path.split('\\')[-1]
        img_path1=os.path.join(src_dir,path1,img_name)
        img_path2 = os.path.join(src_dir, path2, img_name)
        img_path3 = os.path.join(src_dir, path3, img_name)
        save_path=os.path.join(save_dir,img_name)
        if os.path.exists(img_path1):
            shutil.copy(img_path1,save_path)
            print(i,len(img_list),img_path1,img_name,save_path)
        if os.path.exists(img_path2):
            shutil.copy(img_path2,save_path)
            print(i,len(img_list),img_path2,img_name,save_path)
        if os.path.exists(img_path3):
            shutil.copy(img_path3,save_path)
            print(i,len(img_list),img_path3,img_name,save_path)
if __name__=="__main__":
    b()