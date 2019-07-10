import pandas as pd
import os
import shutil
if __name__=='__main__':
    data=pd.read_csv("D:/project/ShuffleNet/csv/train_result1.csv")
    data_error=data[data['flag']==False]
    #img_list=data_error['name'].tolist()
    data_error['name']=data_error['name'].apply(lambda x:x.split('\\')[-1])
    #data_error['res_int']=data['res_int']
    #data_error['res_index']=data['res_index']
    #data_error['res_cate']=data['res_cate']
    #data_error['label']=data['label']
    data_error=data_error.sort_values('name')
    data_error.to_csv('D:/project/ShuffleNet/csv/train_error.csv')
    print(data_error)

    #print(data_error)
    #print(len(data_error))
   # img_list=img_list[:10]
    #print(img_list)
    '''save_dir="D:/train_error"
    for i in range(len(img_list)):
        img_path=img_list[i]
        img_name=img_path.split('\\')[-1]
        save_path=os.path.join(save_dir,img_name)
        shutil.copy(img_path,save_path)
        print(i,len(img_list),img_path,img_name,save_path)'''
