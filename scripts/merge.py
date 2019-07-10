#拼接csv

import pandas as pd

def a():
    res = pd.read_csv('D:/res/res1.csv')
    pred = pd.read_csv('D:/res/res1.9.csv')
    #pred = pd.read_csv("D:/project/ShuffleNet/csv/mobile_result2.csv")
    pred['filename'] = pred['filename'].apply(lambda x: x.split('/')[-1])
    pred['filename'] = pred['filename'].apply(lambda x: x.split('.')[0])
    print(pred[:10])
    df = pd.merge(res, pred, how='left', on='filename')
    res['pred_int'] = df['res_int']
    res['pred_cate']=df['res_cate']
    res.to_csv('D:/res/result1.9.csv', index=False, encoding='utf-8')
if __name__=="__main__":
    a()

    '''
    res=pd.read_csv('D:/res/res3.csv')
    category =res['category']
    count=0
    for i in range(len(category)):
         print(i,category[i][10])
         if int(category[i][10])==0:
                 count+=1
    print(count, len(category))'''