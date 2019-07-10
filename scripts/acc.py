#结果

import pandas as pd

def func(a):
    for i in range(len(a)):
        a[i]=1 if a[i]==True else 0
        #a[i]=int(a[i])
    return a

def func1(a):
    return int(a[1]+a[4])

if __name__=="__main__":
    res=pd.read_csv('D:/res/result1.9.csv')
    flag=[]
    cat=[]
    dog=[]
    cat_dog=[]
    other=[]
    label=res['multilabel']
    category=res['pred_int']
    for i in range(len(res)):
        if func1(label[i])==func1(category[i]):
            if func1(label[i])==0:
                other.append(1)
            if func1(label[i])==1:
                dog.append(1)
            if func1(label[i])==10:
                cat.append(1)
            if func1(label[i])==11:
                cat_dog.append(1)
            print(i,1)
            flag.append(1)
        else:
            if func1(label[i])==0:
                other.append(0)
            if func1(label[i])==1:
                dog.append(0)
            if func1(label[i])==10:
                cat.append(0)
            if func1(label[i])==11:
                cat_dog.append(0)
            print(i,0)
            flag.append(0)
    print('all:',sum(flag),len(flag),(sum(flag)/len(flag)))
    print('cat:',sum(cat) ,len(cat),(sum(cat)/len(cat)))
    print('dog:',sum(dog),len(dog),(sum(dog)/(len(dog)-34)))
    print('other:',sum(other),len(other),(sum(other)/len(other)))
    print('cat_dog',sum(cat_dog),len(cat_dog),(sum(cat_dog)/len(cat_dog)))
