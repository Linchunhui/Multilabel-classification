import os
if __name__=="__main__":
    dir="D:/train_error2/error/all"
    dele_dir="D:/train"
    list=os.listdir(dir)
    path1 = 'Cat'
    path2 = 'Dog'
    path3 = 'Other'
    #print(list)
    for f in list:
        img_path1=os.path.join(dele_dir,path1,f)
        img_path2 = os.path.join(dele_dir, path2, f)
        img_path3 = os.path.join(dele_dir, path3, f)
        #img_path1 = os.path.join(src_dir, path1, img_name)
        #img_path2 = os.path.join(src_dir, path2, img_name)
        #img_path3 = os.path.join(src_dir, path3, img_name)
        #print(img_path1)
        if os.path.exists(img_path1):
            os.remove(img_path1)
            print(img_path1)
        if os.path.exists(img_path2):
            os.remove(img_path2)
            print(img_path2)
        if os.path.exists(img_path3):
            os.remove(img_path3)
            print(img_path3)
        #os.remove(img_path)
        #print(img_path)