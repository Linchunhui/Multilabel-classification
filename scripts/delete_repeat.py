import os
from hashlib import md5

def getmd5(filename):
    file_txt = open(filename, 'rb').read()
    #m = md5.new(file_txt)
    #return m.hexdigest()
    return md5(file_txt).hexdigest
def main():
    path = 'D:/train/Cat/'
    all_size = {}
    total_file = 0
    total_delete = 0
    i=0
    for file in os.listdir(path):
        i+=1
        print(i,file)
        total_file += 1
        real_path = os.path.join(path, file)
        if os.path.isfile(real_path) == True:
            size = os.stat(real_path).st_size
            name_and_md5 = [real_path, '']
            if size in all_size.keys():
                new_md5 = getmd5(real_path)
                if all_size[size][1] == '':
                    all_size[size][1] = getmd5(all_size[size][0])
                if new_md5 in all_size[size]:
                    os.remove(file)
                    print('删除', file)
                    total_delete += 1
                else:
                    all_size[size].append(new_md5)
            else:
                all_size[size] = name_and_md5
    print('文件个数：', total_file)
    print('删除个数：', total_delete)
if __name__ == '__main__':
    main()