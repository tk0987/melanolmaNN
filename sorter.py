import numpy as np
import os, glob
import shutil

# classes
nevi=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/nv/"
vasc=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/vasc/"
mel=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/mel/"
df=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/df/"
bcc=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/bcc/"
akiec=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/akiec/"
bkl=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/bkl/"
# main folder
main=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_2/"
# labels dir
labels=r"/home/geniusz/nn/swiatowidCNN/labels.txt"
data=[]
f=open(labels,"r")
for line in f:
    data.append(line.split("	"))
print(np.shape(data))

def check(name,list: list,file):
    for i in range(0,len(list),1):
        if str(name)==str(list[i][0])+".jpg":
            # print(str(list[i][1]))
            if list[i][1]=="nv\n":
                print(file)
                shutil.copy2(file, nevi)

            if list[i][1]=="mel\n":
                print(file)
                shutil.copy2(file, mel)

            if list[i][1]=="vasc\n":
                print(file)
                shutil.copy2(file, vasc)

            if list[i][1]=="akiec\n":
                print(file)
                shutil.copy2(file, akiec)

            if list[i][1]=="bcc\n":
                print(file)
                shutil.copy2(file, bcc)

            if list[i][1]=="bkl\n":
                print(file)
                shutil.copy2(file, bkl)

            if list[i][1]=="df\n":
                print(file)
                shutil.copy2(file, df)

main_dir=os.chdir(main)
extension="*.jpg"
for file in glob.glob(extension):
    name=file
    # print(name)
    check(name,data,file)
    # if list[i][1]=="nv":
    #     shutil.copy2(file, nevi)

    # if list[i][1]=="mel":
    #     shutil.copy2(file, mel)

    # if list[i][1]=="vasc":
    #     shutil.copy2(file, vasc)

    # if list[i][1]=="akiec":
    #     shutil.copy2(file, akiec)

    # if list[i][1]=="bcc":
    #     shutil.copy2(file, bcc)

    # if list[i][1]=="dr":
    #     shutil.copy2(file, df)
