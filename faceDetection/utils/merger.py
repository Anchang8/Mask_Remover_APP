import os
import numpy as np
from PIL import Image,ImageDraw
import argparse
import cv2
def getArgument():
    parser=argparse.ArgumentParser(description='file name')
    parser.add_argument('--name',required=True)
    parser.add_argument('--ext',required=False,default='jpg')

    return parser.parse_args()

def getImage(path,size=256):
    img_list=[]
    path_list=[]

    for img in os.listdir(path):
        file_path=os.path.join(path,img)
        path_list.append(file_path)
    path_list = sorted(path_list)
    for file_path in path_list:
        if os.path.isfile(file_path):
            ext=img.split('.')[1]
            if ext=='jpg' or ext=='png':
                tmp_img=cv2.imread(file_path)
                if tmp_img is None:
                    print('{} fail to open'.format(file_path))
                tmp_img=cv2.resize(tmp_img,(size,size))
                img_list.append(tmp_img)
    return img_list
            
def merge(args):
    pos_path='cropped/before/'+args.name+'/box.txt'
    src_path='src/'+args.name+'.'+args.ext
    cropped_path='cropped/after/'+args.name
    out_dir='output/'+args.name
    f=open(pos_path,'r') 
    
    img=cv2.imread(src_path)
    if img is None:
        print('fail to open image')

    cropped_img=getImage(cropped_path,args.size)
    
    for i,line in enumerate(f.readlines()):
        line=list(map(int,line.split(',')))
        #print(line)
        img[line[1]:line[3],line[0]:line[2]]=cropped_img[i]

    #make dir
    if os.path.isdir(out_dir) is False:
        os.mkdir(out_dir)
    
    dst_path=os.path.join(out_dir,args.name+'.jpg')
    cv2.imwrite(dst_path,img)


    f.close()
    



if __name__=='__main__':
    args=getArgument()
    merge(args)



