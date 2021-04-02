from facenet_pytorch import MTCNN
import numpy as np
import argparse
import os
import torch
import cv2
from PIL import Image,ImageDraw
import shutil
'''
architecture

html ------main.py
        |
        ---src-----name1.jpg
        |
        ---cropped---before-------------name1-------name1_1.jpg,name1_2.jpg . . . . .,box.txt
        |           |                |             
        |           |                |
        |           --after          ---name2
        |
        |
        ---output-----name1---name1.jpg
                   |
                   |
                   ---name2---name2.jpg
        

'''
def main(args):
    
    #check cuda 
    device='cpu'
    if torch.cuda.is_available():
        device='cuda'
    #filename ex) test
    file_name=os.path.basename(args.src).split('.')[0]
    dst_dir=os.path.join(args.dst,file_name)

    #make subdirectory for cropped face images
    if os.path.isdir(dst_dir) is False:
        os.mkdir(dst_dir)

    f=open(os.path.join(dst_dir,'box.txt'),'w')

    mtcnn=MTCNN(keep_all=args.all,device=device)
    img=cv2.imread(args.src)
    
    #if fail to load image, load default image 
    if img is None:
        try:
            print('open default image')
            img=cv2.imread('image.jpg')
        except:
            print('fail to open image')
            return False
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)

    # cv2.imshow('opened image',img)
    # cv2.waitKey(-1)
    boxes,probs=mtcnn.detect(img)
    print('{} faces detected'.format(len(boxes)))
    for i,box in enumerate(boxes):
        cropped_name=file_name+'_{}'.format(i)+'.jpg'
        box=box.tolist()

        midpos=[int((box[0]+box[2])//2),int((box[1]+box[3])//2)]
        #print('mid pos : ',midpos)
        rec_pos=[midpos[0]-args.size//2,midpos[1]-args.size//2,midpos[0]+args.size//2,midpos[1]+args.size//2]
        #print('box position : ',rec_pos)
        #cropped_img=img[rec_pos[1]:rec_pos[3],rec_pos[0]:rec_pos[2]]
        frame=img.crop(tuple(rec_pos))
        #print('type of image : ',cropped_img)
        
        frame.save(os.path.join(dst_dir,cropped_name))
        f.writelines(','.join(map(str,rec_pos)))
        f.write('\n')



    f.close()
        
    return True


def getArgument():
    parser=argparse.ArgumentParser(description='face detection')
    parser.add_argument('--src',required=True,help='sorce image address')
    parser.add_argument('--dst',required=False,default='cropped/before')
    parser.add_argument('--all',required=False,default=True,type=bool)
    parser.add_argument('--size',required=False,type=int,default=512)

    return parser.parse_args()


if __name__=='__main__':

    args=getArgument()
    print(main(args))
    

