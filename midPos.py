from facenet_pytorch import MTCNN
import cv2
from PIL import Image,ImageDraw,ImageShow
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
#keep_all option은 얼굴 여러 개 return받음
#still fast without using cuda
mtcnn=MTCNN(keep_all=True,device='cuda:0')
frame=cv2.imread('C:/Users/sorjt/Python/img.jpg')

frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
frame=Image.fromarray(frame)

#call mtcnn(frame) instead,call mtcnn.detect Function
boxes,probs,points=mtcnn.detect(frame,landmarks=True)
#copy from orgin image and paste to new Image variable
img_draw=frame.copy()
#PIL iamge position start upper left from (0,0)
draw=ImageDraw.Draw(img_draw)
img_list=list()
#landmark가 어떤걸 의미하는지는 모르겠지만 같이 얻어서 표시하는거 가능, 해보니 눈코입 표시해주는듯?
for i,(box,point) in enumerate(zip(boxes,points)):
    #imageDraw.rectangle argument : tow point of bounding box [x0,y0,x1,y1] or [(x0,y0),(x1,y1)],outline,fill, width
    box=box.tolist()
    midpos=[(box[0]+box[2])//2,(box[1]+box[3])//2]
    rec_pos=[midpos[0]-128,midpos[1]-128,midpos[0]+128,midpos[1]+128]
    #draw.rectangle(rec_pos,width=2,outline='red')
    draw.rectangle(rec_pos,width=2,outline='red')
    
    #image cropping 
    #img_list.append(img_draw.crop(tuple(rec_pos)))


# for img in img_list:
#     ImageShow.show(img)

