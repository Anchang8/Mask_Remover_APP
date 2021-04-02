import shutil
import os
import argparse
def removeAllFile(path):
    
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            
        else:
            os.remove(path)
            

def getArgument():
    parser=argparse.ArgumentParser(description='face detection')
    parser.add_argument('--src',required=True,help='sorce image address')
    parser.add_argument('--ori',required=False,default=True,type=bool,help='remove original image')
    parser.add_argument('--crp',required=False,default=True,type=bool,help='remove cropped image')
    parser.add_argument('--out',required=False,default=True,type=bool,help='remove output image')
    return parser.parse_args()

def main(args):
    
    file_name=os.path.basename(args.src).split('.')[0]
    crop_dir=['cropped/before','cropped/after']
    out_dir='output'
    if args.ori is True:
        removeAllFile(args.src)
        print('remove src image')
    
    if args.crp is True:
        
        for path in crop_dir:
            path=os.path.join(path,file_name)
            removeAllFile(path)
        print('remove cropped image')

    if args.out is True:
        path=os.path.join(out_dir,file_name)
        removeAllFile(path)
        print('remove output image')

    

