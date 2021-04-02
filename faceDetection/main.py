import argparse
import os
from utils import extract
from utils import merger
from utils import remover
from mask import maskRemover
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected')


parser = argparse.ArgumentParser(description='extract faces and merge')
# 소스이미지의 경로와 검출된 얼굴 이미지를 저장할 경로 설정
parser.add_argument('--src', required=True, help='sorce image address')
parser.add_argument('--dst', required=False, default='cropped/before')
parser.add_argument('--all', required=False, default=True, type=bool)
parser.add_argument('--size', required=False, type=int,
                    default=512, help='crop size')
# merge arguments
parser.add_argument('--name', required=False)
parser.add_argument('--ext', required=False, default='jpg')
# remove arguments
parser.add_argument('--rmsrc', required=False, help='sorce image address')
parser.add_argument('--ori', required=False, default=False,
                    type=str2bool, help='remove original image')
parser.add_argument('--crp', required=False, default=False,
                    type=str2bool, help='remove cropped image')
parser.add_argument('--out', required=False, default=False,
                    type=str2bool, help='remove output image')
args = parser.parse_args()

args.name = os.path.basename(args.src).split('.')[0]
args.rmsrc = args.src
# extract faces
ret = extract.main(args)
# time.sleep(5)
if ret is False:
    print('error occur in extract.py or not detected face')

# Mask Remover
else:
    ret = maskRemover.main(args)
    # time.sleep(5)
    print(ret)
    if ret is not True:
        print('error occur in maskRemover.py')
    else:
        ret = merger.merge(args)
        if ret is False:
            print('error occur in merger.py')
        else:
            print('mask removal success')
        # time.sleep(5)
remover.main(args)
