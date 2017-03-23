from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from PIL import Image
import sys
import os
from data import AnnotationTransform, VOCDetection, base_transform
from timeit import default_timer as timer
import argparse
import numpy as np
from ssd import build_ssd
import cv2

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
args = parser.parse_args()

args.trained_model = 'weights/ssd_300_VOC0712.pth'
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()
transform = base_transform(net.size,(104,117,123))

# def test_net(save_folder, net, cuda, valset, transform, top_k, thresh):
#
#     # dump predictions and assoc. ground truth to text file for now
#     filename = save_folder+'test1.txt'
#     num_images = len(valset)
#     for i in range(num_images):
#         print('Testing image {:d}/{:d}....'.format(i+1,num_images))
#         img = valset.pull_image(i)
#         img_id, annotation = valset.pull_anno(i)
#         x = Variable(transform(img).unsqueeze_(0))
#
#         with open(filename, mode='a') as f:
#             f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
#             for box in annotation:
#                 f.write('label: '+' || '.join(str(b) for b in box)+'\n')
#         if cuda:
#             x = x.cuda()
#
#         y = net(x)      # forward pass
#         detections = y.data
#         # scale each detection back up to the image
#         scale = torch.Tensor([img.size[0],img.size[1],img.size[0],img.size[1]])
#         pred_num = 0
#         for i in range(detections.size(1)):
#             j = 0
#             while detections[0,i,j,0] >= 0.6:
#                 if pred_num == 0:
#                     with open(filename, mode='a') as f:
#                         f.write('PREDICTIONS: '+'\n')
#                 score = detections[0,i,j,0]
#                 label_name = labelmap[i-1]
#                 pt = (detections[0,i,j,1:]*scale).cpu().numpy()
#                 coords = (pt[0], pt[1], pt[2], pt[3])
#                 pred_num+=1
#                 with open(filename, mode='a') as f:
#                     f.write(str(pred_num)+' label: '+label_name+' score: ' \
#                             +str(score) +' '+' || '.join(str(c) for c in coords)+'\n')
#                 j+=1



# if __name__ == '__main__':
#     # load net
#     net = build_ssd('test', 300, 21)    # initialize SSD
#     net.load_state_dict(torch.load(args.trained_model))
#     net.eval()
#     print('Finished loading model!')
#     # load data
#     valset = VOCDetection(VOCroot, 'val', None, AnnotationTransform())
#     if args.cuda:
#         net = net.cuda()
#         cudnn.benchmark = True
#     # evaluation
#     test_net(args.save_folder, net, args.cuda, valset, base_transform(net.size,(104,117,123)), args.top_k, thresh=args.confidence_threshold)



def predict(frame):
    #res = cv2.resize(img,(0.5*width, 0.5*height), interpolation = cv2.INTER_CUBIC)
    height = frame.shape[0]
    width = frame.shape[1]
    im = Image.fromarray(frame)
    x = Variable(transform(img).unsqueeze_(0))
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width,height,width,height])
    pred_num = 0
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labelmap[i-1]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            j+=1
    return (pt[0], pt[1], pt[2], pt[3],label_name)



video_capture = cv2.VideoCapture(0)
# anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    res = predict(frame)
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (res[0], res[1]), (res[2]-res[0]+1, res[3]-res[1]+1), color, 2)
    cv2.putText(frame, res[4], (res[0], res[1]), font, 4,(255,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
