import torch
from PIL import Image
import socket
import argparse
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import os
from data import VOCroot, v2, v1, AnnotationTransform, VOCDetection, detection_collate, base_transform
from modules import MultiBoxLoss
from ssd import build_ssd
from time import sleep
import json
import argparse




parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
args = parser.parse_args()

input_dir = 'webcam/inputs'
input_ext = '.jpg'
output_dir = 'webcam/outputs'

args.trained_model = 'weights/ssd_300_VOC0712.pth'
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()
transform = base_transform(net.size,(104,117,123))




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
    preds = dict()
    boxes = list()
    scores = list()
    labels = list()
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labelmap[i-1]
            pts = (detections[0,i,j,1:]*scale).cpu().numpy().tolist()
            boxes.append(pts), scores.append(score), lables.append(label_name)
            j+=1
    preds['boxes'] = coords, preds['scores'] = scores, preds['labels'] = labels
    return preds
    # return (pts[0], pts[1], pts[2], pts[3], score, label_name)


#
# local function strip_ext(path)
#   local ext = paths.extname(path)
#   return string.sub(path, 1, -#ext-2)
# end
#
# local function sleep(sec)
#   socket.select(nil, nil, sec)
# end

# -- Load the checkpoint
# print('loading checkpoint from ' .. opt.checkpoint)
# local checkpoint = torch.load(opt.checkpoint)
# local model = checkpoint.model


while true:
    for filename in os.listdir(input_dir):
        if filename.endswith(input_ext):
            in_path = os.path.join(input_dir, filename)
            base, ext = os.path.splitext(filename)
            out_path = output_dir + base + '.json'
            print('Running model on image ' + in_path)


            img = img = cv2.imread(in_path).astype(np.float32)
            # img = Image.open(in_path).('RGB')
            output = predict(img)

            #   local output_struct = {
            #     boxes = boxes_xywh:float():totable(),
            #     captions = captions,
            #     height = ori_H,
            #     width = ori_W,
            #   }

            os.remove(in_path)
            with open(out_path, 'w') as fp:
                json.dump(output, fp)
        sleep(0.05)
