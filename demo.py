from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform, BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300
import cv2
import torch.utils.data as data
from layers.functions import Detect, PriorBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from utils.nms_wrapper import cpu_nms as nms
from utils.timer import Timer
from collections import OrderedDict

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

classes = ['aeroplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court',
           'ground_track_field', 'harbor', 'bridge', 'vehicle']


def nms_py(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-i', '--img', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB_vgg_NWPU_300.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
else:
    print('Unkown version!')
cfg = VOC_300
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
numclass = 21
img = cv2.imread(args.img)
scale = torch.Tensor([img.shape[1], img.shape[0],
                      img.shape[1], img.shape[0]])
net = build_net('test', 300, numclass)  # initialize detector

transform = BaseTransform(net.size, (123, 117, 104), (2, 0, 1))
with torch.no_grad():
    x = transform(img).unsqueeze(0)
    if args.cuda:
        x = x.cuda()
        scale = scale.cuda()
state_dict = torch.load(args.trained_model)
# create new OrderedDict that does not contain `module.`

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()
if args.cuda:
    net = net.cuda()
    cudnn.benchmark = True
else:
    net = net.cpu()
print('Finished loading model!')
# print(net)
detector = Detect(numclass, 0, cfg)
out = net(x)  # forward pass
boxes, scores = detector.forward(out, priors)
boxes = boxes[0]
scores = scores[0]
boxes *= scale
boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()
# Create figure and axes
# Display the image    
fig, ax = plt.subplots(1)
ax.imshow(img)

# scale each detection back up to the image
result_set = []
for j in range(1, numclass):
    max_ = max(scores[:, j])
    inds = np.where(scores[:, j] > 0.2)[0]  # conf > 0.6
    if inds is None:
        continue
    c_bboxes = boxes[inds]
    c_scores = scores[inds, j]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = nms_py(c_dets, 0.6)
    c_dets = c_dets[keep, :]
    c_bboxes = c_dets[:, :4]
    for bbox in c_bboxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2]) - int(bbox[0]) + 1,
                                 int(bbox[3]) - int(bbox[1]) + 1, linewidth=1, edgecolor='r')
        # Add the patch to the Axes
        ax.add_patch(rect)
        result_set.append(str(rect))
        # plt.show()
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.imwrite("my_test.png", img)


class Picture(QWidget):
    def __init__(self):
        super(Picture, self).__init__()

        self.resize(1280, 720)
        self.setWindowTitle("label显示图片")
        self.label = QLabel(self)
        self.label.setText("    显示图片")

        # self.label.move(160, 160)
        # self.label.setStyleSheet("QLabel{background:white;}"
        #                          "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}")
        img_data = QtGui.QPixmap("my_test.png")
        img_size = img_data.size()
        self.label.resize(img_size.width(), img_size.height())
        print(img_size.width(), img_size.height())
        self.label.setPixmap(img_data)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my = Picture()
    my.show()
    sys.exit(app.exec_())
