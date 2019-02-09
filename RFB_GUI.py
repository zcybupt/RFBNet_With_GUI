from __future__ import print_function
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import BaseTransform, VOC_300, VOC_512
import cv2
from layers.functions import Detect, PriorBox
import matplotlib.patches as patches
from collections import OrderedDict
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from models.RFB_Net_vgg import build_net
import time

classes = ['aeroplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court',
           'ground_track_field', 'harbor', 'bridge', 'vehicle']


class RFB_GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RFB_GUI, self).__init__()

        self.MyMessageBox(self)
        self.setWindowTitle("RFB-GUI Demo Program")
        self.resize(1280, 900)
        self.setFocus()

        self.file_item = QtWidgets.QAction('Open image', self)
        self.file_item.setShortcut('Ctrl+O')
        self.file_item.triggered.connect(self.select_file)

        self.label = self.DragLabel("Please drag image here\nor\nPress Ctrl+O to select", self)
        self.label.addAction(self.file_item)
        self.setCentralWidget(self.label)

        self.priorbox = PriorBox(self.cfg)
        self.cuda = True
        self.numclass = 21
        self.net = build_net('test', self.input_size, self.numclass)  # initialize detector
        state_dict = torch.load(self.trained_model)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True
        else:
            self.net = self.net.cpu()
        print('Finished loading model!')

    def select_file(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select image',
                                                          r'/home/zcy/data/NWPU_VHR-10_dataset/positive_image_set',
                                                          "Image files(*.bmp *.jpg *.pbm *.pgm *.png *.ppm *.xbm *.xpm)"
                                                          ";;All files (*.*)")
        # try:
        self.detect(file_path[0], self)
        # except Exception as e:
        #     QtWidgets.QMessageBox.information(self, "Alert", str(e))

    def detect(self, file_name, object):
        print(file_name)
        start_time = time.time()
        img = cv2.imread(file_name.strip())
        if img is None:
            QtWidgets.QMessageBox.information(self, "Alert", "Please select images")
            return
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        detector = Detect(object.numclass, 0, object.cfg)
        transform = BaseTransform(object.net.size, (123, 117, 104), (2, 0, 1))
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if object.cuda:
                x = x.cuda()
                scale = scale.cuda()
        out = object.net(x)
        with torch.no_grad():
            priors = object.priorbox.forward()
            if object.cuda:
                priors = priors.cuda()
        boxes, scores = detector.forward(out, priors)
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        result_set = []
        for j in range(1, object.numclass):
            max_ = max(scores[:, j])
            inds = np.where(scores[:, j] > 0.2)[0]  # conf > 0.6
            if inds is None:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = object.nms_py(c_dets, 0.6)
            c_dets = c_dets[keep, :]
            c_bboxes = c_dets[:, :4]
            for bbox in c_bboxes:
                # Create a Rectangle patch
                rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2]) - int(bbox[0]) + 1,
                                         int(bbox[3]) - int(bbox[1]) + 1, linewidth=1, edgecolor='r')
                result_set.append(str(rect))
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.imwrite("my_test.png", img)

        end_time = time.time()
        print(end_time - start_time)
        img_data = QtGui.QPixmap("my_test.png")
        height = object.height()
        width = object.height() / img_data.height() * img_data.width()
        img_data = img_data.scaled(width, height)
        object.label.resize(width, height)
        object.label.setPixmap(img_data)
        self.setFocus()

    def nms_py(self, dets, thresh):
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

    class MyMessageBox(QMessageBox):
        def __init__(self, parent):
            super().__init__()
            self.setWindowTitle('Input Size')
            self.setText("Please choose the input image size ")
            self.setFont(QtGui.QFont("Ubuntu Mono", 14))

            _300_button = self.addButton(self.tr('300 x 300'), QMessageBox.ActionRole)
            _512_button = self.addButton(self.tr('512 x 512'), QMessageBox.ActionRole)
            cancel_button = self.addButton(' Cancel ', QMessageBox.ActionRole)
            self.exec_()
            button = self.clickedButton()
            if button == _300_button:
                parent.input_size = 300
                parent.cfg = VOC_300
                parent.trained_model = 'weights/RFB_vgg_NWPU_300.pth'
            elif button == _512_button:
                parent.input_size = 512
                parent.cfg = VOC_512
                parent.trained_model = 'weights/RFB_vgg_NWPU_512.pth'
            elif button == cancel_button:
                sys.exit()

    class DragLabel(QLabel):
        def __init__(self, text, parent):
            super().__init__(text, parent)
            self.parent = parent
            self.setAcceptDrops(True)
            self.setFont(QtGui.QFont("Ubuntu Mono", 30))
            self.setAlignment(Qt.AlignCenter)

        def dragEnterEvent(self, QDragEnterEvent):
            QDragEnterEvent.accept()

        def dropEvent(self, QDropEvent):
            self.parent.detect(QDropEvent.mimeData().text()[7:], self.parent)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    rfb_gui = RFB_GUI()
    rfb_gui.show()
    sys.exit(app.exec_())
