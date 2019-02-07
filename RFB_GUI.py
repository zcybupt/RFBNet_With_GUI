import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image


class RFB_GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RFB_GUI, self).__init__()

        self.setWindowTitle("RFB-GUI Demo Program")
        self.resize(1280, 720)
        self.statusBar()
        self.setFocus()

        self.file_item = QtWidgets.QAction('Open image', self)
        self.file_item.setShortcut('Ctrl+O')
        self.file_item.setStatusTip('Open new file')
        self.file_item.triggered.connect(self.select_file)

        self.file = self.menuBar().addMenu('File')
        self.file.addAction(self.file_item)

    def select_file(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Select image', r'./',
                                                          "Image files(*.bmp *.jpg *.pbm *.pgm *.png *.ppm *.xbm *.xpm)"
                                                          ";;All files (*.*)")
        try:
            img = Image.open(file_name[0])
        except Exception as e:
            QtWidgets.QMessageBox.information(self, "Alert", str(e))

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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    rfb_gui = RFB_GUI()
    rfb_gui.show()
    sys.exit(app.exec_())
