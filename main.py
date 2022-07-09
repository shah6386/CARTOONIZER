import sys
import cv2
import numpy as np
from AnimeGAN import cartoonify
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication


class SaveThread(QThread):
    def __init__(self, frame, loc):
        super().__init__()
        self.frame = frame
        self.loc = loc

    def run(self):
        self.frame = cartoonify(self.frame, "cpu")
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.loc), self.frame)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cpu=False):
        super().__init__()
        self._run_flag = True
        self.cpu = cpu
        self.frame_to_cap = None
        self.to_capture = False
        self.save_thread_pool = []
        self.cam = None

    def run(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while self._run_flag:
            status, frame = self.cam.read()
            if status:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame[120:600, 320:960], cv2.COLOR_BGR2RGB)
                if self.to_capture:
                    self.frame_to_cap = frame.copy()
                    self.to_capture = False
                frame = cv2.resize(frame, (480, 360))
                if not self.cpu:
                    frame = cartoonify(frame, "gpu")
                frame = cv2.resize(frame, (400, 300))
                self.change_pixmap_signal.emit(frame)

        self.cam.release()

    def save(self, file_loc):
        self.save_thread_pool.append(SaveThread(self.frame_to_cap, file_loc))
        self.save_thread_pool[-1].start()

    def stop(self):
        self._run_flag = False
        self.cam.release()
        for thread in self.save_thread_pool:
            thread.wait()
        self.wait()


class MainScreen(QtWidgets.QDialog):
    def __init__(self):
        super(MainScreen, self).__init__()
        loadUi("mainscreen.ui", self)
        self.goliveButton.clicked.connect(self.goto_live_screen)
        self.capturecartoonizeButton.clicked.connect(self.goto_capture_screen_cpu)
        self.uploadimageButton.clicked.connect(self.getfile)
        self.dlg = QtWidgets.QFileDialog()
        self.save_thread = None

    def goto_live_screen(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def goto_capture_screen_cpu(self):
        widget.setCurrentIndex(widget.currentIndex() + 2)

    def getfile(self):
        file_name = self.dlg.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.jpeg *.png)")
        if file_name[0] != '':
            frame = cv2.imread(file_name[0])
            print(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_name = self.dlg.getSaveFileName(self, 'Save file', '', "Image files (*.jpg)")
            self.save_thread = SaveThread(frame, save_name[0])
            self.save_thread.start()

    def closeEvent(self, event):
        self.save_thread.wait()
        event.accept()


class GoLiveScreen(QtWidgets.QDialog):
    def __init__(self, cpu=False):
        super(GoLiveScreen, self).__init__()
        loadUi("golive.ui", self)
        self.return_index = 1
        self.cpu = cpu
        if self.cpu:
            self.return_index = 2
        self.thread = None
        self.back_button.clicked.connect(self.goto_main_screen)
        self.shutter.clicked.connect(self.capture)
        self.dlg = QtWidgets.QFileDialog()
        self.camera_started = False
        self.start.clicked.connect(self.start_stop_camera)

    def goto_main_screen(self):
        if self.camera_started:
            self.thread.stop()
            self.camera_started = False
        widget.setCurrentIndex(widget.currentIndex() - self.return_index)

    def capture(self):
        if self.camera_started:
            self.thread.to_capture = True
            self.thread.save(self.getfile_loc())

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.videoframe.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_qt_format)

    def start_stop_camera(self):
        if not self.camera_started:
            self.thread = VideoThread(self.cpu)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
            self.camera_started = True
        else:
            self.thread.stop()
            self.camera_started = False

    def getfile_loc(self):
        file_name = self.dlg.getSaveFileName(self, 'Save file', '', "Image files (*.jpg)")
        return file_name[0]

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


app = QApplication(sys.argv)
app.setApplicationName("Cartoonizer")
app.setWindowIcon(QIcon('icon.png'))
main_screen = MainScreen()
go_live_screen = GoLiveScreen()
go_live_screen_cpu = GoLiveScreen(cpu=True)
widget = QtWidgets.QStackedWidget()
widget.addWidget(main_screen)
widget.addWidget(go_live_screen)
widget.addWidget(go_live_screen_cpu)
widget.setFixedHeight(480)
widget.setFixedWidth(640)
widget.show()
try:
    sys.exit(app.exec())
except Exception as e:
    print(e)
