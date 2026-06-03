#!/usr/bin/env python

import ctypes
import os
import sys
from pathlib import Path
from typing import Final

APP_DIR = Path(__file__).resolve().parent
LIB_DIR = APP_DIR / "libraries"
MODEL_PATH = APP_DIR / "models" / "yolo_x_phantom_best.pt"
SRC_DIR = APP_DIR / "src"
LEFT_ARROW_ICON_PATH = SRC_DIR / "move_left.png"
RIGHT_ARROW_ICON_PATH = SRC_DIR / "move_right.png"
OK_ICON_PATH = SRC_DIR / "correct.png"
PY_TAG = f"python{sys.version_info.major}{sys.version_info.minor}"
PY_LIB_DIR = LIB_DIR / PY_TAG
LIB_SEARCH_DIRS = [PY_LIB_DIR, LIB_DIR]

dll_dir_handles = []
libcast_handle = None


def find_lib(filename):
    for lib_dir in LIB_SEARCH_DIRS:
        path = lib_dir / filename
        if path.exists():
            return path
    checked = ", ".join(str(lib_dir / filename) for lib_dir in LIB_SEARCH_DIRS)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {checked}")


for lib_dir in LIB_SEARCH_DIRS:
    if lib_dir.exists() and str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

if sys.platform.startswith("win"):
    for lib_dir in LIB_SEARCH_DIRS:
        if lib_dir.exists():
            dll_dir_handles.append(os.add_dll_directory(str(lib_dir)))
    ctypes.WinDLL(str(find_lib("cast.dll")))

elif sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL(str(find_lib("libcast.so")), ctypes.RTLD_GLOBAL)._handle
    ctypes.cdll.LoadLibrary(str(find_lib("pyclariuscast.so")))

import pyclariuscast
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot
from ultralytics import YOLO

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14
YOLO_CONF: Final = 0.25
YOLO_IMGSZ: Final = 640
CENTRE_TOLERANCE_FRACTION: Final = 0.12
GUIDANCE_ICON_SIZE_FRACTION: Final = 0.18
GUIDANCE_ICON_MARGIN: Final = 20


class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen


class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks


class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))


class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True


signaller = Signaller()


class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast=None, controls_output_size=False):
        QtWidgets.QGraphicsView.__init__(self)
        self.cast = cast
        self.controls_output_size = controls_output_size
        self.image = QtGui.QImage()
        self.setScene(QtWidgets.QGraphicsScene())
        self.setMinimumSize(320, 240)

    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()
        self.viewport().update()

    def saveImage(self, filename):
        if not self.image.isNull():
            self.image.save(str(filename))

    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        if self.controls_output_size and self.cast is not None:
            self.cast.setOutputSize(w, h)
        self.setSceneRect(0, 0, w, h)
        super().resizeEvent(evt)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)


class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.cast = cast
        self.yolo_model = None
        self.guidance_icons = {}
        self.setWindowTitle("Clarius Cast Dual Display Demo")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        ip = QtWidgets.QLineEdit("192.168.1.1")
        ip.setInputMask("000.000.000.000")
        port = QtWidgets.QLineEdit("5828")
        port.setInputMask("00000")

        conn = QtWidgets.QPushButton("Connect")
        self.run = QtWidgets.QPushButton("Run")
        quit = QtWidgets.QPushButton("Quit")
        depthUp = QtWidgets.QPushButton("< Depth")
        depthDown = QtWidgets.QPushButton("> Depth")
        gainInc = QtWidgets.QPushButton("> Gain")
        gainDec = QtWidgets.QPushButton("< Gain")
        captureImage = QtWidgets.QPushButton("Capture Image")
        captureCine = QtWidgets.QPushButton("Capture Movie")
        saveImage = QtWidgets.QPushButton("Save Local")
        bMode = QtWidgets.QPushButton("B Mode")
        cfiMode = QtWidgets.QPushButton("Color Mode")

        def tryConnect():
            if not cast.isConnected():
                if cast.connect(ip.text(), int(port.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    conn.setText("Disconnect")
                else:
                    self.statusBar().showMessage(f"Failed to connect to {ip.text()}")
            elif cast.disconnect():
                self.statusBar().showMessage("Disconnected")
                conn.setText("Connect")
            else:
                self.statusBar().showMessage("Failed to disconnect")

        def tryFreeze():
            if cast.isConnected():
                cast.userFunction(CMD_FREEZE, 0)

        def tryDepthUp():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_DEC, 0)

        def tryDepthDown():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_INC, 0)

        def tryGainDec():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_DEC, 0)

        def tryGainInc():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_INC, 0)

        def tryCaptureImage():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_IMAGE, 0)

        def tryCaptureCine():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_CINE, 0)

        def trySaveImage():
            self.originalView.saveImage(Path.home() / "Pictures/clarius_original_image.png")
            self.processedView.saveImage(Path.home() / "Pictures/clarius_processed_image.png")
            self.statusBar().showMessage("Saved original and processed images")

        def tryBMode():
            if cast.isConnected():
                cast.userFunction(CMD_B_MODE, 0)

        def tryCfiMode():
            if cast.isConnected():
                cast.userFunction(CMD_CFI_MODE, 0)

        conn.clicked.connect(tryConnect)
        self.run.clicked.connect(tryFreeze)
        quit.clicked.connect(self.shutdown)
        depthUp.clicked.connect(tryDepthUp)
        depthDown.clicked.connect(tryDepthDown)
        gainInc.clicked.connect(tryGainInc)
        gainDec.clicked.connect(tryGainDec)
        captureImage.clicked.connect(tryCaptureImage)
        captureCine.clicked.connect(tryCaptureCine)
        saveImage.clicked.connect(trySaveImage)
        bMode.clicked.connect(tryBMode)
        cfiMode.clicked.connect(tryCfiMode)

        self.originalView = ImageView(cast, controls_output_size=True)
        self.processedView = ImageView()

        originalGroup = QtWidgets.QGroupBox("Original ultrasound image")
        originalLayout = QtWidgets.QVBoxLayout()
        originalLayout.addWidget(self.originalView)
        originalGroup.setLayout(originalLayout)

        processedGroup = QtWidgets.QGroupBox("Processed image")
        processedLayout = QtWidgets.QVBoxLayout()
        processedLayout.addWidget(self.processedView)
        processedGroup.setLayout(processedLayout)

        displayLayout = QtWidgets.QHBoxLayout()
        displayLayout.addWidget(originalGroup)
        displayLayout.addWidget(processedGroup)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(displayLayout)

        inplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(inplayout)
        inplayout.addWidget(ip)
        inplayout.addWidget(port)

        connlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(connlayout)
        connlayout.addWidget(conn)
        connlayout.addWidget(self.run)
        connlayout.addWidget(quit)
        central.setLayout(layout)

        prmlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(prmlayout)
        prmlayout.addWidget(depthUp)
        prmlayout.addWidget(depthDown)
        prmlayout.addWidget(gainDec)
        prmlayout.addWidget(gainInc)

        caplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(caplayout)
        caplayout.addWidget(captureImage)
        caplayout.addWidget(captureCine)
        caplayout.addWidget(saveImage)

        modelayout = QtWidgets.QHBoxLayout()
        layout.addLayout(modelayout)
        modelayout.addWidget(bMode)
        modelayout.addWidget(cfiMode)

        signaller.freeze.connect(self.freeze)
        signaller.button.connect(self.button)
        signaller.image.connect(self.image)

        self.yolo_model = self.loadYoloModel()
        self.guidance_icons = self.loadGuidanceIcons()

        path = os.path.expanduser("~/")
        if cast.init(path, 640, 480):
            msg = "Initialized"
            if self.yolo_model is not None:
                msg += " with YOLO"
            self.statusBar().showMessage(msg)
        else:
            self.statusBar().showMessage("Failed to initialize")

    def loadYoloModel(self):
        if not MODEL_PATH.exists():
            self.statusBar().showMessage(f"YOLO model not found: {MODEL_PATH}")
            return None
        try:
            return YOLO(str(MODEL_PATH))
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load YOLO model: {exc}")
            return None

    def loadGuidanceIcons(self):
        paths = {"left": LEFT_ARROW_ICON_PATH, "right": RIGHT_ARROW_ICON_PATH, "ok": OK_ICON_PATH}
        icons = {}
        for key, path in paths.items():
            icon = QtGui.QPixmap(str(path))
            if not icon.isNull():
                icons[key] = icon
        return icons

    def qImageToRgbArray(self, img):
        rgb_img = img.convertToFormat(QtGui.QImage.Format_RGB888)
        width = rgb_img.width()
        height = rgb_img.height()
        buffer = rgb_img.bits()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, rgb_img.bytesPerLine()))
        return arr[:, :width * 3].reshape((height, width, 3)).copy()

    def getGuidanceState(self, box_centre_x, image_width):
        image_centre_x = image_width / 2
        tolerance = image_width * CENTRE_TOLERANCE_FRACTION
        if box_centre_x > image_centre_x + tolerance:
            return "left"
        if box_centre_x < image_centre_x - tolerance:
            return "right"
        return "ok"

    def drawGuidanceIcon(self, painter, output_img, guidance_state):
        icon = self.guidance_icons.get(guidance_state)
        icon_size = max(48, int(output_img.width() * GUIDANCE_ICON_SIZE_FRACTION))
        y = output_img.height() - icon_size - GUIDANCE_ICON_MARGIN

        if guidance_state == "right":
            x = GUIDANCE_ICON_MARGIN
        elif guidance_state == "left":
            x = output_img.width() - icon_size - GUIDANCE_ICON_MARGIN
        else:
            x = (output_img.width() - icon_size) // 2

        if icon is not None:
            scaled_icon = icon.scaled(icon_size, icon_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            draw_x = x + (icon_size - scaled_icon.width()) // 2
            draw_y = y + (icon_size - scaled_icon.height()) // 2
            painter.drawPixmap(draw_x, draw_y, scaled_icon)
            return

        fallback_text = {"left": "←", "right": "→", "ok": "OK"}[guidance_state]
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(max(20, icon_size // 3))
        painter.setFont(font)
        painter.drawText(QtCore.QRectF(x, y, icon_size, icon_size), QtCore.Qt.AlignCenter, fallback_text)

    def processImageForDisplay(self, original_img):
        output_img = original_img.copy().convertToFormat(QtGui.QImage.Format_ARGB32)
        if self.yolo_model is None or original_img.isNull():
            return output_img

        try:
            frame = self.qImageToRgbArray(original_img)
            results = self.yolo_model.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
        except Exception as exc:
            self.statusBar().showMessage(f"YOLO failed: {exc}")
            return output_img

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return output_img

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx].item())
        cls_id = int(boxes.cls[best_idx].item()) if boxes.cls is not None else None
        label = f"{self.yolo_model.names.get(cls_id, cls_id)} {conf:.2f}" if cls_id is not None else f"{conf:.2f}"

        box_centre_x = (x1 + x2) / 2
        guidance_state = self.getGuidanceState(box_centre_x, output_img.width())

        painter = QtGui.QPainter(output_img)
        pen = QtGui.QPen(QtCore.Qt.green)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
        painter.drawText(QtCore.QPointF(x1 + 4, max(16, y1 - 6)), label)
        self.drawGuidanceIcon(painter, output_img, guidance_state)
        painter.end()

        return output_img

    @Slot(bool)
    def freeze(self, frozen):
        if frozen:
            self.run.setText("Run")
            self.statusBar().showMessage("Image Stopped")
        else:
            self.run.setText("Freeze")
            self.statusBar().showMessage("Image Running (check firewall settings if no image seen)")

    @Slot(int, int)
    def button(self, btn, clicks):
        self.statusBar().showMessage(f"Button {btn} pressed w/ {clicks} clicks")

    @Slot(QtGui.QImage)
    def image(self, img):
        self.originalView.updateImage(img)
        self.processedView.updateImage(self.processImageForDisplay(img))

    @Slot()
    def shutdown(self):
        if sys.platform.startswith("linux") and libcast_handle is not None:
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)

        self.cast.destroy()
        QtWidgets.QApplication.quit()


# called when a new processed image is streamed
# this is the displayable scan-converted ultrasound image
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    bpp = sz / (width * height)
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    signaller.usimage = img.copy()
    QtCore.QCoreApplication.postEvent(signaller, ImageEvent())


# called when a new raw pre scan-converted image is streamed
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    return


def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


def newImuData(imu):
    return


def freezeFn(frozen):
    QtCore.QCoreApplication.postEvent(signaller, FreezeEvent(frozen))


def buttonsFn(button, clicks):
    QtCore.QCoreApplication.postEvent(signaller, ButtonEvent(button, clicks))


def main():
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, newImuData, freezeFn, buttonsFn)
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWidget(cast)
    widget.resize(1200, 600)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
