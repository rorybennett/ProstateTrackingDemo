#!/usr/bin/env python

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Final

APP_DIR = Path(__file__).resolve().parent
LIB_DIR = APP_DIR / "libraries"
YOLO_MODEL_PATH = APP_DIR / "models" / "yolo_x_phantom_best.pt"
UNET_MODEL_PATH = APP_DIR / "models" / "unet_phantom_latest.pth"
UNET_TRAINING_PARAMETERS_PATH = UNET_MODEL_PATH.with_name("training_parameters.txt")
SRC_DIR = APP_DIR / "src"
LEFT_ARROW_ICON_PATH = SRC_DIR / "move_left.png"
RIGHT_ARROW_ICON_PATH = SRC_DIR / "move_right.png"
OK_ICON_PATH = SRC_DIR / "correct.png"
NO_DETECTION_ICON_PATH = SRC_DIR / "no_detection.png"
PY_TAG = f"python{sys.version_info.major}{sys.version_info.minor}"
PY_LIB_DIR = LIB_DIR / PY_TAG
LIB_SEARCH_DIRS = [PY_LIB_DIR, LIB_DIR]

dll_dir_handles = []
libcast_handle = None
SHUTTING_DOWN = False


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

try:
    import torch
    import torch.nn.functional as torch_nn_F
    from UNet import UNet
    UNET_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    torch_nn_F = None
    UNet = None
    UNET_IMPORT_ERROR = exc

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
UNET_INPUT_SIZE: Final = 600
UNET_LOGIT_THRESHOLD: Final = 0.50
UNET_LOGIT_THRESHOLD_MIN: Final = 0.00
UNET_LOGIT_THRESHOLD_MAX: Final = 1.00
UNET_LOGIT_THRESHOLD_SCALE: Final = 100
UNET_TRAIN_MEAN: Final = None
UNET_TRAIN_STD: Final = None
UNET_MASK_ALPHA: Final = 90
CENTRE_TOLERANCE_FRACTION: Final = 0.03
GUIDANCE_ICON_SIZE_FRACTION: Final = 0.18
GUIDANCE_ICON_MARGIN: Final = 20
ROI_DEFAULT_PERCENT: Final = 50
ROI_MIN_PERCENT: Final = 1
ROI_MAX_PERCENT: Final = 100
MEASUREMENT_BUTTONS: Final = {3: ("RL", "Calculate RL"), 4: ("AP", "Calculate AP"), 5: ("SI", "Calculate SI")}


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
    image = QtCore.Signal(QtGui.QImage, float, int, int)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()
        self.microns_per_pixel = 0.0
        self.scan_width = 0
        self.scan_height = 0

    def event(self, evt):
        if SHUTTING_DOWN:
            return True
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage, self.microns_per_pixel, self.scan_width, self.scan_height)
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
        if self.controls_output_size and self.cast is not None and not SHUTTING_DOWN:
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
        self.yolo_enabled = True
        self.unet_model = None
        self.unet_device = None
        self.unet_input_size = UNET_INPUT_SIZE
        self.unet_logit_threshold = UNET_LOGIT_THRESHOLD
        self.unet_train_mean = UNET_TRAIN_MEAN
        self.unet_train_std = UNET_TRAIN_STD
        self.unet_enabled = False
        self.guidance_icons = {}
        self.measurements_enabled = {key: False for key, _ in MEASUREMENT_BUTTONS.values()}
        self.latest_scan_width = 0
        self.latest_scan_height = 0
        self.latest_microns_per_pixel = 0.0
        self.latest_image = QtGui.QImage()
        self.roi_percent = ROI_DEFAULT_PERCENT
        self.is_shutting_down = False
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

        def tryToggleYolo(checked):
            self.yolo_enabled = checked
            self.yoloToggleButton.setText("YOLO: On" if checked else "YOLO: Off")
            if checked:
                self.unet_enabled = False
                self.unetToggleButton.blockSignals(True)
                self.unetToggleButton.setChecked(False)
                self.unetToggleButton.blockSignals(False)
                self.unetToggleButton.setText("UNet: Off")
            if checked and self.yolo_model is None:
                self.statusBar().showMessage("YOLO enabled, but model is not loaded")
            else:
                self.statusBar().showMessage(f"YOLO detection {'enabled' if checked else 'disabled'}")

        def tryToggleUnet(checked):
            self.unet_enabled = checked
            self.unetToggleButton.setText("UNet: On" if checked else "UNet: Off")
            if checked:
                self.yolo_enabled = False
                self.yoloToggleButton.blockSignals(True)
                self.yoloToggleButton.setChecked(False)
                self.yoloToggleButton.blockSignals(False)
                self.yoloToggleButton.setText("YOLO: Off")
            if checked and self.unet_model is None:
                self.statusBar().showMessage("UNet enabled, but model is not loaded")
            else:
                self.statusBar().showMessage(f"UNet segmentation {'enabled' if checked else 'disabled'}")

        def tryToolButton(index, checked=False):
            if index in MEASUREMENT_BUTTONS:
                key, _ = MEASUREMENT_BUTTONS[index]
                self.measurements_enabled[key] = checked
                state = "enabled" if checked else "disabled"
                if checked and not (self.yolo_enabled or self.unet_enabled):
                    self.statusBar().showMessage(f"{key} measurement enabled, but YOLO and UNet are off")
                else:
                    self.statusBar().showMessage(f"{key} measurement {state}")
                return
            self.statusBar().showMessage(f"Tool button {index} pressed")

        def tryRoiChanged(value):
            self.roi_percent = int(value)
            self.roiLabel.setText(f"ROI {self.roi_percent}%")
            if not self.latest_image.isNull():
                self.processedView.updateImage(self.processImageForDisplay(self.latest_image, self.latest_microns_per_pixel))
            self.statusBar().showMessage(f"Processed ROI width set to {self.roi_percent}%")

        def tryUnetThresholdChanged(value):
            self.unet_logit_threshold = float(value) / UNET_LOGIT_THRESHOLD_SCALE
            self.unetThresholdLabel.setText(f"UNet Threshold {self.unet_logit_threshold:.2f}")
            if not self.latest_image.isNull():
                self.processedView.updateImage(self.processImageForDisplay(self.latest_image, self.latest_microns_per_pixel))
            self.statusBar().showMessage(f"UNet logit threshold set to {self.unet_logit_threshold:.2f}")

        conn.clicked.connect(tryConnect)
        self.run.clicked.connect(tryFreeze)
        quit.clicked.connect(self.close)
        depthUp.clicked.connect(tryDepthUp)
        depthDown.clicked.connect(tryDepthDown)
        gainInc.clicked.connect(tryGainInc)
        gainDec.clicked.connect(tryGainDec)
        captureImage.clicked.connect(tryCaptureImage)
        captureCine.clicked.connect(tryCaptureCine)
        saveImage.clicked.connect(trySaveImage)
        bMode.clicked.connect(tryBMode)
        cfiMode.clicked.connect(tryCfiMode)

        self.yoloToggleButton = QtWidgets.QPushButton("YOLO: On")
        self.yoloToggleButton.setCheckable(True)
        self.yoloToggleButton.setChecked(self.yolo_enabled)
        self.yoloToggleButton.clicked.connect(tryToggleYolo)

        self.unetToggleButton = QtWidgets.QPushButton("UNet: Off")
        self.unetToggleButton.setCheckable(True)
        self.unetToggleButton.setChecked(self.unet_enabled)
        self.unetToggleButton.clicked.connect(tryToggleUnet)

        self.toolButtons = [self.yoloToggleButton, self.unetToggleButton]
        for index in range(3, 6):
            key, label = MEASUREMENT_BUTTONS[index]
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, idx=index: tryToolButton(idx, checked))
            self.toolButtons.append(button)

        self.roiLabel = QtWidgets.QLabel(f"ROI {self.roi_percent}%")
        self.roiLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.roiSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.roiSlider.setRange(ROI_MIN_PERCENT, ROI_MAX_PERCENT)
        self.roiSlider.setValue(self.roi_percent)
        self.roiSlider.setTickInterval(10)
        self.roiSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.roiSlider.setMinimumWidth(180)
        self.roiSlider.valueChanged.connect(tryRoiChanged)

        self.unetThresholdLabel = QtWidgets.QLabel(f"UNet Threshold {self.unet_logit_threshold:.2f}")
        self.unetThresholdLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.unetThresholdSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.unetThresholdSlider.setRange(int(UNET_LOGIT_THRESHOLD_MIN * UNET_LOGIT_THRESHOLD_SCALE), int(UNET_LOGIT_THRESHOLD_MAX * UNET_LOGIT_THRESHOLD_SCALE))
        self.unetThresholdSlider.setValue(int(round(self.unet_logit_threshold * UNET_LOGIT_THRESHOLD_SCALE)))
        self.unetThresholdSlider.setTickInterval(10)
        self.unetThresholdSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.unetThresholdSlider.setMinimumWidth(180)
        self.unetThresholdSlider.valueChanged.connect(tryUnetThresholdChanged)

        self.originalView = ImageView(cast, controls_output_size=True)
        self.processedView = ImageView()

        originalGroup = QtWidgets.QGroupBox("Original ultrasound image")
        originalLayout = QtWidgets.QVBoxLayout()
        originalLayout.addWidget(self.originalView)
        originalGroup.setLayout(originalLayout)

        processedGroup = QtWidgets.QGroupBox("Processed image")
        processedLayout = QtWidgets.QHBoxLayout()
        processedButtonLayout = QtWidgets.QVBoxLayout()
        processedLayout.addWidget(self.processedView, 1)
        for button in self.toolButtons:
            button.setMinimumWidth(100)
            processedButtonLayout.addWidget(button)
        processedButtonLayout.addWidget(self.roiLabel)
        processedButtonLayout.addWidget(self.roiSlider)
        processedButtonLayout.addWidget(self.unetThresholdLabel)
        processedButtonLayout.addWidget(self.unetThresholdSlider)
        processedButtonLayout.addStretch(1)
        processedLayout.addLayout(processedButtonLayout)
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
        self.unet_model = self.loadUnetModel()
        self.guidance_icons = self.loadGuidanceIcons()

        path = os.path.expanduser("~/")
        if cast.init(path, 640, 480):
            loaded = []
            if self.yolo_model is not None:
                loaded.append("YOLO")
            if self.unet_model is not None:
                loaded.append("UNet")
            self.statusBar().showMessage("Initialized" + (" with " + " and ".join(loaded) if loaded else ""))
        else:
            self.statusBar().showMessage("Failed to initialize")

    def loadYoloModel(self):
        if not YOLO_MODEL_PATH.exists():
            self.statusBar().showMessage(f"YOLO model not found: {YOLO_MODEL_PATH}")
            return None
        try:
            return YOLO(str(YOLO_MODEL_PATH))
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load YOLO model: {exc}")
            return None



    def loadUnetValidationConfig(self, checkpoint=None):
        def as_float(value):
            if value is None:
                return None
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float)):
                return float(value)
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
            return float(match.group(0)) if match else None

        configs = []
        if isinstance(checkpoint, dict):
            configs.extend(item for item in (checkpoint.get("training_parameters"), checkpoint.get("config"), checkpoint.get("args")) if isinstance(item, dict))
            configs.append(checkpoint)

        for config in configs:
            for key in ("image_size", "input_size", "unet_input_size"):
                value = as_float(config.get(key))
                if value:
                    self.unet_input_size = int(value)
                    break
            for key in ("train_mean", "training_mean", "mean"):
                value = as_float(config.get(key))
                if value is not None:
                    self.unet_train_mean = value
                    break
            for key in ("train_std", "training_std", "std"):
                value = as_float(config.get(key))
                if value not in (None, 0):
                    self.unet_train_std = value
                    break

        for path in (UNET_TRAINING_PARAMETERS_PATH, UNET_MODEL_PATH.with_suffix(".txt")):
            if not path.exists():
                continue
            try:
                for line in path.read_text(errors="ignore").splitlines():
                    if ":" not in line:
                        continue
                    key, value_text = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = as_float(value_text)
                    if value is None:
                        continue
                    if key == "image_size":
                        self.unet_input_size = int(value)
                    elif key in {"train_mean", "training_mean", "mean"}:
                        self.unet_train_mean = value
                    elif key in {"train_std", "training_std", "std"} and value != 0:
                        self.unet_train_std = value
            except Exception as exc:
                self.statusBar().showMessage(f"Could not read UNet validation config: {exc}")

    def loadUnetModel(self):
        if torch is None or UNet is None:
            self.statusBar().showMessage(f"UNet unavailable: {UNET_IMPORT_ERROR}")
            return None

        if not UNET_MODEL_PATH.exists():
            self.statusBar().showMessage(f"UNET model not found: {UNET_MODEL_PATH}")
            return None

        try:
            self.unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet().to(self.unet_device)
            checkpoint = torch.load(str(UNET_MODEL_PATH), map_location=self.unet_device)
            self.loadUnetValidationConfig(checkpoint)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load UNet model: {exc}")
            return None

    def loadGuidanceIcons(self):
        paths = {"left": LEFT_ARROW_ICON_PATH, "right": RIGHT_ARROW_ICON_PATH, "ok": OK_ICON_PATH, "no_detection": NO_DETECTION_ICON_PATH}
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

    def qImageToGrayArray(self, img):
        gray_img = img.convertToFormat(QtGui.QImage.Format_Grayscale8)
        width = gray_img.width()
        height = gray_img.height()
        buffer = gray_img.bits()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, gray_img.bytesPerLine()))
        return arr[:, :width].copy()

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


    def clampPoint(self, painter, x, y, margin=6):
        device = painter.device()
        width = device.width() if device is not None else 0
        height = device.height() if device is not None else 0
        x = min(max(float(x), margin), max(margin, width - margin))
        y = min(max(float(y), margin + 14), max(margin + 14, height - margin))
        return QtCore.QPointF(x, y)

    def drawNoDetectionIcon(self, painter, output_img):
        icon = self.guidance_icons.get("no_detection")
        icon_size = max(48, int(output_img.width() * GUIDANCE_ICON_SIZE_FRACTION))
        x = (output_img.width() - icon_size) // 2
        y = output_img.height() - icon_size - GUIDANCE_ICON_MARGIN

        if icon is not None:
            scaled_icon = icon.scaled(icon_size, icon_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(x + (icon_size - scaled_icon.width()) // 2, y + (icon_size - scaled_icon.height()) // 2, scaled_icon)
            return

        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(max(14, icon_size // 5))
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.drawText(QtCore.QRectF(x, y, icon_size, icon_size), QtCore.Qt.AlignCenter, "NO DETECTION")

    def drawMeasurementLine(self, painter, start, end, text, text_pos, colour):
        line_pen = QtGui.QPen(colour)
        line_pen.setWidth(3)
        painter.setPen(line_pen)
        painter.drawLine(QtCore.QPointF(*start), QtCore.QPointF(*end))

        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(14)
        painter.setFont(font)

        text_point = self.clampPoint(painter, *text_pos)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        painter.drawText(text_point + QtCore.QPointF(1, 1), text)
        painter.setPen(QtGui.QPen(colour))
        painter.drawText(text_point, text)

    def drawYoloMeasurements(self, painter, x1, y1, x2, y2, microns_per_pixel):
        if microns_per_pixel <= 0:
            painter.setPen(QtGui.QPen(QtCore.Qt.white))
            painter.drawText(self.clampPoint(painter, x1 + 4, y2 + 22), "Scale unavailable")
            return

        scale_mm = microns_per_pixel / 1000.0
        width_px = max(0.0, x2 - x1)
        height_px = max(0.0, y2 - y1)
        width_mm = width_px * scale_mm
        height_mm = height_px * scale_mm
        hypotenuse_mm = (width_px ** 2 + height_px ** 2) ** 0.5 * scale_mm
        centre_x = (x1 + x2) / 2
        centre_y = (y1 + y2) / 2

        if self.measurements_enabled["RL"]:
            self.drawMeasurementLine(painter, (x1, centre_y), (x2, centre_y), f"RL {width_mm:.1f} mm", (x1 - 110, centre_y + 5), QtGui.QColor("dodgerblue"))
        if self.measurements_enabled["AP"]:
            self.drawMeasurementLine(painter, (centre_x, y1), (centre_x, y2), f"AP {height_mm:.1f} mm", (centre_x - 45, y2 + 24), QtCore.Qt.red)
        if self.measurements_enabled["SI"]:
            self.drawMeasurementLine(painter, (x1, y1), (x2, y2), f"SI {hypotenuse_mm:.1f} mm", (x2 + 8, y2 + 22), QtCore.Qt.green)

    def prepareUnetTensor(self, original_img):
        gray = self.qImageToGrayArray(original_img).astype(np.float32) / 255.0
        if self.unet_train_mean is not None and self.unet_train_std not in (None, 0):
            gray = (gray - float(self.unet_train_mean)) / float(self.unet_train_std)
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.unet_device)
        return torch_nn_F.interpolate(tensor, size=(self.unet_input_size, self.unet_input_size), mode="bilinear", align_corners=False)

    def runUnetMask(self, original_img):
        tensor = self.prepareUnetTensor(original_img)
        with torch.no_grad():
            logits = self.unet_model(tensor)
            mask = logits > self.unet_logit_threshold
            mask = torch_nn_F.interpolate(mask.float(), size=(original_img.height(), original_img.width()), mode="nearest")
        return mask.squeeze().detach().cpu().numpy().astype(bool)

    def drawSegmentationMask(self, painter, mask):
        if mask is None or mask.size == 0:
            return
        height, width = mask.shape
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[mask] = [128, 0, 128, UNET_MASK_ALPHA]
        overlay_img = QtGui.QImage(overlay.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888).copy()
        painter.drawImage(0, 0, overlay_img)

    def getSegmentationGeometry(self, mask):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None

        height, width = mask.shape
        min_x = np.full(height, width, dtype=np.int32)
        max_x = np.full(height, -1, dtype=np.int32)
        np.minimum.at(min_x, ys, xs)
        np.maximum.at(max_x, ys, xs)
        row_widths = max_x - min_x
        rl_row = int(np.argmax(row_widths))

        min_y = np.full(width, height, dtype=np.int32)
        max_y = np.full(width, -1, dtype=np.int32)
        np.minimum.at(min_y, xs, ys)
        np.maximum.at(max_y, xs, ys)
        col_heights = max_y - min_y
        ap_col = int(np.argmax(col_heights))

        top_left_idx = int(np.argmin(xs + ys))
        bottom_right_idx = int(np.argmax(xs + ys))
        return {
            "rl": (float(min_x[rl_row]), float(rl_row), float(max_x[rl_row]), float(rl_row)),
            "ap": (float(ap_col), float(min_y[ap_col]), float(ap_col), float(max_y[ap_col])),
            "si": (float(xs[top_left_idx]), float(ys[top_left_idx]), float(xs[bottom_right_idx]), float(ys[bottom_right_idx])),
        }

    def drawSegmentationMeasurements(self, painter, geometry, microns_per_pixel):
        if geometry is None:
            return
        if microns_per_pixel <= 0:
            painter.setPen(QtGui.QPen(QtCore.Qt.white))
            painter.drawText(self.clampPoint(painter, 10, 24), "Scale unavailable")
            return

        scale_mm = microns_per_pixel / 1000.0
        rl_x1, rl_y1, rl_x2, rl_y2 = geometry["rl"]
        ap_x1, ap_y1, ap_x2, ap_y2 = geometry["ap"]
        si_x1, si_y1, si_x2, si_y2 = geometry["si"]
        rl_mm = abs(rl_x2 - rl_x1) * scale_mm
        ap_mm = abs(ap_y2 - ap_y1) * scale_mm
        si_mm = ((si_x2 - si_x1) ** 2 + (si_y2 - si_y1) ** 2) ** 0.5 * scale_mm

        if self.measurements_enabled["RL"]:
            self.drawMeasurementLine(painter, (rl_x1, rl_y1), (rl_x2, rl_y2), f"RL {rl_mm:.1f} mm", (rl_x1 - 110, rl_y1 + 5), QtGui.QColor("dodgerblue"))
        if self.measurements_enabled["AP"]:
            self.drawMeasurementLine(painter, (ap_x1, ap_y1), (ap_x2, ap_y2), f"AP {ap_mm:.1f} mm", (ap_x2 - 45, ap_y2 + 24), QtCore.Qt.red)
        if self.measurements_enabled["SI"]:
            self.drawMeasurementLine(painter, (si_x1, si_y1), (si_x2, si_y2), f"SI {si_mm:.1f} mm", (si_x2 + 8, si_y2 + 22), QtCore.Qt.green)

    def processUnetImageForDisplay(self, original_img, output_img, microns_per_pixel):
        if self.unet_model is None:
            return output_img

        try:
            mask = self.runUnetMask(original_img)
        except Exception as exc:
            self.statusBar().showMessage(f"UNet failed: {exc}")
            return output_img

        painter = QtGui.QPainter(output_img)
        if not np.any(mask):
            self.drawNoDetectionIcon(painter, output_img)
            painter.end()
            return output_img

        self.drawSegmentationMask(painter, mask)
        geometry = self.getSegmentationGeometry(mask)
        self.drawSegmentationMeasurements(painter, geometry, microns_per_pixel)
        painter.end()
        return output_img

    def processYoloImageForDisplay(self, original_img, output_img, microns_per_pixel):
        if self.yolo_model is None:
            return output_img

        try:
            frame = self.qImageToRgbArray(original_img)
            results = self.yolo_model.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
        except Exception as exc:
            self.statusBar().showMessage(f"YOLO failed: {exc}")
            return output_img

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            painter = QtGui.QPainter(output_img)
            self.drawNoDetectionIcon(painter, output_img)
            painter.end()
            return output_img

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        best_x1, best_y1, best_x2, best_y2 = boxes.xyxy[best_idx].tolist()
        guidance_state = self.getGuidanceState((best_x1 + best_x2) / 2, output_img.width())

        painter = QtGui.QPainter(output_img)
        pen = QtGui.QPen(QtCore.Qt.green)
        pen.setWidth(5)
        painter.setPen(pen)

        conf = float(boxes.conf[best_idx].item())
        cls_id = int(boxes.cls[best_idx].item()) if boxes.cls is not None else None
        label = f"{self.yolo_model.names.get(cls_id, cls_id)} {conf:.2f}" if cls_id is not None else f"{conf:.2f}"
        painter.drawRect(QtCore.QRectF(best_x1, best_y1, best_x2 - best_x1, best_y2 - best_y1))

        label_font = QtGui.QFont()
        label_font.setBold(True)
        label_font.setPointSize(18)
        painter.setFont(label_font)
        label_pos = QtCore.QPointF(best_x1 + 4, max(22, best_y1 - 8))
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        painter.drawText(label_pos + QtCore.QPointF(2, 2), label)
        painter.setPen(pen)
        painter.drawText(label_pos, label)

        self.drawYoloMeasurements(painter, best_x1, best_y1, best_x2, best_y2, microns_per_pixel)
        self.drawGuidanceIcon(painter, output_img, guidance_state)
        painter.end()
        return output_img

    def getRoiImage(self, original_img):
        if original_img.isNull():
            return original_img
        percent = min(max(int(self.roi_percent), ROI_MIN_PERCENT), ROI_MAX_PERCENT)
        roi_width = max(1, int(round(original_img.width() * percent / 100.0)))
        x = max(0, (original_img.width() - roi_width) // 2)
        return original_img.copy(x, 0, roi_width, original_img.height())

    def processImageForDisplay(self, original_img, microns_per_pixel):
        roi_img = self.getRoiImage(original_img)
        output_img = roi_img.copy().convertToFormat(QtGui.QImage.Format_ARGB32)
        if roi_img.isNull():
            return output_img
        if self.unet_enabled:
            return self.processUnetImageForDisplay(roi_img, output_img, microns_per_pixel)
        if self.yolo_enabled:
            return self.processYoloImageForDisplay(roi_img, output_img, microns_per_pixel)
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

    @Slot(QtGui.QImage, float, int, int)
    def image(self, img, microns_per_pixel, scan_width, scan_height):
        if self.is_shutting_down:
            return
        self.latest_microns_per_pixel = microns_per_pixel
        self.latest_scan_width = scan_width
        self.latest_scan_height = scan_height
        self.latest_image = img.copy()
        self.originalView.updateImage(img)
        self.processedView.updateImage(self.processImageForDisplay(img, microns_per_pixel))

    def closeEvent(self, evt):
        self.shutdown()
        evt.accept()

    @Slot()
    def shutdown(self):
        global SHUTTING_DOWN, libcast_handle
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        SHUTTING_DOWN = True

        try:
            signaller.freeze.disconnect(self.freeze)
            signaller.button.disconnect(self.button)
            signaller.image.disconnect(self.image)
        except (RuntimeError, TypeError):
            pass

        try:
            if self.cast is not None and self.cast.isConnected():
                self.cast.disconnect()
        except Exception as exc:
            print(f"Cast disconnect failed: {exc}", file=sys.stderr)

        try:
            if self.cast is not None:
                self.cast.destroy()
        except Exception as exc:
            print(f"Cast destroy failed: {exc}", file=sys.stderr)

        self.cast = None

        if sys.platform.startswith("linux") and libcast_handle is not None:
            try:
                ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
                libcast_handle = None
            except Exception as exc:
                print(f"libcast unload failed: {exc}", file=sys.stderr)

        QtWidgets.QApplication.quit()


# called when a new processed image is streamed
# this is the displayable scan-converted ultrasound image
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    if SHUTTING_DOWN:
        return
    bpp = sz / (width * height)
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    app = QtCore.QCoreApplication.instance()
    if app is not None and not app.closingDown():
        signaller.usimage = img.copy()
        signaller.microns_per_pixel = float(micronsPerPixel)
        signaller.scan_width = int(width)
        signaller.scan_height = int(height)
        QtCore.QCoreApplication.postEvent(signaller, ImageEvent())


# called when a new raw pre scan-converted image is streamed
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    return


def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


def newImuData(imu):
    return


def freezeFn(frozen):
    app = QtCore.QCoreApplication.instance()
    if not SHUTTING_DOWN and app is not None and not app.closingDown():
        QtCore.QCoreApplication.postEvent(signaller, FreezeEvent(frozen))


def buttonsFn(button, clicks):
    app = QtCore.QCoreApplication.instance()
    if not SHUTTING_DOWN and app is not None and not app.closingDown():
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
