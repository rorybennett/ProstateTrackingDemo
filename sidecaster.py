#!/usr/bin/env python

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Final

APP_DIR: Final = Path(__file__).resolve().parent
LIB_DIR: Final = APP_DIR / "libraries"
SRC_DIR: Final = APP_DIR / "src"

YOLO_MODEL_PATH: Final = APP_DIR / "models" / "yolo_x_phantom_best.pt"
UNET_MODEL_PATH: Final = APP_DIR / "models" / "unet_phantom_latest.pth"
UNET_TRAINING_PARAMETERS_PATH: Final = UNET_MODEL_PATH.with_name("training_parameters.txt")

LEFT_ARROW_ICON_PATH: Final = SRC_DIR / "move_left.png"
RIGHT_ARROW_ICON_PATH: Final = SRC_DIR / "move_right.png"
OK_ICON_PATH: Final = SRC_DIR / "correct.png"
NO_DETECTION_ICON_PATH: Final = SRC_DIR / "no_detection.png"

PY_TAG: Final = f"python{sys.version_info.major}{sys.version_info.minor}"
PY_LIB_DIR: Final = LIB_DIR / PY_TAG
LIB_SEARCH_DIRS: Final = (PY_LIB_DIR, LIB_DIR)

DLL_DIR_HANDLES = []
LIBCAST_HANDLE = None
SHUTTING_DOWN = False


def find_lib(filename: str) -> Path:
    for lib_dir in LIB_SEARCH_DIRS:
        path = lib_dir / filename
        if path.exists():
            return path

    checked = ", ".join(str(lib_dir / filename) for lib_dir in LIB_SEARCH_DIRS)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {checked}")


def prepare_library_paths() -> None:
    for lib_dir in LIB_SEARCH_DIRS:
        if lib_dir.exists() and str(lib_dir) not in sys.path:
            sys.path.insert(0, str(lib_dir))


def load_platform_libraries() -> None:
    global LIBCAST_HANDLE

    if sys.platform.startswith("win"):
        for lib_dir in LIB_SEARCH_DIRS:
            if lib_dir.exists():
                DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib_dir)))
        ctypes.WinDLL(str(find_lib("cast.dll")))
        return

    if sys.platform.startswith("linux"):
        LIBCAST_HANDLE = ctypes.CDLL(str(find_lib("libcast.so")), ctypes.RTLD_GLOBAL)._handle
        ctypes.cdll.LoadLibrary(str(find_lib("pyclariuscast.so")))


prepare_library_paths()
load_platform_libraries()

import numpy as np
import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender
from PySide6.QtCore import Slot
from PySide6.QtGui import QQuaternion, QVector3D
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
UNET_TRAIN_MEAN: Final = 0.07007993166086769
UNET_TRAIN_STD: Final = 0.15056420456784336
UNET_MASK_ALPHA: Final = 90
MAX_BOUNDARY_POINTS_PER_FRAME: Final = 50
MAX_RECORDED_BOUNDARY_POINTS: Final = 50000


CENTRE_TOLERANCE_FRACTION: Final = 0.03
GUIDANCE_ICON_SIZE_FRACTION: Final = 0.18
GUIDANCE_ICON_MARGIN: Final = 20

ROI_DEFAULT_PERCENT: Final = 50
ROI_MIN_PERCENT: Final = 1
ROI_MAX_PERCENT: Final = 100

MEASUREMENT_BUTTONS: Final = {
    3: ("RL", "Calculate RL"),
    4: ("AP", "Calculate AP"),
    5: ("SI", "Calculate SI"),
}

FREEZE_EVENT_TYPE: Final = QtCore.QEvent.Type(QtCore.QEvent.User)
BUTTON_EVENT_TYPE: Final = QtCore.QEvent.Type(QtCore.QEvent.User + 1)
IMAGE_EVENT_TYPE: Final = QtCore.QEvent.Type(QtCore.QEvent.User + 2)


class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen: bool):
        super().__init__(FREEZE_EVENT_TYPE)
        self.frozen = frozen


class ButtonEvent(QtCore.QEvent):
    def __init__(self, button: int, clicks: int):
        super().__init__(BUTTON_EVENT_TYPE)
        self.button = button
        self.clicks = clicks


class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(IMAGE_EVENT_TYPE)


class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage, float, int, int, float, float, float, float, bool)

    def __init__(self):
        super().__init__()
        self.usimage = QtGui.QImage()
        self.microns_per_pixel = 0.0
        self.scan_width = 0
        self.scan_height = 0
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.has_imu_data = False

    def event(self, evt):
        if SHUTTING_DOWN:
            return True

        event_type = evt.type()
        if event_type == FREEZE_EVENT_TYPE:
            self.freeze.emit(evt.frozen)
            return True
        if event_type == BUTTON_EVENT_TYPE:
            self.button.emit(evt.button, evt.clicks)
            return True
        if event_type == IMAGE_EVENT_TYPE:
            self.image.emit(self.usimage, self.microns_per_pixel, self.scan_width, self.scan_height, self.qw, self.qx, self.qy, self.qz, self.has_imu_data)
            return True

        return super().event(evt)


signaller = Signaller()



class BoundaryWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.view = Qt3DExtras.Qt3DWindow()
        self.container = QtWidgets.QWidget.createWindowContainer(self.view)
        self.root_entity = Qt3DCore.QEntity()
        self.view.setRootEntity(self.root_entity)
        self.setCentralWidget(self.container)
        self.setWindowTitle("Recorded Boundaries")
        self.setupScene()
        self.setupPointCloud()

    def setupScene(self):
        camera = self.view.camera()
        camera.lens().setPerspectiveProjection(45, 16 / 9, 0.1, 10000)
        camera.setPosition(QVector3D(0, -120, 80))
        camera.setViewCenter(QVector3D(0, 0, 0))

        controller = Qt3DExtras.QOrbitCameraController(self.root_entity)
        controller.setLinearSpeed(80)
        controller.setLookSpeed(180)
        controller.setCamera(camera)

        light_entity = Qt3DCore.QEntity(self.root_entity)
        light = Qt3DRender.QPointLight(light_entity)
        light.setIntensity(1.0)
        light_transform = Qt3DCore.QTransform()
        light_transform.setTranslation(QVector3D(0, -80, 120))
        light_entity.addComponent(light)
        light_entity.addComponent(light_transform)

    def setupPointCloud(self):
        self.cloud_entity = Qt3DCore.QEntity(self.root_entity)
        self.geometry = Qt3DCore.QGeometry(self.cloud_entity)
        self.vertex_buffer = Qt3DCore.QBuffer(self.geometry)

        self.position_attribute = Qt3DCore.QAttribute(self.geometry)
        self.position_attribute.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
        self.position_attribute.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.position_attribute.setVertexSize(3)
        self.position_attribute.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.position_attribute.setBuffer(self.vertex_buffer)
        self.position_attribute.setByteStride(3 * 4)
        self.position_attribute.setCount(0)
        self.geometry.addAttribute(self.position_attribute)

        self.renderer = Qt3DRender.QGeometryRenderer(self.cloud_entity)
        self.renderer.setGeometry(self.geometry)
        self.renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.Points)
        self.renderer.setVertexCount(0)

        material = Qt3DExtras.QPhongMaterial(self.cloud_entity)
        material.setDiffuse(QtGui.QColor("orange"))
        material.setAmbient(QtGui.QColor("orange"))

        self.cloud_entity.addComponent(self.renderer)
        self.cloud_entity.addComponent(material)
        self.cloud_entity.setEnabled(False)

    def clearPoints(self):
        self.points = np.empty((0, 3), dtype=np.float32)
        self.vertex_buffer.setData(QtCore.QByteArray())
        self.position_attribute.setCount(0)
        self.renderer.setVertexCount(0)
        self.cloud_entity.setEnabled(False)
        self.setWindowTitle("Recorded Boundaries - 0 points")

    def appendPoints(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.size == 0:
            return

        self.points = points.copy() if self.points.size == 0 else np.vstack((self.points, points))
        if self.points.shape[0] > MAX_RECORDED_BOUNDARY_POINTS:
            self.points = self.points[-MAX_RECORDED_BOUNDARY_POINTS:]

        self.updatePointCloud()

    def updatePointCloud(self):
        self.vertex_buffer.setData(QtCore.QByteArray(self.points.tobytes()))
        self.position_attribute.setCount(self.points.shape[0])
        self.renderer.setVertexCount(self.points.shape[0])
        self.cloud_entity.setEnabled(self.points.shape[0] > 0)
        self.fitCamera(self.points)
        self.setWindowTitle(f"Recorded Boundaries - {self.points.shape[0]} points")

    def fitCamera(self, points):
        if points.size == 0:
            return

        centre = points.mean(axis=0)
        spread = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        distance = max(80.0, spread * 1.8)
        camera = self.view.camera()
        camera.setViewCenter(QVector3D(float(centre[0]), float(centre[1]), float(centre[2])))
        camera.setPosition(QVector3D(float(centre[0]), float(centre[1] - distance), float(centre[2] + distance * 0.6)))


class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast=None, controls_output_size: bool = False):
        super().__init__()
        self.cast = cast
        self.controls_output_size = controls_output_size
        self.image = QtGui.QImage()
        self.setScene(QtWidgets.QGraphicsScene())
        self.setMinimumSize(320, 240)

    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()
        self.viewport().update()

    def saveImage(self, filename: Path):
        if not self.image.isNull():
            self.image.save(str(filename))

    def resizeEvent(self, evt):
        width = evt.size().width()
        height = evt.size().height()

        if self.controls_output_size and self.cast is not None and not SHUTTING_DOWN:
            self.cast.setOutputSize(width, height)

        self.setSceneRect(0, 0, width, height)
        super().resizeEvent(evt)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)


class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        super().__init__(parent)
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
        self.latest_orientation = (1.0, 0.0, 0.0, 0.0)
        self.latest_has_imu_data = False

        self.boundaryWindow = None
        self.boundary_recording_enabled = False
        self.boundary_recorded_frames = 0
        self.boundary_status_message = ""
        self.roi_percent = ROI_DEFAULT_PERCENT
        self.is_shutting_down = False

        self.setWindowTitle("Clarius Cast Dual Display Demo")
        self.setupUi()
        self.connectSignals()
        self.loadModelsAndIcons()
        self.initialiseCast()

    def setupUi(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.ipInput = QtWidgets.QLineEdit("192.168.1.1")
        self.ipInput.setInputMask("000.000.000.000")
        self.portInput = QtWidgets.QLineEdit("5828")
        self.portInput.setInputMask("00000")

        self.connectButton = QtWidgets.QPushButton("Connect")
        self.run = QtWidgets.QPushButton("Run")
        self.quitButton = QtWidgets.QPushButton("Quit")
        self.depthUpButton = QtWidgets.QPushButton("< Depth")
        self.depthDownButton = QtWidgets.QPushButton("> Depth")
        self.gainIncButton = QtWidgets.QPushButton("> Gain")
        self.gainDecButton = QtWidgets.QPushButton("< Gain")
        self.captureImageButton = QtWidgets.QPushButton("Capture Image")
        self.captureCineButton = QtWidgets.QPushButton("Capture Movie")
        self.saveImageButton = QtWidgets.QPushButton("Save Local")
        self.bModeButton = QtWidgets.QPushButton("B Mode")
        self.cfiModeButton = QtWidgets.QPushButton("Color Mode")

        self.setupProcessingControls()
        self.originalView = ImageView(self.cast, controls_output_size=True)
        self.processedView = ImageView()

        central.setLayout(self.buildMainLayout())

    def setupProcessingControls(self):
        self.yoloToggleButton = QtWidgets.QPushButton("YOLO: On")
        self.yoloToggleButton.setCheckable(True)
        self.yoloToggleButton.setChecked(self.yolo_enabled)

        self.unetToggleButton = QtWidgets.QPushButton("UNet: Off")
        self.unetToggleButton.setCheckable(True)
        self.unetToggleButton.setChecked(self.unet_enabled)

        self.toolButtons = [self.yoloToggleButton, self.unetToggleButton]
        for index, (_, label) in MEASUREMENT_BUTTONS.items():
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, idx=index: self.tryToolButton(idx, checked))
            self.toolButtons.append(button)

        self.roiLabel = QtWidgets.QLabel(f"ROI {self.roi_percent}%")
        self.roiLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.roiSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.roiSlider.setRange(ROI_MIN_PERCENT, ROI_MAX_PERCENT)
        self.roiSlider.setValue(self.roi_percent)
        self.roiSlider.setTickInterval(10)
        self.roiSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.roiSlider.setMinimumWidth(180)

        self.unetThresholdLabel = QtWidgets.QLabel(f"UNet Threshold {self.unet_logit_threshold:.2f}")
        self.unetThresholdLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.unetThresholdSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.unetThresholdSlider.setRange(
            int(UNET_LOGIT_THRESHOLD_MIN * UNET_LOGIT_THRESHOLD_SCALE),
            int(UNET_LOGIT_THRESHOLD_MAX * UNET_LOGIT_THRESHOLD_SCALE),
        )
        self.unetThresholdSlider.setValue(int(round(self.unet_logit_threshold * UNET_LOGIT_THRESHOLD_SCALE)))
        self.unetThresholdSlider.setTickInterval(10)
        self.unetThresholdSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.unetThresholdSlider.setMinimumWidth(180)

        self.recordBoundariesButton = QtWidgets.QPushButton("Record Boundaries")
        self.recordBoundariesButton.setCheckable(True)
        self.recordBoundariesButton.setMinimumWidth(180)

    def buildMainLayout(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.buildDisplayLayout())
        layout.addLayout(self.buildInputLayout())
        layout.addLayout(self.buildConnectionLayout())
        layout.addLayout(self.buildParameterLayout())
        layout.addLayout(self.buildCaptureLayout())
        layout.addLayout(self.buildModeLayout())
        return layout

    def buildDisplayLayout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.buildOriginalGroup())
        layout.addWidget(self.buildProcessedGroup())
        return layout

    def buildOriginalGroup(self):
        group = QtWidgets.QGroupBox("Original ultrasound image")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.originalView)
        group.setLayout(layout)
        return group

    def buildProcessedGroup(self):
        group = QtWidgets.QGroupBox("Processed image")
        layout = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.processedView, 1)
        for button in self.toolButtons:
            button.setMinimumWidth(100)
            button_layout.addWidget(button)

        button_layout.addWidget(self.roiLabel)
        button_layout.addWidget(self.roiSlider)
        button_layout.addWidget(self.unetThresholdLabel)
        button_layout.addWidget(self.unetThresholdSlider)
        button_layout.addWidget(self.recordBoundariesButton)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group

    def buildInputLayout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.ipInput)
        layout.addWidget(self.portInput)
        return layout

    def buildConnectionLayout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.connectButton)
        layout.addWidget(self.run)
        layout.addWidget(self.quitButton)
        return layout

    def buildParameterLayout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.depthUpButton)
        layout.addWidget(self.depthDownButton)
        layout.addWidget(self.gainDecButton)
        layout.addWidget(self.gainIncButton)
        return layout

    def buildCaptureLayout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.captureImageButton)
        layout.addWidget(self.captureCineButton)
        layout.addWidget(self.saveImageButton)
        return layout

    def buildModeLayout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.bModeButton)
        layout.addWidget(self.cfiModeButton)
        return layout

    def connectSignals(self):
        self.connectButton.clicked.connect(self.tryConnect)
        self.run.clicked.connect(lambda: self.sendCastCommand(CMD_FREEZE))
        self.quitButton.clicked.connect(self.close)
        self.depthUpButton.clicked.connect(lambda: self.sendCastCommand(CMD_DEPTH_DEC))
        self.depthDownButton.clicked.connect(lambda: self.sendCastCommand(CMD_DEPTH_INC))
        self.gainIncButton.clicked.connect(lambda: self.sendCastCommand(CMD_GAIN_INC))
        self.gainDecButton.clicked.connect(lambda: self.sendCastCommand(CMD_GAIN_DEC))
        self.captureImageButton.clicked.connect(lambda: self.sendCastCommand(CMD_CAPTURE_IMAGE))
        self.captureCineButton.clicked.connect(lambda: self.sendCastCommand(CMD_CAPTURE_CINE))
        self.saveImageButton.clicked.connect(self.trySaveImage)
        self.bModeButton.clicked.connect(lambda: self.sendCastCommand(CMD_B_MODE))
        self.cfiModeButton.clicked.connect(lambda: self.sendCastCommand(CMD_CFI_MODE))
        self.yoloToggleButton.clicked.connect(self.tryToggleYolo)
        self.unetToggleButton.clicked.connect(self.tryToggleUnet)
        self.roiSlider.valueChanged.connect(self.tryRoiChanged)
        self.unetThresholdSlider.valueChanged.connect(self.tryUnetThresholdChanged)
        self.recordBoundariesButton.clicked.connect(self.toggleBoundaryRecording)

        signaller.freeze.connect(self.freeze)
        signaller.button.connect(self.button)
        signaller.image.connect(self.image)

    def loadModelsAndIcons(self):
        self.yolo_model = self.loadYoloModel()
        self.unet_model = self.loadUnetModel()
        self.guidance_icons = self.loadGuidanceIcons()

    def initialiseCast(self):
        path = os.path.expanduser("~/")
        if not self.cast.init(path, 640, 480):
            self.statusBar().showMessage("Failed to initialize")
            return

        loaded = []
        if self.yolo_model is not None:
            loaded.append("YOLO")
        if self.unet_model is not None:
            loaded.append("UNet")

        message = "Initialized"
        if loaded:
            message += " with " + " and ".join(loaded)
        self.statusBar().showMessage(message)

    def sendCastCommand(self, command: int):
        if self.cast.isConnected():
            self.cast.userFunction(command, 0)

    def tryConnect(self):
        if not self.cast.isConnected():
            if self.cast.connect(self.ipInput.text(), int(self.portInput.text()), "research"):
                self.statusBar().showMessage("Connected")
                self.connectButton.setText("Disconnect")
            else:
                self.statusBar().showMessage(f"Failed to connect to {self.ipInput.text()}")
            return

        if self.cast.disconnect():
            self.statusBar().showMessage("Disconnected")
            self.connectButton.setText("Connect")
        else:
            self.statusBar().showMessage("Failed to disconnect")

    def trySaveImage(self):
        self.originalView.saveImage(Path.home() / "Pictures/clarius_original_image.png")
        self.processedView.saveImage(Path.home() / "Pictures/clarius_processed_image.png")
        self.statusBar().showMessage("Saved original and processed images")

    def tryToggleYolo(self, checked: bool):
        self.yolo_enabled = checked
        self.yoloToggleButton.setText("YOLO: On" if checked else "YOLO: Off")

        if checked:
            self.disableUnetToggle()

        if checked and self.yolo_model is None:
            self.statusBar().showMessage("YOLO enabled, but model is not loaded")
        else:
            state = "enabled" if checked else "disabled"
            self.statusBar().showMessage(f"YOLO detection {state}")

    def tryToggleUnet(self, checked: bool):
        self.unet_enabled = checked
        self.unetToggleButton.setText("UNet: On" if checked else "UNet: Off")

        if checked:
            self.disableYoloToggle()

        if checked and self.unet_model is None:
            self.statusBar().showMessage("UNet enabled, but model is not loaded")
        else:
            state = "enabled" if checked else "disabled"
            self.statusBar().showMessage(f"UNet segmentation {state}")

    def disableYoloToggle(self):
        self.yolo_enabled = False
        self.yoloToggleButton.blockSignals(True)
        self.yoloToggleButton.setChecked(False)
        self.yoloToggleButton.blockSignals(False)
        self.yoloToggleButton.setText("YOLO: Off")

    def disableUnetToggle(self):
        self.unet_enabled = False
        self.unetToggleButton.blockSignals(True)
        self.unetToggleButton.setChecked(False)
        self.unetToggleButton.blockSignals(False)
        self.unetToggleButton.setText("UNet: Off")

    def tryToolButton(self, index: int, checked: bool = False):
        if index not in MEASUREMENT_BUTTONS:
            self.statusBar().showMessage(f"Tool button {index} pressed")
            return

        key, _ = MEASUREMENT_BUTTONS[index]
        self.measurements_enabled[key] = checked
        state = "enabled" if checked else "disabled"

        if checked and not (self.yolo_enabled or self.unet_enabled):
            self.statusBar().showMessage(f"{key} measurement enabled, but YOLO and UNet are off")
        else:
            self.statusBar().showMessage(f"{key} measurement {state}")

    def tryRoiChanged(self, value: int):
        self.roi_percent = int(value)
        self.roiLabel.setText(f"ROI {self.roi_percent}%")
        self.refreshProcessedImage()
        self.statusBar().showMessage(f"Processed ROI width set to {self.roi_percent}%")

    def tryUnetThresholdChanged(self, value: int):
        self.unet_logit_threshold = float(value) / UNET_LOGIT_THRESHOLD_SCALE
        self.unetThresholdLabel.setText(f"UNet Threshold {self.unet_logit_threshold:.2f}")
        self.refreshProcessedImage()
        self.statusBar().showMessage(f"UNet logit threshold set to {self.unet_logit_threshold:.2f}")

    def refreshProcessedImage(self):
        if not self.latest_image.isNull():
            self.processedView.updateImage(self.processImageForDisplay(self.latest_image, self.latest_microns_per_pixel))

    def toggleBoundaryRecording(self, checked: bool):
        if checked:
            self.startBoundaryRecording()
        else:
            self.stopBoundaryRecording()

    def startBoundaryRecording(self):
        if self.unet_model is None:
            self.recordBoundariesButton.blockSignals(True)
            self.recordBoundariesButton.setChecked(False)
            self.recordBoundariesButton.blockSignals(False)
            self.statusBar().showMessage("UNet model is not loaded")
            return

        self.boundary_recording_enabled = True
        self.boundary_recorded_frames = 0
        self.boundary_status_message = ""
        self.recordBoundariesButton.setText("Stop Recording Boundaries")
        self.showBoundaryWindow(clear=True)
        self.statusBar().showMessage("Recording boundaries")

    def stopBoundaryRecording(self):
        self.boundary_recording_enabled = False
        self.recordBoundariesButton.blockSignals(True)
        self.recordBoundariesButton.setChecked(False)
        self.recordBoundariesButton.blockSignals(False)
        self.recordBoundariesButton.setText("Record Boundaries")
        self.statusBar().showMessage(f"Stopped boundary recording after {self.boundary_recorded_frames} frames")

    def setBoundaryRecordingStatus(self, message: str):
        if message != self.boundary_status_message:
            self.boundary_status_message = message
            self.statusBar().showMessage(message)

    def recordBoundaryFromMask(self, mask, image_width: int, image_height: int, microns_per_pixel: float):
        if not self.boundary_recording_enabled:
            return
        if not self.latest_has_imu_data:
            self.setBoundaryRecordingStatus("Recording paused: no IMU orientation for this frame")
            return
        if microns_per_pixel <= 0:
            self.setBoundaryRecordingStatus("Recording paused: scale unavailable")
            return

        boundary_pixels = self.getLargestMaskBoundaryPoints(mask)
        if boundary_pixels.size == 0:
            self.setBoundaryRecordingStatus("Recording paused: no UNet mask boundary")
            return

        local_points = self.boundaryPixelsToLocalPoints(boundary_pixels, image_width, image_height, microns_per_pixel)
        points_3d = self.rotateBoundaryPoints(local_points, self.latest_orientation)
        self.appendBoundaryPoints(points_3d)
        self.boundary_recorded_frames += 1
        self.setBoundaryRecordingStatus(f"Recording boundaries: {self.boundary_recorded_frames} frames, {len(points_3d)} points added")

    def getLargestMaskBoundaryPoints(self, mask):
        component = self.getLargestMaskComponent(mask)
        if component is None:
            return np.empty((0, 2), dtype=np.float32)

        padded = np.pad(component, 1, mode="constant", constant_values=False)
        interior = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        boundary = component & ~interior
        ys, xs = np.nonzero(boundary)
        points = np.column_stack((xs, ys)).astype(np.float32)

        if points.shape[0] > MAX_BOUNDARY_POINTS_PER_FRAME:
            indices = np.linspace(0, points.shape[0] - 1, MAX_BOUNDARY_POINTS_PER_FRAME, dtype=np.int32)
            points = points[indices]
        return points

    def getLargestMaskComponent(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return None

        height, width = mask.shape
        labels = np.zeros((height, width), dtype=np.int32)
        coords = np.argwhere(mask)
        label = 0
        best_label = 0
        best_count = 0
        neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        for seed_y, seed_x in coords:
            if labels[seed_y, seed_x] != 0:
                continue

            label += 1
            count = 0
            stack = [(int(seed_y), int(seed_x))]
            labels[seed_y, seed_x] = label

            while stack:
                y, x = stack.pop()
                count += 1
                for dy, dx in neighbours:
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        stack.append((ny, nx))

            if count > best_count:
                best_count = count
                best_label = label

        return labels == best_label

    def boundaryPixelsToLocalPoints(self, boundary_pixels, image_width: int, image_height: int, microns_per_pixel: float):
        scale_mm = float(microns_per_pixel) / 1000.0
        xs = boundary_pixels[:, 0]
        ys = boundary_pixels[:, 1]
        x_mm = (xs - (image_width - 1) / 2.0) * scale_mm
        y_mm = ((image_height - 1) / 2.0 - ys) * scale_mm
        z_mm = np.zeros_like(x_mm)
        return np.column_stack((x_mm, y_mm, z_mm)).astype(np.float32)

    def rotateBoundaryPoints(self, points, orientation):
        qw, qx, qy, qz = orientation
        rotation = QQuaternion(float(qw), float(qx), float(qy), float(qz)).normalized()
        axis_correction = QQuaternion.fromEulerAngles(0, 180, 90)
        corrected_rotation = rotation * axis_correction
        rotated = np.empty_like(points, dtype=np.float32)

        for index, (x, y, z) in enumerate(points):
            vector = corrected_rotation.rotatedVector(QVector3D(float(x), float(y), float(z)))
            rotated[index] = (vector.x(), vector.y(), vector.z())
        return rotated

    def ensureBoundaryWindow(self):
        if self.boundaryWindow is None:
            self.boundaryWindow = BoundaryWindow(self)
            self.boundaryWindow.resize(900, 650)
        return self.boundaryWindow

    def showBoundaryWindow(self, clear: bool = False):
        window = self.ensureBoundaryWindow()

        if clear:
            window.clearPoints()

        window.show()
        window.raise_()
        window.activateWindow()

    def appendBoundaryPoints(self, points):
        window = self.ensureBoundaryWindow()
        if not window.isVisible():
            window.show()
        window.appendPoints(points)

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
            keys = ("training_parameters", "config", "args")
            configs.extend(item for item in (checkpoint.get(key) for key in keys) if isinstance(item, dict))
            configs.append(checkpoint)

        for config in configs:
            self.updateUnetConfigFromDict(config, as_float)

        for path in (UNET_TRAINING_PARAMETERS_PATH, UNET_MODEL_PATH.with_suffix(".txt")):
            if path.exists():
                self.updateUnetConfigFromFile(path, as_float)

    def updateUnetConfigFromDict(self, config: dict, as_float):
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

    def updateUnetConfigFromFile(self, path: Path, as_float):
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
        paths = {
            "left": LEFT_ARROW_ICON_PATH,
            "right": RIGHT_ARROW_ICON_PATH,
            "ok": OK_ICON_PATH,
            "no_detection": NO_DETECTION_ICON_PATH,
        }
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
        return arr[:, : width * 3].reshape((height, width, 3)).copy()

    def qImageToGrayArray(self, img):
        gray_img = img.convertToFormat(QtGui.QImage.Format_Grayscale8)
        width = gray_img.width()
        height = gray_img.height()
        buffer = gray_img.bits()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, gray_img.bytesPerLine()))
        return arr[:, :width].copy()

    def getGuidanceState(self, box_centre_x: float, image_width: int):
        image_centre_x = image_width / 2
        tolerance = image_width * CENTRE_TOLERANCE_FRACTION

        if box_centre_x > image_centre_x + tolerance:
            return "left"
        if box_centre_x < image_centre_x - tolerance:
            return "right"
        return "ok"

    def drawGuidanceIcon(self, painter, output_img, guidance_state: str):
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

    def clampPoint(self, painter, x, y, margin: int = 6):
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
            draw_x = x + (icon_size - scaled_icon.width()) // 2
            draw_y = y + (icon_size - scaled_icon.height()) // 2
            painter.drawPixmap(draw_x, draw_y, scaled_icon)
            return

        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(max(14, icon_size // 5))
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.drawText(QtCore.QRectF(x, y, icon_size, icon_size), QtCore.Qt.AlignCenter, "NO DETECTION")

    def drawMeasurementLine(self, painter, start, end, text: str, text_pos, colour):
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

    def drawYoloMeasurements(self, painter, x1, y1, x2, y2, microns_per_pixel: float):
        if microns_per_pixel <= 0:
            painter.setPen(QtGui.QPen(QtCore.Qt.white))
            painter.drawText(self.clampPoint(painter, x1 + 4, y2 + 22), "Scale unavailable")
            return

        scale_mm = microns_per_pixel / 1000.0
        width_px = max(0.0, x2 - x1)
        height_px = max(0.0, y2 - y1)
        width_mm = width_px * scale_mm
        height_mm = height_px * scale_mm
        hypotenuse_mm = (width_px**2 + height_px**2) ** 0.5 * scale_mm
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

    def drawSegmentationMeasurements(self, painter, geometry, microns_per_pixel: float):
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

    def processUnetImageForDisplay(self, original_img, output_img, microns_per_pixel: float):
        if self.unet_model is None:
            return output_img

        try:
            mask = self.runUnetMask(original_img)
        except Exception as exc:
            self.statusBar().showMessage(f"UNet failed: {exc}")
            return output_img

        painter = QtGui.QPainter(output_img)
        if not np.any(mask):
            if self.boundary_recording_enabled:
                self.setBoundaryRecordingStatus("Recording paused: no UNet mask boundary")
            self.drawNoDetectionIcon(painter, output_img)
            painter.end()
            return output_img

        self.recordBoundaryFromMask(mask, original_img.width(), original_img.height(), microns_per_pixel)
        self.drawSegmentationMask(painter, mask)
        geometry = self.getSegmentationGeometry(mask)
        self.drawSegmentationMeasurements(painter, geometry, microns_per_pixel)
        painter.end()
        return output_img

    def processYoloImageForDisplay(self, original_img, output_img, microns_per_pixel: float):
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
        width = original_img.width()
        height = original_img.height()
        roi_width = max(1, int(round(width * percent / 100.0)))
        x = max(0, (width - roi_width) // 2)
        roi_img = original_img.copy(x, 0, roi_width, height)
        padded_img = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        padded_img.fill(QtCore.Qt.black)

        painter = QtGui.QPainter(padded_img)
        painter.drawImage(x, 0, roi_img)
        painter.end()
        return padded_img

    def processImageForDisplay(self, original_img, microns_per_pixel: float):
        roi_img = self.getRoiImage(original_img)
        output_img = roi_img.copy().convertToFormat(QtGui.QImage.Format_ARGB32)
        if roi_img.isNull():
            return output_img
        if self.unet_enabled or self.boundary_recording_enabled:
            return self.processUnetImageForDisplay(roi_img, output_img, microns_per_pixel)
        if self.yolo_enabled:
            return self.processYoloImageForDisplay(roi_img, output_img, microns_per_pixel)
        return output_img

    @Slot(bool)
    def freeze(self, frozen: bool):
        if frozen:
            self.run.setText("Run")
            self.statusBar().showMessage("Image Stopped")
        else:
            self.run.setText("Freeze")
            self.statusBar().showMessage("Image Running (check firewall settings if no image seen)")

    @Slot(int, int)
    def button(self, button: int, clicks: int):
        self.statusBar().showMessage(f"Button {button} pressed w/ {clicks} clicks")

    @Slot(QtGui.QImage, float, int, int, float, float, float, float, bool)
    def image(self, img, microns_per_pixel: float, scan_width: int, scan_height: int, qw: float, qx: float, qy: float, qz: float, has_imu_data: bool):
        if self.is_shutting_down:
            return

        self.latest_microns_per_pixel = microns_per_pixel
        self.latest_scan_width = scan_width
        self.latest_scan_height = scan_height
        self.latest_image = img.copy()
        self.latest_orientation = (qw, qx, qy, qz)
        self.latest_has_imu_data = has_imu_data
        self.originalView.updateImage(img)
        self.processedView.updateImage(self.processImageForDisplay(img, microns_per_pixel))

    def closeEvent(self, evt):
        self.shutdown()
        evt.accept()

    @Slot()
    def shutdown(self):
        global LIBCAST_HANDLE, SHUTTING_DOWN

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
            if self.boundaryWindow is not None:
                self.boundaryWindow.close()
        except Exception as exc:
            print(f"Boundary window close failed: {exc}", file=sys.stderr)

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

        if sys.platform.startswith("linux") and LIBCAST_HANDLE is not None:
            try:
                ctypes.CDLL("libc.so.6").dlclose(LIBCAST_HANDLE)
                LIBCAST_HANDLE = None
            except Exception as exc:
                print(f"libcast unload failed: {exc}", file=sys.stderr)

        QtWidgets.QApplication.quit()


# Called when a displayable scan-converted ultrasound image is streamed.
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
        signaller.has_imu_data = imu is not None and len(imu) > 0

        if signaller.has_imu_data:
            signaller.qw = float(getattr(imu[0], "qw", 1.0))
            signaller.qx = float(getattr(imu[0], "qx", 0.0))
            signaller.qy = float(getattr(imu[0], "qy", 0.0))
            signaller.qz = float(getattr(imu[0], "qz", 0.0))

        QtCore.QCoreApplication.postEvent(signaller, ImageEvent())


# Called when a raw pre scan-converted image is streamed.
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
