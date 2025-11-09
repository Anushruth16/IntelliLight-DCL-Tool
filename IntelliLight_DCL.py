# -*- coding: utf-8 -*-
"""
@author: Anushruthpal Keshavathi Jayapal
"""

import traceback, sys
from PyQt6.QtCore import QSize, Qt, QRunnable, pyqtSlot, QThreadPool, QObject, pyqtSignal, QByteArray, QRectF
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QMainWindow, QHBoxLayout, QVBoxLayout,
    QCheckBox, QLabel, QGroupBox, QStatusBar, QSplitter, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter
import socket, struct
import pandas as pd
import pyqtgraph as pg
import time
import os
import paramiko
from camera_detection_test import YOLOCamera
import cv2
from collections import deque
from datetime import datetime
import numpy as np
import joblib


# Logging

log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
if not os.path.exists("logs"):
    os.makedirs("logs")
log_path = os.path.join("logs", log_filename)
sys.stdout = open(log_path, 'w')
sys.stderr = sys.stdout
print(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


class RedPitayaSensor:
    def __init__(self):
        self.size_of_raw_adc = 25000
        self.buffer_size = (self.size_of_raw_adc + 17) * 4
        self.msg_from_client = "-i 1"
        self.hostIP = "169.254.148.148"
        self.data_port = 61231
        self.ssh_port = 22
        self.server_address_port = (self.hostIP, self.data_port)

        self.sensor_status_message = "Waiting to Connect with RedPitaya UDP Server!"
        print(self.sensor_status_message)

        self.udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_client_socket.settimeout(30)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.header_length = None
        self.total_data_blocks = None
        self.data_counter = 0
        self.local_time_sync = None
        self.first_synced_time = None

    def give_ssh_command(self, command):
        try:
            print("Trying SSH connection to RedPitaya...")
            self.client.connect(self.hostIP, self.ssh_port, "root", "root", timeout=5)
            print("[SSH] Connection established.")
            self.set_sensor_message(f"Connected to Redpitaya {self.hostIP}")

            print(f"[SSH] Sending command: {command}")
            stdin, stdout, stderr = self.client.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()

            self.set_sensor_message(f"Output: {output}")
            if error:
                self.set_sensor_message(f"Error: {error}")
            else:
                self.set_sensor_message(f"SSH command executed successfully")

            if output:
                print(f"SSH Output: {output}")
                return output
        except Exception as e:
            print("SSH connection error:", e)
            self.set_sensor_message(f"SSH connection error: {e}")
        finally:
            self.client.close()
            self.set_sensor_message("Connection closed")

    def set_sensor_message(self, message):
        print(f"[Sensor Status] {message}")
        self.sensor_status_message = message

    def get_sensor_status_message(self):
        return self.sensor_status_message

    def send_msg_to_server(self):
        bytes_to_send = str.encode(self.msg_from_client)
        print(f"Sending message: {self.msg_from_client}")
        self.udp_client_socket.sendto(bytes_to_send, self.server_address_port)

    def get_data_info_from_server(self):
        self.msg_from_client = "-i 1"
        self.send_msg_to_server()
        try:
            packet = self.udp_client_socket.recv(self.buffer_size)
            self.sensor_status_message = f"Sensor Connected Successfully at {self.server_address_port}!"
            print(self.sensor_status_message)
            print(f"Total Received: {len(packet)} Bytes.")

            self.header_length = int(struct.unpack('@f', packet[:4])[0])
            self.total_data_blocks = int(struct.unpack('@f', packet[56:60])[0])
            synced_time = int(struct.unpack('@f', packet[20:24])[0])

            header_data = [h[0] for h in struct.iter_unpack('@f', packet[:self.header_length])]
            print(f"Length of Header: {len(header_data)}")

            self.local_time_sync = time.time() * 1000
            self.first_synced_time = synced_time
            return synced_time, header_data
        except socket.timeout:
            print("UDP receive timeout")
            self.set_sensor_message("UDP receive timeout")
            return None, None

    def get_data_from_server(self, start_time):
        ultrasonic_data = []
        self.data_counter = 0
        acquisition_start = time.time() * 1000

        for i in range(self.total_data_blocks):
            time.sleep(0.001)
            self.msg_from_client = "-a 1"
            self.send_msg_to_server()
            try:
                packet1 = self.udp_client_socket.recv(self.buffer_size)
            except socket.timeout:
                print("UDP receive timeout in data acquisition")
                return None, self.data_counter

            current_time = time.time() * 1000
            elapsed_time = current_time - self.local_time_sync + start_time
            header = [h[0] for h in struct.iter_unpack('@f', packet1[:self.header_length])] if i == 0 else []

            current_data_block_number = int(struct.unpack('@f', packet1[60:64])[0])
            redpitaya_acq_time_stamp = int(struct.unpack('@f', packet1[64:68])[0])

            if i != current_data_block_number:
                print(f"Error: Expected block {i} but received block {current_data_block_number}")
                break

            self.set_sensor_message(
                f"Block {current_data_block_number + 1} received at "
                f"{elapsed_time:.2f}ms (client), {redpitaya_acq_time_stamp}ms (RedPitaya)"
            )

            for j in struct.iter_unpack('@h', packet1[self.header_length:]):
                ultrasonic_data.append(j[0])

            self.data_counter += 1

        print(f"Length of Ultrasonic Data: {len(ultrasonic_data)}")
        print(f"Acquisition took {(time.time() * 1000 - acquisition_start):.2f}ms")

        if len(ultrasonic_data) != self.size_of_raw_adc * self.total_data_blocks:
            print("Invalid data length, returning None")
            return None, self.data_counter

        df = pd.DataFrame({'raw_adc': ultrasonic_data})
        return df['raw_adc'].values, self.data_counter

    def close(self):
        print("Closing RedPitaya connections")
        try:
            self.udp_client_socket.close()
        except Exception as e:
            print(f"Error closing UDP socket: {e}")
        try:
            self.client.close()
        except Exception as e:
            print(f"Error closing SSH client: {e}")

class PlotWorker(QRunnable):
    def __init__(self, func_is_button_checked, rp_sensor, model, *args, **kwargs):
        super().__init__()
        self.func_is_button_checked = func_is_button_checked
        self.rp_sensor = rp_sensor
        self.model = model
        self.dataFilePath = None
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_running = True
        self.saved_signals_count = 0
        self.sensor_data_buffer = deque(maxlen=10)
        self.total_signals_count = 0
        self.acquisition_start_time = None

    @pyqtSlot()
    def run(self):
        print("Start of sensor thread")
        while self.func_is_button_checked(*self.args, **self.kwargs) and self.is_running:
            try:
                if self.acquisition_start_time is None:
                    self.acquisition_start_time = time.time()

                result, data_counter = self.rp_sensor.get_data_from_server(self.acquisition_start_time)
                if result is None:
                    print("No valid data received, skipping")
                    continue

                sensor_timestamp = time.time()
                self.sensor_data_buffer.append({"timestamp": sensor_timestamp, "data": result})
                self.total_signals_count += data_counter
                self.signals.total_signals_count_updated.emit(self.total_signals_count)
                print(f"Emitting result with data length: {len(result)}")
                self.signals.result.emit(result)
                self.signals.finished.emit()
                print("One loop complete!")

                if self.model is not None:
                    print("Features extraction started")
                    features = self.extract_features(result)
                    prediction = self.model.predict([features])[0]
                    self.signals.prediction_result.emit(prediction)
                    print(f"Prediction: {prediction}")
            except Exception as e:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            finally:
                time.sleep(0.2)

    def extract_features(self, signal):
        print("Features extraction")
        return [
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.ptp(signal),
            np.var(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
        ]

    def save_data(self, data, label_folder, timestamp):
        if not self.dataFilePath:
            print("Error: dataFilePath is not set. Skipping save.")
            return

        subfolder = os.path.join(self.dataFilePath, label_folder)
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')}.csv"
        filepath = os.path.join(subfolder, filename)

        df = pd.DataFrame({'raw_adc': data})
        print(f"Saving data shape: {df.shape} to {filepath}")
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        self.saved_signals_count += 1
        self.signals.signals_count_updated.emit(self.saved_signals_count)

    def get_labeled_data(self, detection_timestamp, window_size=1.0):
        if not self.sensor_data_buffer:
            print("Sensor data buffer empty")
            return None, 0

        closest_sensor_data = min(self.sensor_data_buffer, key=lambda x: abs(x["timestamp"] - detection_timestamp))
        sensor_timestamp_diff = abs(closest_sensor_data["timestamp"] - detection_timestamp)

        print(f"YOLO detection timestamp: {detection_timestamp:.3f}")
        print(f"Closest sensor data timestamp: {closest_sensor_data['timestamp']:.3f}")
        print(f"Timestamp difference: {sensor_timestamp_diff:.3f}")

        if sensor_timestamp_diff <= window_size:
            match_confidence = 1 - (sensor_timestamp_diff / window_size)
            print(f"Matched sensor data at diff {sensor_timestamp_diff:.3f}s (confidence: {match_confidence:.2f})")
            return closest_sensor_data["data"], match_confidence
        else:
            print("No sensor data sync within threshold, skipping save.")
            return None, 0

    def set_dataFilePath(self, dataFilePath):
        self.dataFilePath = dataFilePath

class SensorInitWorker(QRunnable):
    def __init__(self, rp_sensor, on_success_callback):
        super().__init__()
        self.rp_sensor = rp_sensor
        self.on_success_callback = on_success_callback
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            commands = ["cd /usr/RedPitaya/Examples/C", "./dma_with_udp_faster"]
            full_command = " && ".join(commands)
            self.rp_sensor.give_ssh_command(full_command)
            print("Waiting for RedPitaya to start the UDP server...")
            time.sleep(3)
            start_time, header_info = self.rp_sensor.get_data_info_from_server()
            if start_time is None or header_info is None:
                raise Exception("Failed to initialize sensor: No header data received")
            self.signals.result.emit((start_time, header_info))
        except Exception as e:
            print("Sensor init error:", e)
            traceback.print_exc()
            self.signals.error.emit((type(e), e, traceback.format_exc()))
        finally:
            self.signals.finished.emit()

class YOLODetectionWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.camera = YOLOCamera(model_path="yolov8n.pt")
        self.signals = WorkerSignals()
        self.is_running = True

    @pyqtSlot()
    def run(self):
        print("Starting YOLO detection thread")
        while self.is_running:
            try:
                detection_data = self.camera.run_person_detection_step()
                if detection_data is None:
                    print(f"No frame received from RealSense at {time.time():.3f}")
                    continue

                person_detected = detection_data["person_detected"]
                distance = detection_data["distance"] if detection_data["distance"] is not None else 0.0
                processed_image = detection_data["image"]
                timestamp = detection_data["timestamp"]

                if person_detected:
                    print(f"Person detected at {timestamp:.3f}, distance: {distance:.2f}m")
                else:
                    print(f"No person detected at {timestamp:.3f}")

                self.signals.person_detected.emit(person_detected, distance, timestamp)

                height, width, channel = processed_image.shape
                bytes_per_line = 3 * width
                qimg = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
                self.signals.update_frame.emit(qimg)

                time.sleep(0.01)
            except Exception as e:
                print(f"YOLO detection error: {e}")
                self.signals.error.emit((type(e), e, traceback.format_exc()))

        print("Stopping YOLO detection thread")
        self.camera.stop()
        cv2.destroyAllWindows()

    def stop(self):
        print("Stopping YOLO worker")
        self.is_running = False
        self.camera.stop()

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(tuple)
    finished = pyqtSignal()
    signals_count_updated = pyqtSignal(int)
    total_signals_count_updated = pyqtSignal(int)
    person_detected = pyqtSignal(bool, float, float)
    update_frame = pyqtSignal(object)
    prediction_result = pyqtSignal(int)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rp_sensor = RedPitayaSensor()
        self.start_time = None
        self.header_info = None
        self.yolo_worker = None
        self.plotWorker = None
        self.threadpool = QThreadPool()
        self.sensor_status_message = self.rp_sensor.get_sensor_status_message()
        self.previous_distance = 0.0
        self.distance_history = deque(maxlen=5)
        self.button_is_checked = True
        self.realtime_chkbox_checked = False
        self.show_region_to_select = False
        self.raw_adc_data = None
        self.previous_range_selector_region = (100, 1000)
        self.prediction_mode = False
        self.model = None  # ← Loaded only in prediction mode

        # Default save folder (project root/DataCollection/Default)

        self.default_data_folder = os.path.join(os.path.dirname(__file__), "DataCollection/Default")
        os.makedirs(self.default_data_folder, exist_ok=True)
        self.dataFilePath = self.default_data_folder

        self.setWindowTitle("IntelliLight DCL")
        self.setMinimumSize(1200, 800)

        self.setStyleSheet("""
            QMainWindow { background-color: #2E2E2E; }
            QPushButton {background-color:#4A4A4A;color:white;border-radius:5px;padding:8px;font-size:14px;}
            QPushButton:hover {background-color:#5A5A5A;}
            QPushButton:pressed {background-color:#3A3A3A;}
            QCheckBox {color:white;font-size:14px;}
            QLabel {color:white;font-size:14px;}
            QGroupBox {border:1px solid #555;border-radius:5px;margin-top:10px;color:white;font-weight:bold;}
            QGroupBox::title {subcontrol-origin:margin;subcontrol-position:top left;padding:0 3px;color:white;}
        """)

        # Plot & Video
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('black')
        self.plot_widget.setTitle("Sensor Data", color="w", size="16pt")
        self.plot_widget.setLabel('left', 'ADC Value', color='white', size='14pt')
        self.plot_widget.setLabel('bottom', 'Time (samples)', color='white', size='14pt')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #555; border-radius: 5px;")
        self.video_label.setMinimumSize(400, 300)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.video_label)
        splitter.setSizes([800, 400])

        # Controls
        controls_group = QGroupBox("Plot Controls")
        self.controls_mid_layout = QHBoxLayout()
        self.realtime_chkbox = QCheckBox("Realtime")
        self.realtime_chkbox.setToolTip("Enable real-time sensor data acquisition")
        self.controls_mid_layout.addWidget(self.realtime_chkbox)
        self.show_region_chkbox = QCheckBox("Region-Select")
        self.show_region_chkbox.setToolTip("Select a region of the plot to analyze")
        self.controls_mid_layout.addWidget(self.show_region_chkbox)
        self.confirm_region_btn = QPushButton("Confirm")
        self.confirm_region_btn.setToolTip("Confirm the selected plot region")
        self.controls_mid_layout.addWidget(self.confirm_region_btn)

        self.browse_save_btn = QPushButton("Browse Save Folder")
        self.browse_save_btn.setToolTip("Choose where collected data will be stored")
        self.controls_mid_layout.addWidget(self.browse_save_btn)

        controls_group.setLayout(self.controls_mid_layout)

        sensor_group = QGroupBox("Sensor Controls")
        self.ssh_command_layout = QHBoxLayout()
        self.start_sensor_btn = QPushButton(QIcon("start_icon.png"), "Start Sensor")
        self.start_sensor_btn.setToolTip("Start the RedPitaya sensor server")
        self.ssh_command_layout.addWidget(self.start_sensor_btn)
        self.stop_sensor_btn = QPushButton(QIcon("stop_icon.png"), "Stop Sensor")
        self.stop_sensor_btn.setToolTip("Stop the RedPitaya sensor server")
        self.ssh_command_layout.addWidget(self.stop_sensor_btn)
        self.open_yolo_button = QPushButton(QIcon("camera_icon.png"), "Start YOLO Detection")
        self.open_yolo_button.setToolTip("Start/Stop YOLO person detection")
        self.ssh_command_layout.addWidget(self.open_yolo_button)
        sensor_group.setLayout(self.ssh_command_layout)

        prediction_group = QGroupBox("Prediction Mode")
        prediction_layout = QHBoxLayout()

        self.prediction_chkbox = QCheckBox("Sensor Only Prediction Mode")
        self.prediction_chkbox.setToolTip("Enable sensor-based presence detection")
        prediction_layout.addWidget(self.prediction_chkbox, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.prediction_indicator = QLabel()
        self.prediction_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_indicator.setStyleSheet("""
            background-color: gray;
            border-radius: 25px;
            border: 2px solid #555;
            min-width: 50px;
            min-height: 50px;
        """)
        self.prediction_indicator.setVisible(False)  # hidden until mode ON


        self.prediction_label = QLabel("Prediction: Unknown")
        self.prediction_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.prediction_label.setVisible(False)  # ← HIDDEN by default

        left_box = QVBoxLayout()
        left_box.addWidget(self.prediction_indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        left_box.addWidget(self.prediction_label, alignment=Qt.AlignmentFlag.AlignCenter)
        left_box.addStretch()

        self.light_icon = QLabel()
        self.light_icon.setFixedSize(50, 50)
        self.light_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.light_icon.setStyleSheet("background-color:#333;border-radius:25px;border:2px solid #555;")
        self.light_icon.setPixmap(self._bulb_icon(False, size=50))
        self.light_icon.setVisible(False)

        self.light_label = QLabel("Light: OFF")
        self.light_label.setStyleSheet("font-size:14px;font-weight:bold;color:white;")
        self.light_label.setVisible(False)

        self.absence_counter_label = QLabel("Absence count: 0 / 5")
        self.absence_counter_label.setStyleSheet("color:#aaa;font-size:12px;")
        self.absence_counter_label.setVisible(False)

        right_box = QVBoxLayout()
        right_box.addWidget(self.light_icon, alignment=Qt.AlignmentFlag.AlignCenter)
        right_box.addWidget(self.light_label, alignment=Qt.AlignmentFlag.AlignCenter)
        right_box.addWidget(self.absence_counter_label, alignment=Qt.AlignmentFlag.AlignCenter)

        prediction_layout.addLayout(left_box)
        prediction_layout.addStretch()
        prediction_layout.addLayout(right_box)
        prediction_layout.addStretch()

        prediction_group.setLayout(prediction_layout)
        
        status_counts_group = QGroupBox("Data Status")
        status_layout = QVBoxLayout()

        # signal counters
        counters_layout = QHBoxLayout()
        self.signal_numbers_label = QLabel("0 Signals saved")
        self.total_signal_numbers_label = QLabel("0 Signals received")
        counters_layout.addWidget(self.signal_numbers_label)
        counters_layout.addWidget(self.total_signal_numbers_label)
        status_layout.addLayout(counters_layout)

        # current folder label
        self.save_folder_label = QLabel(f"Save folder: {self.dataFilePath}")
        self.save_folder_label.setStyleSheet("color:#aaa;font-size:12px;")
        self.save_folder_label.setWordWrap(True)
        status_layout.addWidget(self.save_folder_label)

        status_counts_group.setLayout(status_layout)

        # Main layout

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(sensor_group)
        main_layout.addWidget(prediction_group)
        main_layout.addWidget(status_counts_group)

        self.widget = QWidget()
        self.widget.setLayout(main_layout)
        self.setCentralWidget(self.widget)
 
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.sensor_status_label = QLabel(self.sensor_status_message)
        self.app_status_label = QLabel("App Started")
        self.status_bar.addWidget(self.sensor_status_label)
        self.status_bar.addWidget(self.app_status_label)
        self.status_bar.setStyleSheet("QStatusBar { background-color:#3A3A3A;color:white; }")

        self.button = QPushButton("Toggle Plot")
        self.button.setCheckable(True)
        self.button.setChecked(self.button_is_checked)
        self.button.setFixedSize(QSize(100, 40))
        self.button.setToolTip("Toggle plot visibility")
        main_layout.addWidget(self.button)

        self.range_selector = pg.LinearRegionItem()


        # Connections 
        self.show_region_chkbox.stateChanged.connect(self.show_region_handler)
        self.realtime_chkbox.stateChanged.connect(self.realtime_checkbox_handler)
        self.confirm_region_btn.clicked.connect(self.confirm_region_selection_btn_handler)
        self.start_sensor_btn.clicked.connect(self.start_sensor_btn_handler)
        self.stop_sensor_btn.clicked.connect(self.stop_sensor_btn_handler)
        self.open_yolo_button.clicked.connect(self.start_yolo_detection)
        self.button.clicked.connect(self.the_button_was_toggled)
        self.prediction_chkbox.stateChanged.connect(self.prediction_checkbox_handler)
        self.browse_save_btn.clicked.connect(self.browse_save_folder)  # ← NEW

        self.light_on = False
        self.absence_streak = 0
        self.ABSENCE_THRESHOLD = 5

        self._hide_light_widgets()

    # SVG bulb
    def _bulb_icon(self, on: bool, size: int = 50) -> QPixmap:
        color = "#00FF00" if on else "#FF4444"
        svg = f'''
        <svg width="{size}" height="{size}" viewBox="0 0 24 24"
            preserveAspectRatio="xMidYMid meet"
            xmlns="http://www.w3.org/2000/svg">
        <path fill="{color}" d="M12 2a7 7 0 0 0-7 7c0 2.61 1.5 4.83 3.67 6.03L9 18h6l.33-2.97A6.98 6.98 0 0 0 19 9a7 7 0 0 0-7-7z"/>
        <rect fill="{color}" x="9" y="18" width="6" height="2" rx="1"/>
        </svg>
        '''

        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)

        renderer = QSvgRenderer(QByteArray(svg.encode()))
        painter = QPainter(pm)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()

        return pm

    # Show / hide light widgets based on prediction mode
    def _show_light_widgets(self):
        self.light_icon.setVisible(True)
        self.light_label.setVisible(True)
        self.absence_counter_label.setVisible(True)

    def _hide_light_widgets(self):
        self.light_icon.setVisible(False)
        self.light_label.setVisible(False)
        self.absence_counter_label.setVisible(False)

    # Show / hide Prediction widgets
    def _show_prediction_widgets(self):
        self.prediction_indicator.setVisible(True)
        self.prediction_label.setVisible(True)
        self.prediction_label.setText("Prediction: Unknown")

    def _hide_prediction_widgets(self):
        self.prediction_indicator.setVisible(False)
        self.prediction_label.setVisible(False)

    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Save Collected Data",
            self.dataFilePath
        )
        if folder:
            self.dataFilePath = folder
            self.save_folder_label.setText(f"Save folder: {folder}")
            self.set_status_message(f"Save folder: {folder}", "cyan")
            if self.plotWorker:
                self.plotWorker.set_dataFilePath(self.dataFilePath)


    def prediction_checkbox_handler(self, state):
        self.prediction_mode = (state == Qt.CheckState.Checked.value)
        if self.prediction_mode:
            print("Prediction mode enabled (Sensor only)")
            self.set_status_message("Loading model...", "yellow")
            try:
                self.model = joblib.load(r"TrainedModels\presence_detection_model_31102025.pkl")
                print("Trained model loaded successfully.")
                self.set_status_message("Model loaded", "green")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.set_status_message(f"Model load failed: {e}", "red")
                self.prediction_chkbox.setChecked(False)
                return

            self.open_yolo_button.setDisabled(True)
            #if not self.realtime_chkbox_checked:
            #    self.realtime_chkbox.setChecked(True)

            # Show everything
            self._show_prediction_widgets()
            self._show_light_widgets()

        else:
            print("Prediction mode disabled")
            self.model = None
            self.set_status_message("Prediction mode disabled", "white")
            self.open_yolo_button.setDisabled(False)

            # Hide everything except checkbox
            self._hide_prediction_widgets()
            self._hide_light_widgets()

    # Prediction update – light logic + UI
    def update_prediction_indicator(self, prediction: int):
        if self.prediction_mode:
            if prediction == 1:
                if not self.light_on:
                    self.light_on = True
                    self.light_icon.setPixmap(self._bulb_icon(True, size=50))
                    self.light_label.setText("Light: ON")
                    self.light_label.setStyleSheet("color:limegreen;font-weight:bold;font-size:14px;")
                    self.set_status_message("Presence → Light ON", "green")
                self.absence_streak = 0
                self.absence_counter_label.setText(f"Absence count: 0 / {self.ABSENCE_THRESHOLD}")
            else:
                self.absence_streak += 1
                self.absence_counter_label.setText(f"Absence count: {self.absence_streak} / {self.ABSENCE_THRESHOLD}")
                if self.light_on and self.absence_streak >= self.ABSENCE_THRESHOLD:
                    self.light_on = False
                    self.light_icon.setPixmap(self._bulb_icon(False, size=50))
                    self.light_label.setText("Light: OFF")
                    self.light_label.setStyleSheet("color:red;font-weight:bold;font-size:14px;")
                    self.set_status_message(f"{self.ABSENCE_THRESHOLD} consecutive absences → Light OFF", "red")

        if prediction == 1:
            style = "background-color:limegreen;border-radius:25px;border:2px solid #555;min-width: 50px;min-height: 50px;"
            self.prediction_label.setText("Prediction: Presence")
            self.prediction_label.setStyleSheet("color:limegreen;font-weight:bold;")
        else:
            style = "background-color:red;border-radius:25px;border:2px solid #555;min-width: 50px;min-height: 50px;"
            self.prediction_label.setText("Prediction: Absence")
            self.prediction_label.setStyleSheet("color:red;font-weight:bold;")
        self.prediction_indicator.setStyleSheet(style)


    def set_status_message(self, message, color="white"):
        self.app_status_label.setText(message)
        self.app_status_label.setStyleSheet(f"color:{color};")
        print(f"[Status] {message}")

    def start_yolo_detection(self):
        if not self.yolo_worker:
            self.yolo_worker = YOLODetectionWorker()
            self.yolo_worker.signals.update_frame.connect(self.update_video_feed)
            self.yolo_worker.signals.person_detected.connect(self.handle_person_detected)
            self.yolo_worker.signals.error.connect(self.handle_worker_error)
            self.threadpool.start(self.yolo_worker)
            self.open_yolo_button.setText("Stop YOLO Detection")
            self.set_status_message("YOLO Detection started", "green")
        else:
            self.stop_yolo_detection()

    def stop_yolo_detection(self):
        if self.yolo_worker:
            self.yolo_worker.stop()
            self.threadpool.waitForDone(2000)
            self.yolo_worker = None
            self.open_yolo_button.setText("Start YOLO Detection")
            self.set_status_message("YOLO Detection stopped", "white")
            self.video_label.clear()


    def update_video_feed(self, qimg):
        pixmap = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def handle_person_detected(self, detected, distance, timestamp):
        if not self.plotWorker or not self.plotWorker.sensor_data_buffer:
            print("No sensor data available for labeling")
            self.set_status_message("No sensor data available for labeling", "red")
            return
        sensor_data, match_conf = self.plotWorker.get_labeled_data(timestamp, window_size=1.0)
        if sensor_data is None or match_conf < 0.5:
            print("No sensor data sync within threshold, skipping save.")
            self.set_status_message("Detection skipped: poor timestamp sync", "red")
            return
        label_folder = self.classify_detection(distance, detected)
        self.plotWorker.set_dataFilePath(self.dataFilePath)
        self.plotWorker.save_data(sensor_data, label_folder, timestamp)
        self.set_status_message(f"Labeled as '{label_folder}' (dist: {distance:.2f}m, conf: {match_conf:.2f})", "white")
        self.previous_distance = distance

    def classify_detection(self, distance, detected):
        if not detected:
            return "Object"
        if abs(distance - self.previous_distance) > 0.2:
            return "Doubt"
        if 0 < distance < 1.7:
            self.distance_history.append(distance)
            if len(self.distance_history) == 5:
                diffs = [abs(self.distance_history[i+1] - self.distance_history[i]) for i in range(4)]
                if any(diff > 0.1 for diff in diffs):
                    return "MovingPerson"
                else:
                    avg_distance = sum(self.distance_history) / len(self.distance_history)
                    return "SittingPerson" if avg_distance < 1.0 else "StandingPerson"
            else:
                return "Person"
        return "Object"

    def show_region_handler(self, state):
        self.sensor_status_label.setText(self.rp_sensor.get_sensor_status_message())
        if state == Qt.CheckState.Checked.value:
            print("Region select checked!")
            self.realtime_chkbox.setDisabled(True)
            self.confirm_region_btn.setDisabled(False)
            self.show_region_to_select = True
            self.range_selector = pg.LinearRegionItem()
            self.range_selector.sigRegionChangeFinished.connect(self.region_changed_on_linear_region)
            self.range_selector.setRegion(self.previous_range_selector_region)
            self.plot_widget.addItem(self.range_selector)
        elif state == Qt.CheckState.Unchecked.value:
            self.reset_btn_view()
            if hasattr(self, 'range_selector'):
                self.plot_widget.removeItem(self.range_selector)

    def confirm_region_selection_btn_handler(self):
        if self.show_region_to_select:
            print(f"Confirmed Region: {self.range_selector.getRegion()}")
            self.previous_range_selector_region = self.range_selector.getRegion()
            self.plot_adc_data(self.raw_adc_data)
            self.show_region_handler(self.show_region_chkbox.checkState().value)

    def reset_btn_view(self):
        self.realtime_chkbox.setDisabled(False)
        self.show_region_chkbox.setDisabled(False)
        self.confirm_region_btn.setDisabled(True)

    def region_changed_on_linear_region(self):
        print("Region Changed!")
        print(self.range_selector.getRegion())

    def the_button_was_toggled(self, checked):
        self.button_is_checked = checked
        print("Checked:", self.button_is_checked)
        self.button.setText(f"Status: {self.button_is_checked}")
        self.plot_adc_data(self.raw_adc_data)

    def plot_adc_data(self, data=None):
        self.sensor_status_label.setText(self.rp_sensor.get_sensor_status_message())
        self.plot_widget.clear()
        if data is None:
            print("No data to plot.")
            return
        x = np.arange(len(data))
        y = data
        self.raw_adc_data = y
        pen = pg.mkPen(color='cyan', width=2)
        self.plot_widget.plot(x, y, pen=pen)
        print("Data plotted successfully")

    def realtime_checkbox_handler(self, state):
        if state == Qt.CheckState.Checked.value:
            self.realtime_chkbox_checked = True
            print("Go Realtime!")
            self.show_region_chkbox.setDisabled(True)
            self.confirm_region_btn.setDisabled(True)
            self.plotWorker = PlotWorker(
                self.func_is_realtime_checked,
                self.rp_sensor,
                self.model if self.prediction_mode else None
            )
            self.plotWorker.signals.result.connect(self.plot_adc_data)
            self.plotWorker.signals.total_signals_count_updated.connect(self.update_total_signal_numbers)
            self.plotWorker.signals.signals_count_updated.connect(self.update_saved_signal_numbers)
            self.plotWorker.signals.prediction_result.connect(self.update_prediction_indicator)
            self.plotWorker.signals.error.connect(self.handle_worker_error)
            self.threadpool.start(self.plotWorker)
            self.set_status_message("Realtime sensor acquisition started", "green")
        else:
            self.realtime_chkbox_checked = False
            self.reset_btn_view()
            if self.plotWorker:
                self.plotWorker.is_running = False
                self.threadpool.waitForDone(2000)
                self.plotWorker = None
                self.set_status_message("Realtime sensor acquisition stopped", "white")

    def func_is_realtime_checked(self):
        return self.realtime_chkbox_checked

    def start_sensor_btn_handler(self):
        worker = SensorInitWorker(self.rp_sensor, self.start_sensor_callback)
        worker.signals.result.connect(self.start_sensor_callback)
        worker.signals.error.connect(self.handle_worker_error)
        self.threadpool.start(worker)

    def start_sensor_callback(self, result):
        self.start_time, self.header_info = result
        self.set_status_message("Sensor started successfully", "green")

    def stop_sensor_btn_handler(self):
        command = "pidof dma_with_udp_faster"
        pid = self.rp_sensor.give_ssh_command(command)
        if pid and pid.strip():
            self.rp_sensor.give_ssh_command(f"kill {pid.strip()}")
        self.set_status_message("Sensor stopped", "white")

    def handle_worker_error(self, error_info):
        exctype, value, _ = error_info
        print(f"Worker error: {exctype.__name__}: {value}")
        self.set_status_message(f"Error: {exctype.__name__}: {value}", "red")

    def update_saved_signal_numbers(self, count):
        self.signal_numbers_label.setText(f"{count} signals saved")

    def update_total_signal_numbers(self, count):
        self.total_signal_numbers_label.setText(f"{count} signals received")

    def closeEvent(self, event):
        print("Closing application")
        if self.yolo_worker:
            self.stop_yolo_detection()
        if self.plotWorker:
            self.realtime_checkbox_handler(0)
        self.rp_sensor.close()
        self.threadpool.waitForDone(3000)

        # reset UI
        if self.light_on:
            self.light_icon.setPixmap(self._bulb_icon(False))
            self.light_label.setText("Light: OFF")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())