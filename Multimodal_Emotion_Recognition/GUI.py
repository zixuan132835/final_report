# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import datetime
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QFileDialog, QFrame,
                             QScrollArea, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QFontDatabase


class Model(nn.Module):
    """Custom CNN emotion recognition model"""

    def __init__(self):
        super(Model, self).__init__()
        self.bn_x = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(32, momentum=0.5)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32, momentum=0.5)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64, momentum=0.5)

        self.fc1 = nn.Linear(5 * 5 * 64, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.fc3 = nn.Linear(1024, 7)  # 7 emotions

    def forward(self, x):
        x = self.bn_x(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(self.conv1(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.relu(self.bn_conv2(self.conv2(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.relu(self.bn_conv3(self.conv3(x))), kernel_size=3, stride=2, ceil_mode=True)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EmotionClassifier:
    """Emotion classifier wrapper"""

    def __init__(self):
        self._load_face_detector()
        self._load_emotion_model()
        self.emotion_map = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

    def _load_face_detector(self):
        cascade_path = os.path.join('./datasets/haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Failed to load face detection model")

    def _load_emotion_model(self):
        model_path = os.path.join('./model/model_params.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = Model()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def get_emotion(self, inputs):
        """Get emotion classification result"""
        inputs = self.preprocess(inputs)

        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            probability = F.softmax(outputs, dim=1).detach().numpy().flatten()

        return self.emotion_map[predicted.item()], probability

    def preprocess(self, inputs):
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        inputs = trans(inputs)
        inputs = inputs.unsqueeze(0)
        return inputs


class VideoThread(QThread):
    """Thread for video playback with optimized performance"""
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._stop_flag = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.frame_skip = 2  # skip frames to improve performance

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open video: {self.video_path}")
                self.finished.emit()
                return

            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 30

            while not self._stop_flag:
                self.mutex.lock()
                ret, frame = cap.read()
                self.mutex.unlock()

                if not ret:
                    break

                frame_count += 1
                if frame_count % (self.frame_skip + 1) != 0:
                    continue

                self.frame_ready.emit(frame.copy())

                self.msleep(max(1, delay // (self.frame_skip + 1)))

        except Exception as e:
            self.error_occurred.emit(f"Video processing error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            self.finished.emit()

    def stop(self):
        self.mutex.lock()
        self._stop_flag = True
        self.mutex.unlock()
        self.wait(500)


class BarGraphWidget(QWidget):
    """Custom bar graph widget to display emotion probabilities"""

    def __init__(self):
        super().__init__()
        self.probabilities = None
        self.setMinimumSize(300, 200)

        self.bar_colors = [
            QColor(231, 76, 60),   # Angry
            QColor(142, 68, 173),  # Disgust
            QColor(41, 128, 185),  # Fear
            QColor(39, 174, 96),   # Happy
            QColor(44, 62, 80),    # Sad
            QColor(243, 156, 18),  # Surprise
            QColor(149, 165, 166)  # Neutral
        ]

        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities
        self.update()

    def paintEvent(self, event):
        if self.probabilities is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 10
        bar_height = 25
        spacing = 5
        text_width = 50

        max_prob = max(self.probabilities) if max(self.probabilities) > 0 else 1

        for i, (prob, label, color) in enumerate(zip(self.probabilities, self.emotion_labels, self.bar_colors)):
            y = margin + i * (bar_height + spacing)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(60, 60, 60))
            painter.drawRoundedRect(text_width, y, width - text_width - margin, bar_height, 3, 3)

            bar_width = int((width - text_width - margin - 10) * (prob / max_prob))
            painter.setBrush(color)
            painter.drawRoundedRect(text_width, y, bar_width, bar_height, 3, 3)

            painter.setPen(QColor(220, 220, 220))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(margin, y + bar_height - 7, f"{label}")

            painter.drawText(width - 50, y + bar_height - 7, f"{prob * 100:.1f}%")


class DepressionAnalyzer:
    """Analyze depression tendency based on emotion history"""
    def __init__(self, history_size=30):
        self.history = deque(maxlen=history_size)
        # Emotions that contribute to depression risk
        self.negative_emotions = {'Angry', 'Disgust', 'Fear', 'Sad'}
        self.sad_weight = 1.5  # Sad emotion has higher impact

    def update(self, emotion):
        self.history.append(emotion)

    def get_risk(self):
        if not self.history:
            return 0.0

        # Count weighted negative emotions
        negative_count = 0
        for e in self.history:
            if e in self.negative_emotions:
                if e == 'Sad':
                    negative_count += self.sad_weight
                else:
                    negative_count += 1

        # Normalize to 0-1 range (max possible score if all history are Sad)
        max_score = len(self.history) * self.sad_weight
        risk = negative_count / max_score if max_score > 0 else 0
        return risk * 100  # percentage


class EmotionRecognitionApp(QMainWindow):
    """Smart Emotion Recognition System with Depression Detection"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Recognition System with Depression Analysis")
        self.setMinimumSize(1200, 800)

        self._load_fonts()
        self._init_log_file()

        try:
            self.classifier = EmotionClassifier()
        except Exception as e:
            self._show_error(f"Initialization failed: {str(e)}")
            return

        self.cap = None
        self.camera_active = False

        self.depression_analyzer = DepressionAnalyzer(history_size=30)

        self._init_ui()

        self.video_thread = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.is_recognizing = False

    def _init_log_file(self):
        self.log_file_path = "emotion_log.txt"
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Emotion Recognition Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def _load_fonts(self):
        try:
            font_path = os.path.join('fonts', 'arial.ttf')
            if os.path.exists(font_path):
                QFontDatabase.addApplicationFont(font_path)
        except Exception as e:
            print(f"Font loading failed: {str(e)}")

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Emotion Recognition & Depression Risk Analysis")
        title_label.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #3498db;
            padding: 10px;
            background-color: #2c3e50;
            border-radius: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Content area (video + results + log)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # Left panel - video
        video_panel = QFrame()
        video_panel.setFrameShape(QFrame.StyledPanel)
        video_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px;")
        video_panel.setMinimumWidth(640)
        video_panel.setMinimumHeight(480)

        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(5, 5, 5, 5)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; border-radius: 3px;")
        video_layout.addWidget(self.video_label)

        # Middle panel - results
        results_panel = QFrame()
        results_panel.setFrameShape(QFrame.StyledPanel)
        results_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px;")
        results_panel.setMinimumWidth(300)

        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_layout.setSpacing(15)

        result_title = QLabel("Detection Results")
        result_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        result_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(result_title)

        self.emotion_label = QLabel("Waiting for detection...")
        self.emotion_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #4a90e2;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 10px;
        """)
        self.emotion_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.emotion_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #3a3a3a;")
        results_layout.addWidget(separator)

        # Emotion probability bar chart
        self.bar_graph = BarGraphWidget()
        results_layout.addWidget(self.bar_graph)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet("color: #3a3a3a;")
        results_layout.addWidget(separator2)

        # Depression risk section
        depression_title = QLabel("Depression Risk Analysis")
        depression_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        depression_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(depression_title)

        self.depression_risk_label = QLabel("Risk: 0.0%")
        self.depression_risk_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            color: #e74c3c;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 5px;
        """)
        self.depression_risk_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.depression_risk_label)

        # Risk progress bar
        self.depression_progress = QProgressBar()
        self.depression_progress.setMinimum(0)
        self.depression_progress.setMaximum(100)
        self.depression_progress.setValue(0)
        self.depression_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: #1e1e1e;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
                border-radius: 3px;
            }
        """)
        results_layout.addWidget(self.depression_progress)

        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setStyleSheet("color: #3a3a3a;")
        results_layout.addWidget(separator3)

        suggestion_title = QLabel("Analysis Suggestions")
        suggestion_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        suggestion_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(suggestion_title)

        self.suggestion_label = QLabel("System initializing...")
        self.suggestion_label.setStyleSheet("""
            font-size: 14px; 
            color: #f39c12;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 10px;
        """)
        self.suggestion_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.suggestion_label.setWordWrap(True)
        results_layout.addWidget(self.suggestion_label)

        # Right panel - log
        log_panel = QFrame()
        log_panel.setFrameShape(QFrame.StyledPanel)
        log_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px;")
        log_panel.setMinimumWidth(300)

        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.setSpacing(10)

        log_title = QLabel("Recognition Log")
        log_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        log_title.setAlignment(Qt.AlignCenter)
        log_layout.addWidget(log_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                border: none;
                background: #2d2d2d;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            background-color: #1e1e1e;
            color: #e0e0e0;
            border-radius: 5px;
            padding: 5px;
            font-size: 12px;
            border: none;
        """)
        scroll_area.setWidget(self.log_text)
        log_layout.addWidget(scroll_area)

        content_layout.addWidget(video_panel)
        content_layout.addWidget(results_panel)
        content_layout.addWidget(log_panel)

        main_layout.addLayout(content_layout)

        # Bottom button panel
        button_panel = QFrame()
        button_panel.setStyleSheet("background-color: transparent;")

        button_layout = QHBoxLayout(button_panel)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        self.toggle_btn = self._create_button("Start Recognition", "#4a90e2", self.toggle_recognition)
        self.image_btn = self._create_button("Image Detection", "#7ed321", self.open_image)
        self.video_btn = self._create_button("Video Detection", "#f5a623", self.open_video)
        self.quit_btn = self._create_button("Exit", "#d0021b", self.close)

        button_layout.addWidget(self.toggle_btn)
        button_layout.addWidget(self.image_btn)
        button_layout.addWidget(self.video_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.quit_btn)

        main_layout.addWidget(button_panel)

    def _create_button(self, text, color, callback):
        btn = QPushButton(text)
        btn.setFixedHeight(40)
        btn.setMinimumWidth(120)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color, 20)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 30)};
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def _darken_color(self, hex_color, percent):
        color = QColor(hex_color)
        h, s, l, a = color.getHslF()
        l = max(0, l - (percent / 100))
        return QColor.fromHslF(h, s, l, a).name()

    def _show_error(self, message):
        error_label = QLabel(message)
        error_label.setStyleSheet("color: red; font-size: 16px;")
        error_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(error_label)
        QMessageBox.critical(self, "System Error", message)

    def init_camera(self):
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.log_text.append("Error: Cannot open camera")
                    self.cap = None
                    return False
                self.camera_active = True
                return True
            except Exception as e:
                self.log_text.append(f"Camera initialization error: {str(e)}")
                self.cap = None
                return False
        return True

    def release_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.camera_active = False

    def update_frame(self):
        if not self.is_recognizing:
            return

        if not self.camera_active:
            if not self.init_camera():
                self.is_recognizing = False
                self.toggle_btn.setText("Start Recognition")
                return

        ret, frame = self.cap.read()
        if not ret:
            self.log_text.append("Error: Cannot read frame from camera")
            self.release_camera()
            self.is_recognizing = False
            self.toggle_btn.setText("Start Recognition")
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_frame(rgb_frame)

        processed_frame, emotion, probability = self._process_frame(frame)

        self.emotion_label.setText(f"Detected Emotion: {emotion}")

        # Update depression analysis
        if emotion != "No face detected" and probability is not None:
            self.depression_analyzer.update(emotion)
            risk = self.depression_analyzer.get_risk()
            self.depression_risk_label.setText(f"Risk: {risk:.1f}%")
            self.depression_progress.setValue(int(risk))
        else:
            risk = 0.0

        self.update_suggestion(emotion, probability, risk)

        if probability is not None:
            self.bar_graph.set_probabilities(probability)
            self._log_recognition(emotion, probability, risk)

    def _process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = self.classifier.face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion = "No face detected"
        probability = None

        if len(faces) > 0:
            (x, y, w, h) = [coord * 2 for coord in faces[0]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            try:
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0:
                    return frame, emotion, probability

                face = cv2.resize(face_roi, (42, 42))
                emotion, probability = self.classifier.get_emotion(Image.fromarray(face))

                # Draw emotion label with PIL for better font support
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()

                draw.text((x, y - 30), emotion, font=font, fill=(255, 0, 0))
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.log_text.append(f"Face processing error: {str(e)}")

        return frame, emotion, probability

    def _display_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def toggle_recognition(self):
        self.is_recognizing = not self.is_recognizing

        if self.is_recognizing:
            self.toggle_btn.setText("Pause Recognition")
            self.toggle_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self._darken_color("#4a90e2", 20)};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-size: 14px;
                    min-width: 100px;
                }}
            """)
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
        else:
            self.toggle_btn.setText("Start Recognition")
            self.toggle_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #4a90e2;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-size: 14px;
                    min-width: 100px;
                }}
            """)
            self.release_camera()

    def open_image(self):
        if self.is_recognizing:
            self.toggle_recognition()

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.webp)"
        )

        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    pil_image = Image.open(file_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                processed_image, emotion, probability = self._process_frame(image)
                self._display_frame(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

                self.emotion_label.setText(f"Detected Emotion: {emotion}")

                # Update depression for a single image (reset history)
                self.depression_analyzer = DepressionAnalyzer(history_size=30)
                if emotion != "No face detected":
                    self.depression_analyzer.update(emotion)
                risk = self.depression_analyzer.get_risk()
                self.depression_risk_label.setText(f"Risk: {risk:.1f}%")
                self.depression_progress.setValue(int(risk))

                self.update_suggestion(emotion, probability, risk)

                if probability is not None:
                    self.bar_graph.set_probabilities(probability)
                    self._log_recognition(emotion, probability, risk)

            except Exception as e:
                self.log_text.append(f"Error: Cannot process image - {str(e)}")
                self._save_log_to_file(f"Error: Cannot process image - {str(e)}")

    def open_video(self):
        if self.is_recognizing:
            self.toggle_recognition()

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )

        if file_path:
            try:
                self.video_thread = VideoThread(file_path)
                self.video_thread.frame_ready.connect(self._process_video_frame)
                self.video_thread.finished.connect(self._video_finished)
                self.video_thread.error_occurred.connect(self._video_error)
                self.video_thread.start()

                self.emotion_label.setText("Playing video...")
                self.log_text.append(f"Started video: {os.path.basename(file_path)}")
                self._save_log_to_file(f"Started video: {os.path.basename(file_path)}")
            except Exception as e:
                self.log_text.append(f"Error: Cannot play video - {str(e)}")
                self._save_log_to_file(f"Error: Cannot play video - {str(e)}")

    def _process_video_frame(self, frame):
        try:
            processed_frame, emotion, probability = self._process_frame(frame)
            self._display_frame(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            self.emotion_label.setText(f"Detected Emotion: {emotion}")

            if emotion != "No face detected" and probability is not None:
                self.depression_analyzer.update(emotion)
            risk = self.depression_analyzer.get_risk()
            self.depression_risk_label.setText(f"Risk: {risk:.1f}%")
            self.depression_progress.setValue(int(risk))

            self.update_suggestion(emotion, probability, risk)

            if probability is not None:
                self.bar_graph.set_probabilities(probability)
                self._log_recognition(emotion, probability, risk)
        except Exception as e:
            self.log_text.append(f"Video frame processing error: {str(e)}")

    def _video_finished(self):
        self.video_thread = None
        self.emotion_label.setText("Video finished")
        self._save_log_to_file("Video playback finished")

    def _video_error(self, message):
        self.log_text.append(f"Video error: {message}")
        self._save_log_to_file(f"Video error: {message}")
        self.video_thread = None

    def _log_recognition(self, emotion, probability, depression_risk=0.0):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - Emotion: {emotion}"
        if depression_risk > 0:
            log_entry += f", Depression Risk: {depression_risk:.1f}%"
        self.log_text.append(log_entry)
        self.log_text.ensureCursorVisible()
        self._save_log_to_file(log_entry)

    def _save_log_to_file(self, message):
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"Cannot write to log file: {str(e)}")

    def update_suggestion(self, emotion, probability=None, depression_risk=0.0):
        suggestions = {
            'Angry': "Anger detected.\n"
                     "1. Try relaxation techniques and deep breathing.\n"
                     "2. Focus on positive thoughts.\n"
                     "3. Maintain a calm and composed demeanor.",

            'Disgust': "Disgust detected.\n"
                       "1. Identify the source of discomfort and address it.\n"
                       "2. Shift focus to neutral or positive stimuli.\n"
                       "3. Practice acceptance and open-mindedness.",

            'Fear': "Fear/Anxiety detected.\n"
                    "1. Take slow, deep breaths to calm down.\n"
                    "2. Remind yourself of your strengths and capabilities.\n"
                    "3. Face the situation gradually with confidence.",

            'Happy': "Happiness detected!\n"
                     "1. Maintain this positive emotional state.\n"
                     "2. Share your joy with others to boost social connection.\n"
                     "3. Use this energy to accomplish tasks effectively.",

            'Sad': "Sadness detected.\n"
                   "1. Engage in activities that usually bring you joy.\n"
                   "2. Connect with friends or family for support.\n"
                   "3. Practice self-compassion and allow yourself to feel.",

            'Surprise': "Surprise detected.\n"
                        "1. Stay calm and process the unexpected event.\n"
                        "2. Assess the situation rationally before reacting.\n"
                        "3. Return to a stable emotional baseline.",

            'Neutral': "Neutral emotion detected.\n"
                       "1. You are in a stable emotional state.\n"
                       "2. Maintain this balanced mindset.\n"
                       "3. Continue monitoring for any changes.",

            'No face detected': "No face detected.\n"
                                "1. Ensure your face is clearly visible in the frame.\n"
                                "2. Adjust lighting conditions.\n"
                                "3. Check camera functionality."
        }

        suggestion = suggestions.get(emotion, "System is running normally")

        if probability is not None and emotion != "No face detected":
            max_index = np.argmax(probability)
            max_prob = probability[max_index]
            suggestion += f"\n\nConfidence: {max_prob * 100:.1f}%"

        if depression_risk > 30:
            suggestion += f"\n\n⚠ Depression Risk Alert: {depression_risk:.1f}%"
            if depression_risk > 70:
                suggestion += "\nHigh depression risk detected. Consider consulting a mental health professional."
            elif depression_risk > 50:
                suggestion += "\nModerate depression risk. Pay attention to your emotional well-being."
        elif depression_risk > 0:
            suggestion += f"\n\nDepression Risk: {depression_risk:.1f}% (low)"

        self.suggestion_label.setText(suggestion)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait(1000)
            self.video_thread = None

        self.release_camera()

        self._save_log_to_file(f"=== Recognition session ended - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    try:
        window = EmotionRecognitionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application crashed: {str(e)}")
        QMessageBox.critical(None, "System Error", f"A critical error occurred:\n{str(e)}")