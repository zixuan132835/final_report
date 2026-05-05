# -*- coding: utf-8 -*-
# @Time    : 2025/2/14 15:28
# @Author  : shaocanfan
# @File    : UI6.0_face_recognition_dlib_fatigue.py
import sys
import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import datetime
from collections import Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QFileDialog, QFrame,
                             QScrollArea, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QFontDatabase

# 确保dlib检测器加载
try:
    # 加载dlib人脸检测器和68点特征预测器
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR_PATH = "../datasets/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(PREDICTOR_PATH):
        # 尝试系统路径
        PREDICTOR_PATH = os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat")

    if os.path.exists(PREDICTOR_PATH):
        PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
        DLIB_AVAILABLE = True
    else:
        DLIB_AVAILABLE = False
        print("警告: 未找到shape_predictor_68_face_landmarks.dat，将使用传统方法")
except Exception as e:
    print(f"加载dlib失败: {e}")
    DLIB_AVAILABLE = False


class Model(nn.Module):
    """自定义CNN情绪识别模型"""

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
        self.fc3 = nn.Linear(1024, 7)  # 7种情绪

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


class DlibFatigueDetector:
    """基于dlib 68点特征的高精度疲劳检测类"""

    def __init__(self):
        self.face_tracking = {}
        self.frame_counter = 0

        # 68点特征索引定义
        self.EYE_LANDMARKS = {
            'left_eye': list(range(36, 42)),
            'right_eye': list(range(42, 48)),
            'mouth': list(range(48, 68)),
            'jaw': list(range(0, 17)),
            'eyebrows': list(range(17, 36))
        }

        # 优化的EAR/MAR阈值（基于dlib关键点）
        self.EAR_THRESHOLD = 0.25  # 眼睛纵横比阈值
        self.EAR_CONSEC_FRAMES = 3  # 连续闭眼帧数
        self.MAR_THRESHOLD = 0.6  # 嘴巴纵横比阈值
        self.MAR_CONSEC_FRAMES = 2  # 连续张嘴帧数

        # 眨眼和打哈欠计数器
        self.BLINK_THRESHOLD = 5  # 眨眼次数阈值
        self.YAWN_THRESHOLD = 2  # 打哈欠次数阈值

        # 疲劳判定参数
        self.FATIGUE_SCORE_THRESHOLD = 70  # 疲劳分数阈值
        self.SCORE_DECAY_RATE = 0.8  # 分数衰减率

    def _calculate_EAR(self, eye):
        """计算眼睛纵横比 (Eye Aspect Ratio)"""
        # 垂直距离
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        # 水平距离
        C = np.linalg.norm(eye[0] - eye[3])

        # EAR计算公式
        ear = (A + B) / (2.0 * C)
        return ear

    def _calculate_MAR(self, mouth):
        """计算嘴巴纵横比 (Mouth Aspect Ratio)"""
        # 垂直距离
        A = np.linalg.norm(mouth[13] - mouth[19])
        B = np.linalg.norm(mouth[14] - mouth[18])
        C = np.linalg.norm(mouth[15] - mouth[17])

        # 水平距离
        D = np.linalg.norm(mouth[12] - mouth[16])

        # MAR计算公式
        mar = (A + B + C) / (3.0 * D)
        return mar

    def _shape_to_np(self, shape):
        """将dlib的shape对象转换为numpy数组"""
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _get_face_id(self, face_rect):
        """生成稳定的人脸ID"""
        return f"{face_rect.left()}_{face_rect.top()}_{face_rect.width()}_{face_rect.height()}"

    def detect_fatigue_dlib(self, frame):
        """使用dlib进行高精度疲劳检测"""
        if not DLIB_AVAILABLE:
            return [], []

        self.frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用dlib检测人脸
        faces = DETECTOR(gray, 0)
        fatigue_results = []
        face_rects = []

        for face in faces:
            # 转换为opencv格式的矩形
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_rects.append((x, y, w, h))

            # 获取68点特征
            shape = PREDICTOR(gray, face)
            shape_np = self._shape_to_np(shape)

            # 获取眼睛和嘴巴区域
            left_eye = shape_np[self.EYE_LANDMARKS['left_eye']]
            right_eye = shape_np[self.EYE_LANDMARKS['right_eye']]
            mouth = shape_np[self.EYE_LANDMARKS['mouth']]

            # 计算EAR和MAR
            left_ear = self._calculate_EAR(left_eye)
            right_ear = self._calculate_EAR(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = self._calculate_MAR(mouth)

            # 获取或初始化人脸跟踪数据
            face_id = self._get_face_id(face)
            if face_id not in self.face_tracking:
                self.face_tracking[face_id] = {
                    'ear_history': [],
                    'mar_history': [],
                    'eye_closed_frames': 0,
                    'mouth_open_frames': 0,
                    'blink_count': 0,
                    'yawn_count': 0,
                    'fatigue_score': 0,
                    'status': 'Alert',
                    'last_blink': 0,
                    'last_yawn': 0,
                    'last_update': self.frame_counter
                }

            track_data = self.face_tracking[face_id]

            # 更新EAR历史
            track_data['ear_history'].append(ear)
            if len(track_data['ear_history']) > 10:
                track_data['ear_history'].pop(0)

            # 更新MAR历史
            track_data['mar_history'].append(mar)
            if len(track_data['mar_history']) > 10:
                track_data['mar_history'].pop(0)

            # 检测闭眼
            if ear < self.EAR_THRESHOLD:
                track_data['eye_closed_frames'] += 1

                # 检测眨眼（闭眼后睁眼）
                if track_data['eye_closed_frames'] == self.EAR_CONSEC_FRAMES:
                    track_data['blink_count'] += 1
                    track_data['last_blink'] = self.frame_counter
                    track_data['fatigue_score'] += 5  # 眨眼加分
            else:
                # 重置闭眼计数器
                if track_data['eye_closed_frames'] >= self.EAR_CONSEC_FRAMES:
                    pass  # 已计为眨眼
                track_data['eye_closed_frames'] = 0

            # 检测打哈欠
            if mar > self.MAR_THRESHOLD:
                track_data['mouth_open_frames'] += 1

                if track_data['mouth_open_frames'] == self.MAR_CONSEC_FRAMES:
                    track_data['yawn_count'] += 1
                    track_data['last_yawn'] = self.frame_counter
                    track_data['fatigue_score'] += 15  # 打哈欠加分
            else:
                track_data['mouth_open_frames'] = 0

            # 计算疲劳分数
            # 闭眼时间长加分
            if track_data['eye_closed_frames'] > self.EAR_CONSEC_FRAMES:
                track_data['fatigue_score'] += track_data['eye_closed_frames'] * 0.5

            # 分数衰减
            track_data['fatigue_score'] = max(0, track_data['fatigue_score'] * self.SCORE_DECAY_RATE)

            # 判断疲劳状态
            fatigue_reason = ""
            if track_data['fatigue_score'] >= self.FATIGUE_SCORE_THRESHOLD:
                track_data['status'] = 'Fatigued'

                # 分析疲劳原因
                if track_data['blink_count'] >= self.BLINK_THRESHOLD:
                    fatigue_reason = "Excessive blinking"
                elif track_data['yawn_count'] >= self.YAWN_THRESHOLD:
                    fatigue_reason = "Frequent yawning"
                elif track_data['eye_closed_frames'] > self.EAR_CONSEC_FRAMES * 2:
                    fatigue_reason = "Eyes closed too long"
                else:
                    fatigue_reason = "General fatigue"
            else:
                track_data['status'] = 'Alert'

            # 更新最后更新时间
            track_data['last_update'] = self.frame_counter

            # 绘制面部特征点（可选，用于调试）
            # for (x, y) in shape_np:
            #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 绘制眼睛和嘴巴轮廓
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

            # 添加结果
            fatigue_results.append({
                'status': track_data['status'],
                'reason': fatigue_reason,
                'score': int(track_data['fatigue_score']),
                'ear': ear,
                'mar': mar,
                'blinks': track_data['blink_count'],
                'yawns': track_data['yawn_count']
            })

        # 清理过期的人脸跟踪数据
        expired_ids = []
        for face_id, data in self.face_tracking.items():
            if self.frame_counter - data['last_update'] > 15:
                expired_ids.append(face_id)

        for face_id in expired_ids:
            del self.face_tracking[face_id]

        return fatigue_results, face_rects

    def detect_fatigue_fallback(self, frame):
        """备用的传统疲劳检测方法"""
        self.frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用opencv检测人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        fatigue_results = []
        for (x, y, w, h) in faces:
            fatigue_results.append({
                'status': 'Alert',
                'reason': 'Fallback mode',
                'score': 0,
                'ear': 0.3,
                'mar': 0.4,
                'blinks': 0,
                'yawns': 0
            })

        return fatigue_results, faces.tolist() if len(faces) > 0 else []

    def detect_fatigue(self, frame):
        """统一的疲劳检测接口"""
        if DLIB_AVAILABLE:
            return self.detect_fatigue_dlib(frame)
        else:
            return self.detect_fatigue_fallback(frame)

    def reset(self):
        """重置疲劳检测状态"""
        self.face_tracking = {}
        self.frame_counter = 0


class EmotionClassifier:
    """情绪分类器封装类（中文UI，英文标签）"""

    def __init__(self):
        self._init_face_detector()
        self._init_emotion_model()
        self._init_fatigue_detector()

        # 核心映射：中文<->英文
        self.emotion_cn_to_en = {
            '愤怒': 'Anger',
            '厌恶': 'Disgust',
            '恐惧': 'Fear',
            '高兴': 'Happy',
            '悲伤': 'Sad',
            '惊讶': 'Surprise',
            '正常': 'Neutral'
        }

        self.emotion_en_to_cn = {v: k for k, v in self.emotion_cn_to_en.items()}

        # 情绪索引映射（英文）
        self.emotion_index_to_en = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

        # 疲劳状态映射
        self.fatigue_en_to_cn = {
            'Fatigued': '疲劳',
            'Alert': '清醒',
            'Excessive blinking': '频繁眨眼',
            'Frequent yawning': '频繁打哈欠',
            'Eyes closed too long': '闭眼过久',
            'General fatigue': '整体疲劳',
            'Fallback mode': '备用模式'
        }

        # 颜色映射（英文标签）
        self.color_map = {
            'Anger': (0, 0, 255),  # Red
            'Disgust': (128, 0, 128),  # Purple
            'Fear': (0, 0, 128),  # Dark Blue
            'Happy': (0, 255, 0),  # Green
            'Sad': (255, 0, 0),  # Blue
            'Surprise': (255, 165, 0),  # Orange
            'Neutral': (128, 128, 128),  # Gray
            'Fatigued': (0, 255, 255),  # Yellow
            'Alert': (0, 128, 0)  # Dark Green
        }

    def _init_face_detector(self):
        cascade_path = os.path.join('./datasets/haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _init_emotion_model(self):
        model_path = os.path.join('./model/model_params.pkl')
        if os.path.exists(model_path):
            self.model = Model()
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
        else:
            print(f"警告: 未找到模型文件 {model_path}")
            self.model = None

    def _init_fatigue_detector(self):
        """初始化高精度疲劳检测器"""
        self.fatigue_detector = DlibFatigueDetector()

    def get_emotion_batch(self, face_images):
        """批量获取多个人脸的情绪分类结果（返回英文标签）"""
        if not face_images or self.model is None:
            return [], []

        inputs = []
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        for face_img in face_images:
            img = trans(face_img)
            inputs.append(img)

        inputs = torch.stack(inputs)

        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1).detach().numpy()

        # 返回英文标签
        emotions = [self.emotion_index_to_en[pred.item()] for pred in predicted]
        probabilities_list = [prob.flatten() for prob in probabilities]

        return emotions, probabilities_list


class VideoThread(QThread):
    """优化的视频播放线程"""
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._stop_flag = False
        self.mutex = QMutex()
        self.frame_skip = 0  # 不跳帧，保证检测精度

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"无法打开视频文件: {self.video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 30

            while not self._stop_flag:
                self.mutex.lock()
                ret, frame = cap.read()
                self.mutex.unlock()

                if not ret:
                    break

                self.frame_ready.emit(frame.copy())
                self.msleep(max(1, delay))

        except Exception as e:
            self.error_occurred.emit(f"视频处理错误: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            self.finished.emit()

    def stop(self):
        self.mutex.lock()
        self._stop_flag = True
        self.mutex.unlock()
        self.wait(1000)


class BarGraphWidget(QWidget):
    """自定义柱状图小部件（中英文混合显示）"""

    def __init__(self):
        super().__init__()
        self.probabilities = None
        self.setMinimumSize(300, 220)

        # 中文标签 + 英文映射
        self.emotion_labels_cn = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '正常']
        self.emotion_labels_en = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.bar_colors = [
            QColor(231, 76, 60),  # 愤怒/Anger
            QColor(142, 68, 173),  # 厌恶/Disgust
            QColor(41, 128, 185),  # 恐惧/Fear
            QColor(39, 174, 96),  # 高兴/Happy
            QColor(44, 62, 80),  # 悲伤/Sad
            QColor(243, 156, 18),  # 惊讶/Surprise
            QColor(149, 165, 166)  # 正常/Neutral
        ]

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
        bar_height = 28
        spacing = 6
        text_width = 80

        max_prob = max(self.probabilities) if max(self.probabilities) > 0 else 1

        # 绘制每个柱状图（中文标签 + 英文标注）
        for i, (prob, label_cn, label_en, color) in enumerate(zip(
                self.probabilities, self.emotion_labels_cn, self.emotion_labels_en, self.bar_colors)):
            y = margin + i * (bar_height + spacing)

            # 绘制背景
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(60, 60, 60))
            painter.drawRoundedRect(text_width, y, width - text_width - margin, bar_height, 3, 3)

            # 绘制柱状图
            bar_width = int((width - text_width - margin - 10) * (prob / max_prob))
            painter.setBrush(color)
            painter.drawRoundedRect(text_width, y, bar_width, bar_height, 3, 3)

            # 绘制中文标签 + 英文标注
            painter.setPen(QColor(220, 220, 220))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            # 中文标签
            painter.drawText(5, y + bar_height - 6, label_cn)
            # 英文标注（小号）
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(45, y + bar_height - 6, f"({label_en})")

            # 百分比
            percentage_text = f"{prob * 100:.1f}%"
            painter.drawText(width - 60, y + bar_height - 6, percentage_text)


class EmotionRecognitionApp(QMainWindow):
    """人脸情绪识别系统（中文UI，英文标签）"""

    def __init__(self):
        super().__init__()

        # 首先初始化互斥锁，确保所有方法都能使用
        self.ui_mutex = QMutex()
        self.camera_mutex = QMutex()

        # 窗口初始化（中文标题）
        self.setWindowTitle("实时多模态人脸情绪识别系统")
        self.setMinimumSize(1400, 900)

        # 先初始化UI组件
        self._init_ui()

        # 然后初始化日志文件（需要log_text已存在）
        self._init_log_file()

        # 初始化情绪分类器
        self._init_classifier()

        # 状态变量
        self.cap = None
        self.camera_active = False
        self.is_recognizing = False
        self.video_thread = None

        # 定时器（实时检测）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _init_log_file(self):
        """初始化日志文件"""
        self.log_file_path = "face_emotion_fatigue_recognition_dlib_log.txt"
        # 创建日志文件
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== 人脸情绪与疲劳识别日志 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Dlib状态: {'已加载' if DLIB_AVAILABLE else '未加载'}\n")

            # 记录初始化日志
            self._log("系统初始化完成")
        except Exception as e:
            print(f"初始化日志文件失败: {str(e)}")

    def _init_classifier(self):
        """初始化情绪分类器"""
        try:
            self.classifier = EmotionClassifier()
            self._log(f"Dlib面部特征检测: {'已启用' if DLIB_AVAILABLE else '未启用（使用备用模式）'}")
        except Exception as e:
            error_msg = f"初始化失败: {str(e)}"
            self._log(error_msg)
            self._show_error(error_msg)

    def _init_ui(self):
        """初始化UI（中文界面）"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 标题（中文）
        title_label = QLabel("实时多模态人脸情绪识别系统")
        title_label.setStyleSheet("""
            font-size: 26px; 
            font-weight: bold; 
            color: #3498db;
            padding: 12px;
            background-color: #2c3e50;
            border-radius: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        main_layout.addWidget(title_label)

        # 状态提示
        dlib_status = "已加载（高精度模式）" if DLIB_AVAILABLE else "未加载（基础模式）"
        status_label = QLabel(f"Dlib 68点特征检测器: {dlib_status}")
        status_label.setStyleSheet("""
            font-size: 14px; 
            color: #e74c3c;
            padding: 5px;
            background-color: #1e1e1e;
            border-radius: 5px;
        """)
        status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(status_label)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧面板 - 视频显示
        video_panel = QFrame()
        video_panel.setFrameShape(QFrame.StyledPanel)
        video_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px; padding: 5px;")
        video_panel.setMinimumWidth(750)
        video_panel.setMinimumHeight(550)

        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(5, 5, 5, 5)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; border-radius: 3px;")
        self.video_label.setMinimumSize(730, 530)
        video_layout.addWidget(self.video_label)

        # 中间面板 - 结果显示
        results_panel = QFrame()
        results_panel.setFrameShape(QFrame.StyledPanel)
        results_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px; padding: 5px;")
        results_panel.setMinimumWidth(350)
        results_panel.setMaximumWidth(380)

        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_layout.setSpacing(12)

        # 结果标题（中文）
        result_title = QLabel("多人情绪与疲劳检测结果（高精度）")
        result_title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #ffffff;
            padding: 5px;
        """)
        result_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(result_title)

        # 检测到的人数（中文）
        self.person_count_label = QLabel("检测到人数: 0")
        self.person_count_label.setStyleSheet("""
            font-size: 16px; 
            color: #4a90e2;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 8px;
            text-align: center;
        """)
        self.person_count_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.person_count_label)

        # 当前主要情绪（中英文显示）
        self.main_emotion_label = QLabel("主要情绪: 等待检测...")
        self.main_emotion_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            color: #4a90e2;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
        """)
        self.main_emotion_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.main_emotion_label)

        # 疲劳状态统计（中文+英文）
        self.fatigue_status_label = QLabel("疲劳状态: 无")
        self.fatigue_status_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            color: #e74c3c;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
        """)
        self.fatigue_status_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.fatigue_status_label)

        # 疲劳分数显示
        self.fatigue_score_label = QLabel("疲劳分数: 0/100")
        self.fatigue_score_label.setStyleSheet("""
            font-size: 16px; 
            color: #f39c12;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 8px;
            text-align: center;
        """)
        self.fatigue_score_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.fatigue_score_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #3a3a3a; margin: 5px 0;")
        results_layout.addWidget(separator)

        # 概率柱状图（中文标签+英文标注）
        self.bar_graph = BarGraphWidget()
        self.bar_graph.setMinimumSize(320, 260)
        results_layout.addWidget(self.bar_graph)

        # 右侧面板 - 日志（中文）
        log_panel = QFrame()
        log_panel.setFrameShape(QFrame.StyledPanel)
        log_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px; padding: 5px;")
        log_panel.setMinimumWidth(350)

        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.setSpacing(10)

        # 日志标题（中文）
        log_title = QLabel("识别日志（含高精度疲劳检测）")
        log_title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #ffffff;
            padding: 5px;
        """)
        log_title.setAlignment(Qt.AlignCenter)
        log_layout.addWidget(log_title)

        # 日志文本区域
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
                margin: 0px 0px 0px 0px;
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
            padding: 8px;
            font-size: 11px;
            border: none;
        """)
        self.log_text.setMinimumHeight(650)
        scroll_area.setWidget(self.log_text)
        log_layout.addWidget(scroll_area)

        # 添加面板到布局
        content_layout.addWidget(video_panel)
        content_layout.addWidget(results_panel)
        content_layout.addWidget(log_panel)
        main_layout.addLayout(content_layout)

        # 底部按钮面板（中文按钮）
        button_panel = QFrame()
        button_panel.setStyleSheet("background-color: transparent;")
        button_panel.setMinimumHeight(60)

        button_layout = QHBoxLayout(button_panel)
        button_layout.setContentsMargins(20, 10, 20, 10)
        button_layout.setSpacing(15)

        # 中文按钮
        self.toggle_btn = self._create_button("开始识别", "#4a90e2", self.toggle_recognition)
        self.image_btn = self._create_button("图片检测", "#7ed321", self.open_image)
        self.video_btn = self._create_button("视频检测", "#f5a623", self.open_video)
        self.reset_btn = self._create_button("重置疲劳状态", "#9b59b6", self.reset_fatigue_status)
        self.quit_btn = self._create_button("退出系统", "#d0021b", self.close)

        # 调整按钮大小
        for btn in [self.toggle_btn, self.image_btn, self.video_btn, self.reset_btn, self.quit_btn]:
            btn.setFixedHeight(45)
            btn.setMinimumWidth(140)

        button_layout.addWidget(self.toggle_btn)
        button_layout.addWidget(self.image_btn)
        button_layout.addWidget(self.video_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.quit_btn)

        main_layout.addWidget(button_panel)

    def _create_button(self, text, color, callback):
        """创建中文按钮"""
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 15px;
                font-weight: bold;
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
        """将十六进制颜色变暗"""
        color = QColor(hex_color)
        h, s, l, a = color.getHslF()
        l = max(0, l - (percent / 100))
        return QColor.fromHslF(h, s, l, a).name()

    def _show_error(self, message):
        """显示错误信息（中文）"""
        QMessageBox.critical(self, "系统错误", message)

    def init_camera(self):
        """初始化摄像头"""
        if self.cap is None:
            try:
                cv2.destroyAllWindows()
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                if not self.cap.isOpened():
                    self._log("错误: 无法打开摄像头")
                    return False
                self.camera_active = True
                self._log("摄像头初始化成功")
                return True
            except Exception as e:
                self._log(f"摄像头初始化错误: {str(e)}")
                return False
        return True

    def release_camera(self):
        """释放摄像头资源"""
        self.ui_mutex.lock()
        try:
            if self.cap is not None:
                self.timer.stop()
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                    cv2.destroyAllWindows()
                    self._log("摄像头资源已释放")
                except Exception as e:
                    self._log(f"释放摄像头错误: {str(e)}")
                finally:
                    self.cap = None
                    self.camera_active = False
                    self.timer.start(30)
        finally:
            self.ui_mutex.unlock()

    def reset_fatigue_status(self):
        """重置疲劳检测状态"""
        if self.classifier and hasattr(self.classifier, 'fatigue_detector'):
            self.classifier.fatigue_detector.reset()
            self.fatigue_status_label.setText("疲劳状态: 已重置")
            self.fatigue_score_label.setText("疲劳分数: 0/100")
            self._log("高精度疲劳检测状态已重置")

            # 3秒后恢复显示
            QTimer.singleShot(3000, lambda: self.fatigue_status_label.setText("疲劳状态: 无"))

    def update_frame(self):
        """实时更新帧并检测"""
        if not self.is_recognizing:
            return

        self.camera_mutex.lock()
        try:
            if not self.camera_active:
                if not self.init_camera():
                    self.is_recognizing = False
                    self.toggle_btn.setText("开始识别")
                    return

            ret, frame = self.cap.read()
            if not ret:
                self._log("错误: 无法从摄像头读取帧")
                self.release_camera()
                self.is_recognizing = False
                self.toggle_btn.setText("开始识别")
                return

            # 镜像翻转
            frame = cv2.flip(frame, 1)

            # 检测处理
            processed_frame, emotions, probabilities_list, fatigue_results = self._detect_faces_emotions_fatigue(frame)

            # 显示处理后的帧
            self._display_frame(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            # 更新结果显示
            self.ui_mutex.lock()
            try:
                self._update_multi_face_results(emotions, probabilities_list, fatigue_results)
            finally:
                self.ui_mutex.unlock()

            # 记录日志
            if emotions:
                self._log_emotion_fatigue_results(emotions, probabilities_list, fatigue_results)

        except Exception as e:
            self._log(f"帧更新错误: {str(e)}")
        finally:
            self.camera_mutex.unlock()

    def _detect_faces_emotions_fatigue(self, frame):
        """使用dlib检测人脸、识别情绪和检测疲劳（英文标签）"""
        try:
            # 高精度疲劳检测
            fatigue_results, face_rects = self.classifier.fatigue_detector.detect_fatigue(frame)

            emotions = []
            probabilities_list = []

            if len(face_rects) > 0:
                face_images = []

                # 处理每个人脸
                for i, (x, y, w, h) in enumerate(face_rects):
                    # 边界检查
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    if w < 40 or h < 40:
                        continue

                    # 提取人脸区域
                    face_roi = frame[y:y + h, x:x + w]
                    face_img = cv2.resize(face_roi, (42, 42))
                    face_images.append(Image.fromarray(face_img))

                # 批量识别情绪
                if face_images:
                    emotions, probabilities_list = self.classifier.get_emotion_batch(face_images)

                # 绘制标注（英文标签，解决乱码）
                for i, ((x, y, w, h), emotion) in enumerate(zip(face_rects, emotions)):
                    # 获取疲劳状态
                    if i < len(fatigue_results):
                        fatigue_data = fatigue_results[i]
                        fatigue_state = fatigue_data['status']
                        fatigue_reason = fatigue_data['reason']
                        fatigue_score = fatigue_data['score']
                    else:
                        fatigue_state = "Alert"
                        fatigue_reason = ""
                        fatigue_score = 0

                    # 选择边框颜色
                    if fatigue_state == "Fatigued":
                        border_color = self.classifier.color_map["Fatigued"]
                    else:
                        border_color = self.classifier.color_map.get(emotion, (0, 255, 0))

                    # 绘制矩形框（更粗的边框）
                    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 3)

                    # 计算标签位置（避免越界）
                    label_y = y - 10 if y - 10 > 10 else y + h + 40
                    label_y = min(label_y, frame.shape[0] - 40)
                    label_x = max(10, x)

                    # 设置抗锯齿字体
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # 情绪标签（英文）
                    emotion_text = f"Face {i + 1}: {emotion}"
                    (text_w, text_h), _ = cv2.getTextSize(emotion_text, font, 0.7, 2)
                    # 绘制背景框
                    cv2.rectangle(frame, (label_x - 2, label_y - text_h - 5),
                                  (label_x + text_w + 8, label_y + 5), (0, 0, 0), -1)
                    # 绘制文字（抗锯齿）
                    cv2.putText(frame, emotion_text, (label_x + 3, label_y),
                                font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    # 疲劳状态标签（英文）
                    fatigue_text = f"State: {fatigue_state}"
                    if fatigue_reason:
                        fatigue_text += f" ({fatigue_reason})"
                    (fatigue_w, fatigue_h), _ = cv2.getTextSize(fatigue_text, font, 0.5, 1)
                    cv2.rectangle(frame, (label_x - 2, label_y + fatigue_h + 5),
                                  (label_x + fatigue_w + 8, label_y + fatigue_h + 25), (0, 0, 0), -1)
                    cv2.putText(frame, fatigue_text, (label_x + 3, label_y + fatigue_h + 20),
                                font, 0.5, self.classifier.color_map.get(fatigue_state, (255, 255, 0)), 1, cv2.LINE_AA)

                    # 疲劳分数
                    score_text = f"Fatigue: {fatigue_score}/100"
                    cv2.putText(frame, score_text, (label_x + 3, label_y + fatigue_h + 40),
                                font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                    # 置信度
                    if i < len(probabilities_list):
                        confidence = max(probabilities_list[i]) * 100
                        conf_text = f"Confidence: {confidence:.1f}%"
                        cv2.putText(frame, conf_text, (label_x + 3, label_y + fatigue_h + 60),
                                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            return frame, emotions, probabilities_list, fatigue_results
        except Exception as e:
            self._log(f"人脸检测错误: {str(e)}")
            return frame, [], [], []

    def _update_multi_face_results(self, emotions, probabilities_list, fatigue_results):
        """更新结果显示（中英文混合）"""
        num_faces = len(emotions)
        self.person_count_label.setText(f"检测到人数: {num_faces}")

        if num_faces == 0:
            self.main_emotion_label.setText("主要情绪: 未检测到人脸")
            self.fatigue_status_label.setText("疲劳状态: 未检测到人脸")
            self.fatigue_score_label.setText("疲劳分数: 0/100")
            self.bar_graph.set_probabilities(None)
            return

        # 统计主要情绪（中英文显示）
        emotion_counter = Counter(emotions)
        main_emotion_en = emotion_counter.most_common(1)[0][0]
        main_emotion_cn = self.classifier.emotion_en_to_cn.get(main_emotion_en, main_emotion_en)
        self.main_emotion_label.setText(f"主要情绪: {main_emotion_cn} ({main_emotion_en})")

        # 统计疲劳状态（中文显示）
        if fatigue_results and len(fatigue_results) > 0:
            fatigue_states = [result['status'] for result in fatigue_results]
            fatigue_count = fatigue_states.count("Fatigued")

            # 计算平均疲劳分数
            total_score = sum([result['score'] for result in fatigue_results])
            avg_score = int(total_score / len(fatigue_results)) if len(fatigue_results) > 0 else 0

            if fatigue_count > 0:
                self.fatigue_status_label.setText(f"疲劳状态: {fatigue_count}/{num_faces} 人疲劳 (Fatigued)")
                self.fatigue_score_label.setText(f"平均疲劳分数: {avg_score}/100")
            else:
                self.fatigue_status_label.setText(f"疲劳状态: 所有人清醒 (All Alert)")
                self.fatigue_score_label.setText(f"平均疲劳分数: {avg_score}/100")
        else:
            self.fatigue_status_label.setText("疲劳状态: 无法检测 (Undetected)")
            self.fatigue_score_label.setText("疲劳分数: 0/100")

        # 更新柱状图
        avg_probabilities = np.zeros(7)
        count = 0

        for prob in probabilities_list:
            avg_probabilities += prob
            count += 1

        if count > 0:
            avg_probabilities /= count
            self.bar_graph.set_probabilities(avg_probabilities)

    def _display_frame(self, frame):
        """显示帧（优化缩放）"""
        try:
            h, w, ch = frame.shape
            label_size = self.video_label.size()

            # 保持宽高比缩放
            scale = min(label_size.width() / w, label_size.height() / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w > 0 and new_h > 0:
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                # 确保数据连续
                if not scaled_frame.flags['C_CONTIGUOUS']:
                    scaled_frame = np.ascontiguousarray(scaled_frame)

                # 转换为QImage
                q_img = QImage(scaled_frame.data, new_w, new_h, new_w * ch, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                # 平滑缩放显示
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
        except Exception as e:
            self._log(f"帧显示错误: {str(e)}")

    def _log(self, message):
        """添加中文日志"""
        self.ui_mutex.lock()
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"

            # 检查log_text是否存在
            if hasattr(self, 'log_text') and self.log_text is not None:
                self.log_text.append(log_entry)
                self.log_text.ensureCursorVisible()
            else:
                print(f"日志（log_text未初始化）: {log_entry}")

            # 保存到文件
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"记录日志错误: {str(e)}")
        finally:
            self.ui_mutex.unlock()

    def _log_emotion_fatigue_results(self, emotions, probabilities_list, fatigue_results):
        """记录识别结果（中英文）"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] 检测到{len(emotions)}个人脸: "

        for i, (emotion, prob) in enumerate(zip(emotions, probabilities_list)):
            confidence = max(prob) * 100
            # 中英文对照显示
            emotion_cn = self.classifier.emotion_en_to_cn.get(emotion, emotion)

            # 疲劳状态
            if i < len(fatigue_results):
                fatigue_data = fatigue_results[i]
                fatigue_state = fatigue_data['status']
                fatigue_reason = fatigue_data['reason']
                fatigue_score = fatigue_data['score']

                fatigue_cn = self.classifier.fatigue_en_to_cn.get(fatigue_state, fatigue_state)
                reason_cn = self.classifier.fatigue_en_to_cn.get(fatigue_reason, fatigue_reason)
            else:
                fatigue_state, fatigue_reason, fatigue_score = ("Alert", "", 0)
                fatigue_cn = "清醒"
                reason_cn = ""

            log_entry += f"人脸{i + 1}: {emotion_cn}({emotion})/{fatigue_cn}({fatigue_state})[分数:{fatigue_score}] "
            if reason_cn:
                log_entry += f"[{reason_cn}] "

        self.ui_mutex.lock()
        try:
            if hasattr(self, 'log_text') and self.log_text is not None:
                self.log_text.append(log_entry)
                self.log_text.ensureCursorVisible()

            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        finally:
            self.ui_mutex.unlock()

    def toggle_recognition(self):
        """切换识别状态"""
        self.is_recognizing = not self.is_recognizing

        if self.is_recognizing:
            self.toggle_btn.setText("暂停识别")
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
        else:
            self.toggle_btn.setText("开始识别")
            self.release_camera()

    def open_image(self):
        """图片检测（中文）"""
        if self.is_recognizing:
            self.toggle_recognition()

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif *.jfif)"
        )

        if file_path:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    raise ValueError(f"文件不存在: {file_path}")

                # 检查文件大小
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    raise ValueError("文件为空")

                # 检查文件扩展名
                valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.jfif']
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext not in valid_extensions:
                    raise ValueError(f"不支持的图片格式: {file_ext}")

                self._log(f"正在读取图片: {file_path}")
                self._log(f"文件大小: {file_size} bytes")
                self._log(f"文件格式: {file_ext}")

                # 方法1：尝试使用OpenCV读取
                self._log("尝试使用OpenCV读取图片...")
                try:
                    # 尝试不同的读取方式
                    image = cv2.imread(file_path)

                    if image is None:
                        # 尝试使用numpy读取
                        img_array = np.fromfile(file_path, dtype=np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if image is None:
                        raise Exception("OpenCV无法读取图片")

                    self._log(f"OpenCV读取成功，尺寸: {image.shape}")
                except Exception as cv_error:
                    self._log(f"OpenCV读取失败: {str(cv_error)}")
                    image = None

                # 方法2：如果OpenCV失败，使用PIL读取
                if image is None:
                    self._log("尝试使用PIL读取图片...")
                    try:
                        pil_image = Image.open(file_path)

                        # 转换为RGB（如果是RGBA）
                        if pil_image.mode == 'RGBA':
                            pil_image = pil_image.convert('RGB')
                        elif pil_image.mode == 'L':
                            pil_image = pil_image.convert('RGB')

                        # 将PIL图像转换为OpenCV格式
                        image_np = np.array(pil_image)

                        # 确保是3通道的RGB图像
                        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        else:
                            # 如果是灰度图
                            image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

                        self._log(f"PIL读取成功，原始模式: {pil_image.mode}，转换后尺寸: {image.shape}")
                    except Exception as pil_error:
                        self._log(f"PIL读取失败: {str(pil_error)}")
                        raise ValueError(f"所有方法均无法读取图片: {str(pil_error)}")

                if image is None:
                    raise ValueError("无法读取图片文件")

                # 检查图片尺寸和显示信息
                h, w = image.shape[:2]
                channels = image.shape[2] if len(image.shape) == 3 else 1
                self._log(f"最终图片尺寸: {w}x{h} 像素，通道数: {channels}")

                # 显示图片的简要信息
                preview_info = f"已加载图片: {os.path.basename(file_path)} ({w}x{h})"
                self._log(preview_info)

                # 限制图片尺寸，避免过大导致性能问题
                max_dimension = 2000
                if w > max_dimension or h > max_dimension:
                    scale = max_dimension / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    self._log(f"图片已缩放至: {new_w}x{new_h} 像素")

                # 检测处理
                self._log("开始人脸检测和情绪分析...")
                processed_image, emotions, probabilities_list, fatigue_results = self._detect_faces_emotions_fatigue(
                    image)

                # 显示处理后的帧
                self._display_frame(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

                # 更新结果显示
                self._update_multi_face_results(emotions, probabilities_list, fatigue_results)

                # 记录日志
                detection_summary = f"图片检测完成: {os.path.basename(file_path)} - 检测到 {len(emotions)} 张人脸"
                self._log(detection_summary)

                if emotions:
                    self._log_emotion_fatigue_results(emotions, probabilities_list, fatigue_results)
                else:
                    self._log("图片中未检测到人脸")

            except Exception as e:
                error_msg = f"图片处理错误: {str(e)}"
                self._log(error_msg)

                # 显示详细的错误信息
                detailed_error = f"""
图片处理失败：

文件: {os.path.basename(file_path)}
路径: {file_path}
大小: {os.path.getsize(file_path) if os.path.exists(file_path) else '文件不存在'} bytes

错误详情: {str(e)}

可能的原因：
1. 图片文件已损坏
2. 不支持的图片格式
3. 内存不足
4. OpenCV/PIL库版本不兼容

解决方案：
1. 尝试使用其他图片
2. 确保图片格式为 JPG, PNG, BMP 等常见格式
3. 检查Python库是否安装完整
                """

                # 在主线程显示错误消息
                QTimer.singleShot(100, lambda: self._show_error(detailed_error))

                # 尝试显示一个占位符图像
                try:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "图片读取失败", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self._display_frame(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB))
                except:
                    pass

    def open_video(self):
        """视频检测（优化版，解决标注框乱码）"""
        if self.is_recognizing:
            self.toggle_recognition()

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.webm)"
        )

        if file_path:
            try:
                # 使用优化的视频线程
                self.video_thread = VideoThread(file_path)
                self.video_thread.frame_ready.connect(self._process_video_frame)
                self.video_thread.finished.connect(self._video_finished)
                self.video_thread.error_occurred.connect(self._video_error)
                self.video_thread.start()

                self._log(f"开始播放视频: {os.path.basename(file_path)}")

            except Exception as e:
                error_msg = f"视频处理错误: {str(e)}"
                self._log(error_msg)
                self._show_error(error_msg)

    def _process_video_frame(self, frame):
        """处理视频帧（优化标注稳定性）"""
        try:
            # 对视频帧进行稳定化处理
            frame = cv2.GaussianBlur(frame, (1, 1), 0)  # 轻微模糊减少噪声

            processed_frame, emotions, probabilities_list, fatigue_results = self._detect_faces_emotions_fatigue(frame)
            self._display_frame(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            self._update_multi_face_results(emotions, probabilities_list, fatigue_results)

            # 降低日志频率，避免刷屏
            if hasattr(self, '_last_video_log'):
                if datetime.datetime.now().timestamp() - self._last_video_log > 1.0:
                    if emotions:
                        self._log_emotion_fatigue_results(emotions, probabilities_list, fatigue_results)
                    self._last_video_log = datetime.datetime.now().timestamp()
            else:
                self._last_video_log = datetime.datetime.now().timestamp()

        except Exception as e:
            self._log(f"视频帧处理错误: {str(e)}")

    def _video_finished(self):
        """视频播放结束"""
        self.video_thread = None
        self._log("视频播放结束")

    def _video_error(self, message):
        """视频错误处理"""
        self._log(f"视频错误: {message}")
        self.video_thread = None

    def closeEvent(self, event):
        """关闭时清理资源"""
        self.timer.stop()

        # 停止视频线程
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait(1000)

        # 释放摄像头
        self.release_camera()

        # 关闭所有窗口
        cv2.destroyAllWindows()

        # 记录结束日志
        self._log("=== 识别会话结束 ===")

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 高DPI支持
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 运行应用
    try:
        window = EmotionRecognitionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序崩溃: {str(e)}")
        QMessageBox.critical(None, "系统错误", f"应用程序发生严重错误:\n{str(e)}")
        sys.exit(1)