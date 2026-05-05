# -*- coding:UTF-8 -*-
from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import datetime


class Model(nn.Module):
    """自定义CNN情绪识别模型"""

    def __init__(self):
        super(Model, self).__init__()
        # 输入数据批归一化
        self.bn_x = nn.BatchNorm2d(1)

        # 第一个卷积层: 1个输入通道, 32个输出通道, 5x5卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(32, momentum=0.5)

        # 第二个卷积层: 32个输入通道, 32个输出通道, 4x4卷积核
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32, momentum=0.5)

        # 第三个卷积层: 32个输入通道, 64个输出通道, 5x5卷积核
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64, momentum=0.5)

        # 全连接层
        self.fc1 = nn.Linear(5 * 5 * 64, 2048)  # 输入特征维度: 5*5*64
        self.bn_fc1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.fc3 = nn.Linear(1024, 7)  # 输出7种情绪

    def forward(self, x):
        """前向传播"""
        # 卷积层部分
        x = self.bn_x(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(self.conv1(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.relu(self.bn_conv2(self.conv2(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.relu(self.bn_conv3(self.conv3(x))), kernel_size=3, stride=2, ceil_mode=True)

        # 展平特征图
        x = x.view(-1, self.num_flat_features(x))

        # 全连接层部分
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """计算特征图展平后的维度"""
        size = x.size()[1:]  # 获取除batch维度外的其他维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EmotionClassifier:
    """情绪分类器封装类"""

    def __init__(self):
        # 初始化人脸检测器
        self._init_face_detector()
        # 初始化情绪识别模型
        self._init_emotion_model()

    def _init_face_detector(self):
        """初始化OpenCV人脸检测器"""
        # 构建人脸检测模型路径
        cascade_path = os.path.join('./model/haarcascade_frontalface_default.xml')

        # 检查模型文件是否存在
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"找不到人脸检测模型文件: {cascade_path}")

        # 加载人脸检测模型
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise (" ValueError加载人脸检测模型失败，请检查文件是否正确")

    def _init_emotion_model(self):
        """初始化情绪识别模型"""
        # 构建模型文件路径
        model_path = os.path.join('./model/model_params.pkl')

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到情绪识别模型文件: {model_path}")

        # 创建模型实例并加载预训练权重
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()  # 设置为评估模式

    def get_emotion(self, inputs):
        """
        获取输入图像的情绪分类结果
        : inputsparam: PIL.Image格式的输入图像
        :return: (情绪类别, 各类别概率)
        """
        # 图像预处理
        inputs = self.preprocess(inputs)

        # 模型预测
        with torch.no_grad():  # 禁用梯度计算
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            probability = F.softmax(outputs, dim=1).detach().numpy().flatten()

        # 情绪类别映射
        emotion_map = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprised',
            6: 'normal'
        }
        return emotion_map[predicted.item()], probability

    def preprocess(self, inputs):
        """图像预处理"""
        trans = transforms.Compose([
            transforms.Grayscale(),  # 转为灰度图
            transforms.ToTensor(),  # 转为Tensor
        ])
        inputs = trans(inputs)
        inputs = inputs.unsqueeze(0)  # 增加batch维度
        return inputs


class App:
    """主应用程序GUI类"""

    def __init__(self, video_source=0):
        # 初始化主窗口
        self._init_window()

        # 初始化视频源
        self.video_source = video_source
        self.is_recognizing = False  # 添加识别状态标志
        self.current_image = None  # 当前显示的图片

        # 初始化情绪分类器
        self.emo_cls = EmotionClassifier()

        # 初始化识别日志
        self.log_file = self._init_log_file()

        # 初始化UI组件
        self._init_ui()

        # 开始视频更新循环
        self.delay = 15  # 更新间隔(毫秒)
        self.update_flag = False  # 添加更新标志
        self.update()

        # 窗口居中显示
        self.center_window()

        # 启动主事件循环
        self.window.mainloop()

    def _init_window(self):
        """初始化主窗口设置"""
        self.window = Tk()
        self.window.title('情绪识别系统')
        self.window.configure(bg='#f5f5f5')  # 设置背景色
        self.window.minsize(1000, 600)  # 设置最小窗口尺寸

    def _init_ui(self):
        """初始化用户界面组件"""
        # 设置字体
        self.title_font = ('Microsoft YaHei', 20, 'bold')  # 使用微软雅黑字体
        self.button_font = ('Microsoft YaHei', 14)
        self.result_font = ('Microsoft YaHei', 18)
        self.bar_font = ('Microsoft YaHei', 10)

        # 主容器框架
        self.main_frame = Frame(self.window, bg='#f5f5f5', padx=20, pady=20)
        self.main_frame.pack(expand=True, fill=BOTH)

        # 视频显示区域
        self._init_video_frame()

        # 结果展示区域
        self._init_result_frame()

        # 概率条区域
        self._init_bar_frame()

        # 添加日志区域
        self._init_log_frame()

        # 按钮区域
        self._init_button_frame()

        # 配置网格权重
        self._configure_grid()

    def _init_video_frame(self):
        """初始化视频显示区域"""
        self.video_frame = Frame(
            self.main_frame,
            bg='white',
            bd=2,
            relief=GROOVE,
            highlightbackground='#e0e0e0',
            highlightthickness=1
        )
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky='nsew')

        self.canvas = Canvas(
            self.video_frame,
            width=640,
            height=480,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack(padx=5, pady=5)

    def _init_result_frame(self):
        """初始化结果展示区域"""
        self.result_frame = Frame(
            self.main_frame,
            bg='white',
            bd=2,
            relief=GROOVE,
            highlightbackground='#e0e0e0',
            highlightthickness=1
        )
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # 情绪结果标签
        self.results_label = Label(
            self.result_frame,
            text='等待检测...',
            font=self.title_font,
            bg='white',
            fg='#333333',
            pady=20
        )
        self.results_label.pack()

    def _init_bar_frame(self):
        """初始化概率条区域"""
        self.bar_frame = Frame(
            self.main_frame,
            bg='white',
            bd=2,
            relief=GROOVE,
            highlightbackground='#e0e0e0',
            highlightthickness=1
        )
        self.bar_frame.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.bar_canvas = Canvas(
            self.bar_frame,
            width=300,
            height=300,
            bg='white',
            highlightthickness=0
        )
        self.bar_canvas.pack(padx=10, pady=10)

    def _init_log_frame(self):
        """初始化日志区域"""
        self.log_frame = Frame(
            self.main_frame,
            bg='white',
            bd=2,
            relief=GROOVE,
            highlightbackground='#e0e0e0',
            highlightthickness=1
        )
        self.log_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10, sticky='nsew')

        self.log_text = Text(
            self.log_frame,
            width=30,
            height=25,
            bg='white',
            fg='#333333',
            font=('Microsoft YaHei', 10),
            wrap=WORD
        )
        self.log_text.pack(padx=5, pady=5, fill=BOTH, expand=True)
        self.log_text.insert(END, "日志记录: \n")

    def _init_button_frame(self):
        """初始化按钮区域"""
        self.button_frame = Frame(self.main_frame, bg='#f5f5f5')
        self.button_frame.grid(row=2, column=0, columnspan=3, pady=20)

        # 开始/暂停按钮
        self.toggle_button = Button(
            self.button_frame,
            text='视频检测',
            font=self.button_font,
            command=self.toggle_recognition,
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60',
            activeforeground='white',
            relief=FLAT,
            padx=20,
            pady=10,
            bd=0
        )
        self.toggle_button.pack(side=LEFT, padx=10)

        # 图片检测按钮
        self.image_button = Button(
            self.button_frame,
            text='图片检测',
            font=self.button_font,
            command=self.open_image,
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief=FLAT,
            padx=20,
            pady=10,
            bd=0
        )
        self.image_button.pack(side=LEFT, padx=10)

        # 退出按钮
        self.quit_button = Button(
            self.button_frame,
            text='退出系统',
            font=self.button_font,
            command=self.window.quit,
            bg='#ff6b6b',
            fg='white',
            activebackground='#ff5252',
            activeforeground='white',
            relief=FLAT,
            padx=20,
            pady=10,
            bd=0
        )
        self.quit_button.pack(side=RIGHT, padx=10)

    def _configure_grid(self):
        """配置网格布局权重"""
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)

    def center_window(self):
        """使窗口居中显示"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def update(self):
        """更新视频帧和检测结果"""
        if self.update_flag and self.is_recognizing:
            ret, frame = self.video_source.read()
            if ret:
                # 水平翻转视频帧(镜像效果)
                frame = cv2.flip(frame, 1)

                # 将OpenCV图像转换为PIL格式并显示
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rendered_img = ImageTk.PhotoImage(img)
                self.canvas.img = rendered_img  # 保持引用防止被垃圾回收
                self.canvas.create_image(0, 0, anchor=NW, image=rendered_img)

                # 转换为灰度图进行人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.emo_cls.face_cascade.detectMultiScale(gray, 1.3, 5)

                # 检测到人脸时进行处理
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # 只处理第一个检测到的人脸
                    face = cv2.resize(frame[y:(y + h), x:(x + w)], (42, 42))
                    emotion, probability = self.emo_cls.get_emotion(Image.fromarray(face))

                    # 更新情绪结果标签
                    emotion_text = f'检测到情绪: {emotion.upper()}'
                    self.results_label.config(text=emotion_text)

                    # 更新概率条
                    self._update_probability_bars(probability)

                    # 记录日志
                    self._log_recognition(emotion, probability)

        # 设置下一次更新
        self.window.after(self.delay, self.update)

    def open_image(self):
        """打开并检测图片"""
        # 暂停视频识别
        if self.is_recognizing:
            self.toggle_recognition()

        # 打开文件对话框
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                # 读取图片
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("无法读取图片文件")

                # 显示图片
                self._display_image(image)

                # 检测图片中的人脸和情绪
                self._detect_image(image)

            except Exception as e:
                self.results_label.config(text=f"错误: {str(e)}")
                self.log_text.insert(END, f"错误: {str(e)}\n")

    def _display_image(self, image):
        """在画布上显示图片"""
        # 调整图片大小以适应画布
        h, w = image.shape[:2]
        ratio = min(640 / w, 480 / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h))

        # 转换为PIL格式并显示
        img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.current_image = ImageTk.PhotoImage(img)
        self.canvas.img = self.current_image  # 保持引用防止被垃圾回收
        self.canvas.delete("all")
        self.canvas.create_image(
            (640 - new_w) // 2, (480 - new_h) // 2,
            anchor=NW,
            image=self.current_image
        )

    def _detect_image(self, image):
        """检测图片中的情绪"""
        # 转换为灰度图进行人脸检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.emo_cls.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # 只处理第一个检测到的人脸
            (x, y, w, h) = faces[0]

            # 在原图上绘制人脸矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self._display_image(image)  # 重新显示带框的图片

            # 提取人脸区域并检测情绪
            face = cv2.resize(image[y:(y + h), x:(x + w)], (42, 42))
            emotion, probability = self.emo_cls.get_emotion(Image.fromarray(face))

            # 更新情绪结果标签
            emotion_text = f'检测到情绪: {emotion.upper()}'
            self.results_label.config(text=emotion_text)

            # 更新概率条
            self._update_probability_bars(probability)

            # 记录日志
            self._log_recognition(emotion, probability)
        else:
            self.results_label.config(text="未检测到人脸")
            self.bar_canvas.delete("all")

    def _update_probability_bars(self, probability):
        """更新概率条显示"""
        # 清空画布
        self.bar_canvas.delete('all')

        # 获取画布尺寸
        bar_width = self.bar_canvas.winfo_width()
        bar_height = self.bar_canvas.winfo_height()

        # 定义显示参数
        emotion_labels = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '正常']
        bar_colors = ['#e74c3c', '#8e44ad', '#3948db', '#2ecc71', '#34495e', '#f39c12', '#95a5a6']

        # 计算最大概率值用于归一化
        max_prob = np.max(probability)

        # 计算每个条目的高度
        bar_height_unit = bar_height / len(probability)

        # 绘制每个情绪的概率条
        for i, (prob, label, color) in enumerate(zip(probability, emotion_labels, bar_colors)):
            # 计算条形图尺寸
            bar_length = (prob / max_prob) * (bar_width - 120)  # 留出标签空间
            y0 = i * bar_height_unit + 10
            y1 = (i + 1) * bar_height_unit - 10

            # 绘制条形图
            self.bar_canvas.create_rectangle(
                10, y0, 10 + bar_length, y1,
                fill=color,
                outline='',
                width=0
            )

            # 添加标签和百分比
            label_text = f"{label}: {prob * 100:.1f}%"
            self.bar_canvas.create_text(
                20 + bar_length, (y0 + y1) / 2,
                text=label_text,
                font=self.bar_font,
                anchor=W,
                fill='#333333'
            )

    def _init_log_file(self):
        """初始化日志文件"""
        log_filename = "recognition_log.txt"
        if not os.path.exists(log_filename):
            with open(log_filename, 'w') as f:
                f.write("情绪识别日志文件\n")
        return log_filename

    def _log_recognition(self, emotion, probability):
        """记录识别结果到日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - 检测到情绪: {emotion.upper()}\n"
        self.log_text.insert(END, log_entry)
        self.log_text.see(END)

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def toggle_recognition(self):
        """切换识别状态"""
        self.is_recognizing = not self.is_recognizing
        if self.is_recognizing:
            self.toggle_button.config(text='暂停识别', bg='#e74c3c')
            self.update_flag = True
        else:
            self.toggle_button.config(text='开始识别', bg='#2ecc71')
            self.update_flag = False


if __name__ == '__main__':
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    try:
        # 启动应用程序
        App(video_source=cap)
    finally:
        # 确保释放摄像头资源
        cap.release()