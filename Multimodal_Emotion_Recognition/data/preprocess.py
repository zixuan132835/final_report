# data/preprocess.py
import cv2
import librosa
import numpy as np
import torch
from torchvision import transforms


class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_mels=64, max_len=300, noise_level=0.05):
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.noise_level = noise_level

    def add_noise(self, waveform):
        noise = np.random.normal(0, self.noise_level * np.max(waveform), len(waveform))
        return waveform + noise

    def extract(self, audio_path):
        # 加载并增强音频
        y, _ = librosa.load(audio_path, sr=self.sr)
        y = self.add_noise(y)  # 添加高斯噪声

        # 提取Log-Mel特征
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel)

        # 标准化长度
        if log_mel.shape[1] < self.max_len:
            pad_width = self.max_len - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :self.max_len]

        return torch.FloatTensor(log_mel)


class ImageFeatureExtractor:
    def __init__(self, img_size=224, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if augment else lambda x: x,
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if augment else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.transform(img)