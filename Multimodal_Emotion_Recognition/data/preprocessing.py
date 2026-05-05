import cv2
import librosa
import numpy as np
import torch


class ImageProcessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = (img / 255.0 - self.mean) / self.std
        return torch.FloatTensor(img.transpose(2, 0, 1))


class AudioProcessor:
    def __init__(self, sr=16000, n_mels=64, max_len=300):
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len

    def __call__(self, audio_path):
        y, _ = librosa.load(audio_path, sr=self.sr)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel)

        # Padding/Cutting
        if log_mel.shape[1] < self.max_len:
            pad_width = self.max_len - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :self.max_len]

        return torch.FloatTensor(log_mel)