import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import librosa
import numpy as np


class MultimodalDataset(Dataset):
    def __init__(self, audio_dir, image_dir, label_file, transform=None, audio_length=5, sr=22050):
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_file)
        self.transform = transform
        self.audio_length = audio_length
        self.sr = sr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, row['image_file'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load audio
        audio_path = os.path.join(self.audio_dir, row['audio_file'])
        audio, _ = librosa.load(audio_path, sr=self.sr)

        # Pad or trim audio
        target_length = self.sr * self.audio_length
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, max(0, target_length - len(audio))), 'constant')

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        audio_features = np.concatenate([mfcc, delta, delta2], axis=0)

        # Convert to tensor
        audio_features = torch.FloatTensor(audio_features)

        # Get label
        label = row['emotion']

        return {
            'image': image,
            'audio': audio_features,
            'label': label
        }