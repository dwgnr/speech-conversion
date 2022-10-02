from .util import files_to_list

import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random


class AudioDataset(torch.utils.data.Dataset):
    """
    Returns only one type of audio (e.g. normal). 
    Can be used with a 'regular' dataset such as LJSpeech. 
    """

    def __init__(self, training_files, segment_length, sampling_rate, spk_id=None, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(x) for x in self.audio_files]

        if spk_id is not None and spk_id != "":
            self.audio_files = [i for i in self.audio_files if str(spk_id) in str(i.name)[:4]]

        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment
        print(f"[AudioDataset]: Dataset contains {len(self.audio_files)} files")

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take audio segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        return audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


class AudioDatasetWhisper(torch.utils.data.Dataset):
    """
    Returns a whispered/normal audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, spk_id=None, augment=True):
        if "all_spk" in spk_id:
            spk_id = None
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.augment = augment
        # We assume that train files list contains only normal audio files
        # and test files list contains only whispered files
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(x) for x in self.audio_files]

        if spk_id is not None and spk_id != "":
            self.audio_files = [i for i in self.audio_files if str(spk_id) in str(i.name)[:4]]

        random.seed(1234)
        random.shuffle(self.audio_files)
        print(f"[AudioDatasetWhisper]: Dataset contains {len(self.audio_files)} files")

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        whispered_audio, whsp_sampling_rate = self.load_wav_to_torch(str(filename).lower().replace("n.wav", "w.wav"))

        # Take normal segment
        audio_start = 0
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # Take whispered segment
        if whispered_audio.size(0) >= self.segment_length:
            whispered_audio = whispered_audio[audio_start:audio_start+self.segment_length]
        else:
            whispered_audio = F.pad(whispered_audio, (0, self.segment_length - whispered_audio.size(0)), 'constant').data

        return audio.unsqueeze(0), whispered_audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate