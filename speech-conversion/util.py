import os
import random
import argparse
import json
import torch
import torch.utils.data
from scipy.io.wavfile import read, write
from pathlib import Path

from .waveglow.tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, dataset):
        self.audio_files = files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        print(f"[Mel2Samp]: Dataset contains {len(self.audio_files)} files")


    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE
        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)


class Mel2SampWhisper(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    NOTE: This assumes the filelist contains only normal files!
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, dataset):

        self.audio_files = files_to_list(training_files)
        print(f"[Mel2SampWhisper]: Dataset contains {len(self.audio_files)} files")

        # Only use normal audio files
        # Note the whispered audio files must be in the same directory
        # self.audio_files = [i for i in self.audio_files if i.endswith("n.wav")]
        # print(f"[Mel2SampWhisper]: {len(self.audio_files)} after filtering")

        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        whispered_audio, whsp_sampling_rate = load_wav_to_torch(filename.replace("n.wav", "w.wav"))

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take normal segment
        audio_start = 0
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        # Take whispered segment
        if whispered_audio.size(0) >= self.segment_length:
            whispered_audio = whispered_audio[audio_start:audio_start+self.segment_length]
        else:
            whispered_audio = torch.nn.functional.pad(whispered_audio,
                                                      (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(whispered_audio)
        audio = audio / MAX_WAV_VALUE

        # Now we return a whispered mel and a normal audio
        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)



def create_waveglow_mel_feats(input_dir, output_dir, config_path="config.json"):
    data_config = None
    with open(config_path, 'r') as j:
        data_config = json.loads(j.read())["data_config"]

    mel2samp = Mel2Samp(**data_config)

    filepaths = Path(input_dir).rglob("*.wav")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filepath in filepaths:
        if "irm" in filepath.name:
            try:
                audio, sr = load_wav_to_torch(filepath)
                melspectrogram = mel2samp.get_mel(audio)
                filename = filepath.name
                new_filepath = output_dir + '/' + str(filename) + '.pt'
                print(new_filepath)
                torch.save(melspectrogram, new_filepath)
            except PermissionError:
                print("ERROR: Permission error occurred for file " + str(filepath.name))

# This main takes a directory of clean audio and makes directory of spectrograms
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]

    if data_config["dataset"] == "wtimit":
        mel2samp = Mel2SampWhisper(**data_config)
    else:
        mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save samples

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    write(file_path, sampling_rate, audio)
