from .mel2wav.modules import Generator, Audio2Mel
from ..util import files_to_list, save_sample

import sys
import torch
import argparse
from pathlib import Path
from librosa.core import load
from librosa.util import normalize


def load_wav_to_torch(full_path, sampling_rate=16000):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = load(full_path, sr=sampling_rate)
    data = 0.95 * normalize(data)

    return torch.from_numpy(data).float(), sampling_rate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",
                        default="./audio/generated",
                        help="Where the generated audio will be stored")
    parser.add_argument("--load_path",
                        default="./model/melgan",
                        help="Where the model weights are located")
    parser.add_argument("--data_path",
                        default="./audio/original",
                        help="Where the test_files.txt resides")
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)
    parser.add_argument("--spk_id", help="Id of a speaker in the wTIMIT dataset", default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    save_root = Path(args.save_path)
    load_root = Path(args.load_path)
    save_root.mkdir(parents=True, exist_ok=True)

    audio_files = files_to_list(Path(args.data_path) / "test_files.txt")
    audio_files = [Path(x) for x in audio_files]

    print(f"Found {len(audio_files)} audio files.")
    if args.spk_id is not None and args.spk_id != "all_spk_by_spk":
        audio_files = [i for i in audio_files if str(args.spk_id) in str(i.name)[:4]]

        print(f"Filtered for speaker {args.spk_id}. Using {len(audio_files)} for inference")
        if len(audio_files) < 1:
            print("ERROR: No files selected for inference!")
            sys.exit(1)

    fft = Audio2Mel(n_mel_channels=args.n_mel_channels, sampling_rate=16000).cuda()

    for f in audio_files:

        netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()

        if load_root and load_root.exists():
            netG.load_state_dict(torch.load(load_root / "netG.pt"))

        audio, sr = load_wav_to_torch(f, sampling_rate=16000)

        with torch.no_grad():
            x_t = audio.unsqueeze(0).unsqueeze(0).cuda()

            filename = f.name
            print(f"File: {filename}, xt_from={torch.min(x_t).item()},"
                  f" xt_to={torch.max(x_t).item()}, xt_shape={x_t.shape}")

            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.cuda())
            pred_audio = x_pred_t.squeeze().cpu()

            print(f"Saving predicted audio at: {save_root}/{filename}, pred_audio_from={torch.min(pred_audio).item()},"
                  f" pred_audio_to={torch.max(pred_audio).item()}, pred_audio_shape={pred_audio.shape}")

            save_sample(save_root / str(filename), 16000, pred_audio.detach())


if __name__ == '__main__':
    main()
