from .train import get_waveform_from_logMel
from .vqvae import VQVAEGAN
from ..waveglow.denoiser import Denoiser
from ..dataset import files_to_list
from ..melgan.mel2wav.modules import Audio2MelTorch
from ..util import save_sample

import os
import torch
from scipy.io.wavfile import write

import argparse
import numpy as np
from pathlib import Path
from librosa.core import load
from librosa.util import normalize
import yaml


def load_wav_to_torch(full_path, sampling_rate=16000):
    """
    Loads wavform into torch array
    """
    data, sampling_rate = load(full_path, sr=sampling_rate)
    data = 0.95 * normalize(data)

    return torch.from_numpy(data).float(), sampling_rate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",
                        default=".",
                        help="Where the generated audio will be stored")
    parser.add_argument("--load_path",
                        default=".",
                        help="Where the model is")
    parser.add_argument("--data_path",
                        default=".",
                        help="Where the test_files.txt resides")
    parser.add_argument("--waveglow_path",
                        default="",
                        help="Path to Waveglow model. If not set, then Griffin Lim is used")
    parser.add_argument("--spk_id", default="all_spk", help="Id of wtimit speaker")
    args = parser.parse_args()
    return args


def main():

    my_args = parse_args()

    save_root = Path(my_args.save_path)
    load_root = Path(my_args.load_path)
    waveglow_path = my_args.waveglow_path
    save_root.mkdir(parents=True, exist_ok=True)

    audio_files = files_to_list(Path(my_args.data_path) / "test_files.txt")
    audio_files = [Path(x) for x in audio_files]

    print(f'Number of test files before filtering {len(audio_files)}')
    if my_args.spk_id is not None and my_args.spk_id != "" and my_args.spk_id != "all_spk_by_spk":
        audio_files = [i for i in audio_files if str(my_args.spk_id) in i.name]
    print(f'Number of test files after filtering {len(audio_files)}')

    fft = Audio2MelTorch(n_fft=1024, hop_length=256, win_length=1024,
                         sampling_rate=16000, n_mel_channels=80).cuda()

    with open(load_root / "args.yml", "r") as f:
        args = yaml.load(f)
    print(args)

    # For regular Encoder + Melgan Decoder
    upsample_ratios = [8, 8, 4, 4]
    # For ConvolutionalEncoder Sonnet + Convolutional Decoder
    if args.use_sonnet_encoder:
        upsample_ratios = [8, 8, 4, 2]

    # For raw audio input
    if args.train_on_raw_audio:
        upsample_ratios = [4, 4, 2, 2]

    if args.use_sonnet_encoder and args.use_melgan_decoder:
        upsample_ratios = [11, 7, 3, 2]

    if waveglow_path is not None and waveglow_path != "":
        MAX_WAV_VALUE = 32768.0
        print(f"Loading WaveGlow model from {waveglow_path}")
        waveglow = torch.load(waveglow_path)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.cuda().eval()

        if args.denoiser_strength > 0:
            denoiser = Denoiser(waveglow).cuda()

    for f in audio_files:

        model = VQVAEGAN(num_hiddens=args.num_hiddens, num_embeddings=args.num_embeddings,
                         embedding_dim=args.embedding_dim, commitment_cost=args.commitment_cost,
                         decay=args.decay, mel_channels=args.n_mel_channels, ngf=args.ngf,
                         n_residual_layers=args.n_residual_layers, ratios=upsample_ratios,
                         use_sonnet_vq=args.use_sonnet_vq, use_sonnet_encoder=args.use_sonnet_encoder,
                         use_kaiming_normal=args.use_kaiming_normal,
                         train_on_raw_audio=args.train_on_raw_audio,
                         use_melgan_decoder=args.use_melgan_decoder,
                         debug=False).cuda()

        if load_root and load_root.exists():
            model.load_state_dict(torch.load(load_root / "vqvaegan.pt"))

        audio, sr = load_wav_to_torch(f, sampling_rate=16000)

        x_t = audio.unsqueeze(0).unsqueeze(0).cuda()

        filename = f.name
        print(f"Handling File: {filename}, xt_from={torch.min(x_t).item()},"
              f" xt_to={torch.max(x_t).item()}, xt_shape={x_t.shape}")

        s_t = fft(x_t).cuda()
        # x_pred_t = model(s_t.cuda())
        _, pred_audio, _ = model(s_t)

        #pred_audio = pred_audio.squeeze().cpu()
        if not args.use_melgan_decoder:
            if waveglow_path is not None and waveglow_path != "":
                with torch.no_grad():
                    print('Waveglow: mel input', pred_audio.shape, torch.min(pred_audio).item(), torch.max(pred_audio).item())
                    wg_audio = waveglow.infer(pred_audio, sigma=1.0)
                    if args.denoiser_strength > 0:
                        wg_audio = denoiser(wg_audio, args.denoiser_strength)
                    print("waveglow before ", wg_audio.shape, torch.min(wg_audio).item(), torch.max(wg_audio).item())
                    wg_audio = wg_audio * MAX_WAV_VALUE
                    wg_audio = wg_audio.squeeze().cpu().numpy().astype('int16')

                print(f"Saving Waveglow synthesis: {filename}, wg_audio_from={np.min(wg_audio)},"
                      f" wg_audio_to={np.max(wg_audio)}, wg_audio_shape={wg_audio.shape}")
                write(os.path.join(str(save_root), str(filename)), 16000, wg_audio)

            else:
                # Here we do Griffin Lim synthesis
                # print(pred_audio.shape)
                pred_audio = get_waveform_from_logMel(pred_audio.squeeze().cuda())

                print(f"Saving Griffin-Lim Synthesis: {filename}, pred_audio_from={torch.min(pred_audio).item()},"
                      f" pred_audio_to={torch.max(pred_audio).item()}, pred_audio_shape={pred_audio.shape}")
                pred_audio = pred_audio.squeeze().detach().cpu()
                save_sample(save_root / str(filename), 16000, pred_audio)
        else:
            pred_audio = pred_audio.squeeze().detach().cpu()
            save_sample(save_root / str(filename), 16000, pred_audio)


if __name__ == '__main__':
    main()