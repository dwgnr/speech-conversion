import sys

from .vqvae import VQVAEGAN
from ..waveglow.denoiser import Denoiser
from ..dataset import AudioDatasetWhisper
from ..melgan.mel2wav.modules import Audio2MelTorch
from ..util import save_sample

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
import numpy as np
import time
import argparse
from pathlib import Path


class DBToAmplitude(nn.Module):
    def __call__(self, features):
        return torch.from_numpy(np.power(10.0, features.detach().cpu().numpy())).cuda()


def get_waveform_from_logMel(features, n_fft=1024, hop_length=256, sr=16000):
    """
        Inverse Transform based on: https://jumpml.com/howto-invert-logmel/output/
    """
    n_mels = features.shape[-2]
    inverse_transform = torch.nn.Sequential(
            DBToAmplitude(),
            torchaudio.transforms.InverseMelScale(n_stft=n_fft//2+1, n_mels=n_mels, sample_rate=sr),
            torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length, n_iter=32)
            ).cuda()
    waveform = inverse_transform(torch.squeeze(features))
    return torch.unsqueeze(waveform, 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",
                        help="Path to save the checkpoints at",
                        default=None, required=True)
    parser.add_argument("--load_path",
                        help="Path to an already existing checkpoint to continue training from",
                        default=None)
    parser.add_argument("--data_path",
                        help="Path where the train_files.txt file is located",
                        default=None, required=True)
    parser.add_argument("--waveglow_path", default="", help="Path to pretrained WaveGlow model for synthesis")

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--use_sonnet_vq", action='store_true',
                        help='The Sonnet Quantizer looks a bit more complicated but should yield the '
                             'same results as the other quantizer but without kmeans init. You probably dont need it.')
    parser.add_argument("--use_sonnet_encoder", action='store_true',
                        help='This uses the ConvolutionalEncoder by Sonnet and only works in combination with '
                             'the Sonnet DeconvolutionalDecoder.'
                             'Hence, this flag can only be set if you plan on synthesizing speech with WaveGlow. '
                             'It can NOT be used if you apply --use_melgan_decoder i.e., '
                             'when you use the MelGAN generator for synthesis')
    parser.add_argument("--use_kaiming_normal", action='store_true',
                        help='apply weight normalization and initialize weights with kaiming')
    parser.add_argument("--train_on_raw_audio", action='store_true',
                        help='Train on raw audio waveforms instead of melspec features')
    parser.add_argument("--use_melgan_decoder", action='store_true',
                        help='Apply the MelGan Generator as Decoder instead of ConvolutionalDecoder')
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--n_test_samples", type=int, default=8)

    parser.add_argument("--spk_id", default=None)
    parser.add_argument("--denoiser_strength", type=float, default=0.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.use_sonnet_encoder and args.use_melgan_decoder:
        print("ERROR: Using the Sonnet Encoder in combination with the MelGAN Decoder is not allowed!")
        print("Unset either --use_sonnet_encoder or --use_melgan_decoder")
        sys.exit(1)

    train(args)


def train(args):

    DEBUG = args.debug
    print(args)

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

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

    if DEBUG:
        print("=================================================")
        print(f"Chosen upsample ratios: {upsample_ratios}")
        print("=================================================")

    model = VQVAEGAN(num_hiddens=args.num_hiddens, num_embeddings=args.num_embeddings,
                     embedding_dim=args.embedding_dim, commitment_cost=args.commitment_cost,
                     decay=args.decay, mel_channels=args.n_mel_channels, ngf=args.ngf,
                     n_residual_layers=args.n_residual_layers, ratios=upsample_ratios,
                     use_sonnet_vq=args.use_sonnet_vq, use_sonnet_encoder=args.use_sonnet_encoder,
                     use_kaiming_normal=args.use_kaiming_normal,
                     train_on_raw_audio=args.train_on_raw_audio,
                     use_melgan_decoder=args.use_melgan_decoder,
                     debug=DEBUG).cuda()

    # fft = Audio2Mel(n_mel_channels=args.n_mel_channels, sampling_rate=16000).cuda()
    fft = Audio2MelTorch(n_fft=1024, hop_length=256, win_length=1024,
                         sampling_rate=16000, n_mel_channels=args.n_mel_channels).cuda()
    # fft = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
    #                                              hop_length=256, n_mels=80).cuda()

    if args.waveglow_path != "":
        MAX_WAV_VALUE = 32768.0
        waveglow = torch.load(args.waveglow_path)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.cuda().eval()

        if args.denoiser_strength > 0:
            denoiser = Denoiser(waveglow).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        model.load_state_dict(torch.load(load_root / "vqvaegan.pt"))

    train_set = AudioDatasetWhisper(
        Path(args.data_path) / "train_files.txt",
        segment_length=args.seq_len,
        sampling_rate=16000,
        spk_id=args.spk_id
    )
    test_set = AudioDatasetWhisper(
        Path(args.data_path) / "test_files.txt",
        segment_length=16000 * 4,
        sampling_rate=16000,
        spk_id=args.spk_id
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=1)

    test_voc = []
    test_audio = []
    # For testing we can take the first argument and ingore the second one since both files should be whispered
    # assuming all files in test_files.txt are whispered
    for i, data in enumerate(test_loader):
        x_t = data[0]
        x_t = x_t.cuda()
        # s_t = torch.log10(fft(x_t)).detach()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 16000, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=16000)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, args.epochs + 1):
        # Assuming train_files.txt contains only normal files,
        # the first return value from train_loader is normal and the second one is whispered
        for iterno, data in enumerate(train_loader):
            x_t, x_t_whsp = data
            optimizer.zero_grad()

            x_t = x_t.cuda()
            x_t_whsp = x_t_whsp.cuda()

            s_t = fft(x_t).detach()
            s_t_whsp = fft(x_t_whsp).detach()

            vq_loss, s_t_pred, perplexity = model(s_t_whsp.cuda())

            if DEBUG:
                print("x_t, shape, min, max", x_t.shape, torch.min(x_t).item(), torch.max(x_t).item())
                print("s_t, shape, min, max", s_t.shape, torch.min(s_t).item(), torch.max(s_t).item())
                print("s_t_pred, shape, min, max", s_t_pred.shape, torch.min(s_t_pred).item(), torch.max(s_t_pred).item())

            if args.use_melgan_decoder:
                # The melgan decoder/generator predicts raw audio,
                # so we have to do an fft on the predicted waveform first
                s_t_pred = fft(s_t_pred)
                if DEBUG: print("s_t_pred_after_fft, shape, min, max", s_t_pred.shape, torch.min(s_t_pred).item(), torch.max(s_t_pred).item())

            recon_error = F.mse_loss(s_t_pred, s_t)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            costs.append([recon_error.item(), vq_loss.item(), loss.item(), perplexity.item()])

            writer.add_scalar("vqvaegan/audio_reconstruction_loss", costs[-1][0], steps)
            writer.add_scalar("vqvaegan/vq_loss", costs[-1][1], steps)
            writer.add_scalar("vqvaegan/total_loss", costs[-1][2], steps)
            writer.add_scalar("vqvaegan/perplexity", costs[-1][3], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                    _, prediction, _ = model(voc)

                    if not args.use_melgan_decoder:

                        # Here we do Griffin Lim synthesis
                        pred_audio = get_waveform_from_logMel(prediction.cuda())

                        if args.waveglow_path != "":
                            with torch.no_grad():
                                print("Mel input to waveglow:", voc.shape)
                                wg_audio = waveglow.infer(prediction, sigma=1.0)
                                if args.denoiser_strength > 0:
                                    wg_audio = denoiser(wg_audio, args.denoiser_strength)

                            wg_audio = wg_audio.squeeze().detach().cpu()

                            print("waveglow", wg_audio.shape, torch.min(wg_audio).item(), torch.max(wg_audio).item())

                        print("griffin_lim", pred_audio.shape, torch.min(pred_audio).item(), torch.max(pred_audio).item())

                    pred_audio = pred_audio.squeeze().detach().cpu()
                    save_sample(root / ("generated_%d.wav" % i), 16000, pred_audio)
                    writer.add_audio(
                        "generated/sample_%d.wav" % i,
                        pred_audio,
                        epoch,
                        sample_rate=16000,
                    )
                    if args.waveglow_path != "":
                        save_sample(root / ("generated_waveglow_%d.wav" % i), 16000, wg_audio)
                        writer.add_audio(
                            "generated/sample_waveglow_%d.wav" % i,
                            wg_audio.unsqueeze(1),
                            epoch,
                            sample_rate=16000,
                        )

                torch.save(model.state_dict(), root / "vqvaegan.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(model.state_dict(), root / "best_vqvaegan.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


def train_on_raw_audio(args):

    DEBUG = False

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "args.yml", "w") as f:
        dump_args = args
        dump_args["data_path"] = str(dump_args["data_path"])
        yaml.dump(dump_args, f)
    writer = SummaryWriter(str(root))

    upsample_ratios = [4, 4, 2, 2]

    model = VQVAEGAN(num_hiddens=args.num_hiddens, num_embeddings=args.num_embeddings,
                     embedding_dim=args.embedding_dim, commitment_cost=args.commitment_cost,
                     decay=args.decay, mel_channels=args.n_mel_channels, ngf=args.ngf,
                     n_residual_layers=args.n_residual_layers, ratios=upsample_ratios,
                     use_sonnet_vq=args.use_sonnet_vq, use_sonnet_encoder=True,
                     train_on_raw_audio=True, use_melgan_decoder=True,
                     use_kaiming_normal=args.use_kaiming_normal).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        model.load_state_dict(torch.load(load_root / "vqvaegan.pt"))

    train_set = AudioDatasetWhisper(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=16000
    )
    test_set = AudioDatasetWhisper(
        Path(args.data_path) / "test_files.txt",
        16000 * 4,
        sampling_rate=16000,
        augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=1)

    test_audio = []
    # For testing we can take the first argument and ingore the second one since both files should be whispered
    # assuming all files in test_files.txt are whispered
    for i, data in enumerate(test_loader):
        x_t = data[0]
        x_t = x_t.cuda()

        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 16000, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=16000)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, args.epochs + 1):
        # Assuming train_files.txt contains only normal files,
        # the first return value from train_loader is normal and the second one is whispered
        for iterno, data in enumerate(train_loader):
            x_t, x_t_whsp = data
            optimizer.zero_grad()

            x_t = x_t.cuda()

            vq_loss, x_pred_t_whsp, perplexity = model(x_t.cuda())

            if DEBUG:
                print("x_pred_t_whsp, shape, min, max", x_pred_t_whsp.shape, torch.min(x_pred_t_whsp).item(), torch.max(x_pred_t_whsp).item())
                print("x_t, shape, min, max", x_t.shape, torch.min(x_t).item(), torch.max(x_t).item())

            recon_error = F.l1_loss(x_pred_t_whsp, x_t)

            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            costs.append([recon_error.item(), vq_loss.item(), loss.item(), perplexity.item()])

            writer.add_scalar("vqvaegan/audio_reconstruction_loss", costs[-1][0], steps)
            writer.add_scalar("vqvaegan/vq_loss", costs[-1][1], steps)
            writer.add_scalar("vqvaegan/total_loss", costs[-1][2], steps)
            writer.add_scalar("vqvaegan/perplexity", costs[-1][3], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    for i, voc in enumerate(test_audio):
                        _, pred_audio, _ = model(voc)

                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 16000, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=16000,
                        )

                torch.save(model.state_dict(), root / "vqvaegan.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(model.state_dict(), root / "best_vqvaegan.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


if __name__ == "__main__":
    main()
