from ..dataset import AudioDataset, AudioDatasetWhisper
from ..util import save_sample
from .mel2wav.modules import Generator, Discriminator, Audio2Mel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
import numpy as np
import argparse
from pathlib import Path
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="Path to save the checkpoints at", required=True)
    parser.add_argument("--load_path",
                        help="Path to an already existing checkpoint to continue training from",
                        default=None)
    parser.add_argument("--data_path",
                        help="Path where the train_files.txt file is located",
                        default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)

    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--n_test_samples", type=int, default=8)

    parser.add_argument("--spk_id", default=None)

    parser.add_argument("--train_normal_only", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.train_normal_only:
        train_on_normal_data_only(args)
    else:
        train_on_whispered_data(args)


def train_on_normal_data_only(args):
    """
    Train the model on normal data only i.e. only normal sounding utterances from the wtimit dataset are used
    :param args: Command line args
    """
    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).cuda()
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels, sampling_rate=16000).cuda()

    print(netG)
    print(netD)

    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))

    train_set = AudioDataset(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=16000
    )
    test_set = AudioDataset(
        Path(args.data_path) / "test_files.txt",
        16000 * 4,
        sampling_rate=16000,
        augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1)

    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t.cuda()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 16000, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=16000)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t.cuda()
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.cuda())

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            # Train Discriminator
            D_fake_det = netD(x_pred_t.cuda().detach())
            D_real = netD(x_t.cuda())

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            # Train Generator
            D_fake = netD(x_pred_t.cuda())

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()
            optG.step()

            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 16000, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=16000,
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time()


def train_on_whispered_data(args):
    """
    Train the model on whispered and normal data. 
    :param args: Command line args
    """

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).cuda()
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels, sampling_rate=16000).cuda()

    print(netG)
    print(netD)

    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))

    train_set = AudioDatasetWhisper(
        Path(args.data_path) / "train_files.txt",
        args.seq_len,
        sampling_rate=16000,
        spk_id=args.spk_id
    )
    test_set = AudioDatasetWhisper(
        Path(args.data_path) / "test_files.txt",
        16000 * 4,
        sampling_rate=16000,
        augment=False,
        spk_id=args.spk_id
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1)

    test_voc = []
    test_audio = []
    # For testing we can take the first argument and ignore the second one since both files should be whispered
    # assuming all files in test_files.txt are whispered
    for i, data in enumerate(test_loader):
        x_t = data[0]
        x_t = x_t.cuda()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 16000, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=16000)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, args.epochs + 1):
        # Assuming train_files.txt contains only normal files,
        # the first return value from train_loader is normal and the second one is whispered
        for iterno, data in enumerate(train_loader):
            x_t, x_t_whsp = data
            x_t = x_t.cuda()
            x_t_whsp = x_t_whsp.cuda()

            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.cuda())

            s_t_whsp = fft(x_t_whsp).detach()
            x_pred_t_whsp = netG(s_t_whsp.cuda())

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

                s_pred_t_whsp = fft(x_pred_t_whsp.detach())
                s_error_whsp = F.l1_loss(s_t_whsp, s_pred_t_whsp).item()

            D_fake_det_whsp = netD(x_pred_t_whsp.cuda().detach())
            D_real = netD(x_t.cuda())

            loss_D = 0
            for scale in D_fake_det_whsp:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            D_fake = netD(x_pred_t.cuda())
            D_fake_whsp = netD(x_pred_t_whsp.cuda())

            loss_G = 0

            for scale in D_fake_whsp:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1) # 4 / (4 + 1)
            D_weights = 1.0 / args.num_D # 1 / 3
            wt = D_weights * feat_weights # 1/3 * 4/5
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake_whsp[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()
            optG.step()

            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error, s_error_whsp])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction_normal", costs[-1][3], steps)
            writer.add_scalar("loss/mel_reconstruction_whisper", costs[-1][4], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 16000, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=16000,
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time()


if __name__ == "__main__":
    main()
