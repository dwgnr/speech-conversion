import argparse
import json
import os
import sys
import torch

from .distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
from .glow import WaveGlow, WaveGlowLoss
from ..dataset import AudioDataset
from ..melgan.mel2wav.modules import Audio2MelTorch
from ..util import save_sample


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    global logger
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config).cuda()

    fft = Audio2MelTorch(n_fft=1024, hop_length=256, win_length=1024,
                         sampling_rate=16000, n_mel_channels=80).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    if data_config["dataset"] == "melgan_compatible":
        print("Attempting to train WaveGlow in MelGAN compatibility mode")

        trainset = AudioDataset(
            data_config["training_files"],
            data_config["segment_length"],
            sampling_rate=data_config["sampling_rate"],
            spk_id=data_config["spk_id"]
        )
    else:
        print(f'ERROR: Invalid dataset {data_config["dataset"]}')
        sys.exit(1)

    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory))

        testset = AudioDataset(
            data_config["training_files"],
            segment_length=16000 * 4,
            sampling_rate=16000,
            spk_id=None
        )

        test_loader = DataLoader(testset, batch_size=1)

        # Dumping original audio 
        test_voc = []
        # For testing we can take the first argument and ingore the second one since both files should be whispered
        # assuming all files in test_files.txt are whispered
        for i, audio in enumerate(test_loader):
            # mel, audio = batch
            mel = fft(audio.cuda()).detach()
            test_voc.append(mel.cuda())
            # test_audio.append(audio)

            audio = audio.squeeze().cpu()
            save_sample(os.path.join(output_directory, "original_%d.wav" % i), 16000, audio)
            logger.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=16000)

            # Hardcoded 8 test samples
            if i == 7:
                break

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, audio in enumerate(train_loader):
            model.zero_grad()

            # mel, audio = batch
            mel = fft(audio.cuda()).detach()

            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.squeeze(1).cuda())
            # print("mel", mel.shape)
            # print("audio", audio.shape)
            outputs = model((mel.contiguous(), audio.contiguous()))
            # outputs = (outputs[0].contiguous(), outputs[1].contiguous())
            loss = criterion(outputs)

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if iteration % iters_per_checkpoint == 0:
                print("Epoch {}, Iter {}:\t{:.9f}".format(epoch, iteration, reduced_loss))

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/waveglow_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

                    if with_tensorboard:
                        logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)
                        MAX_WAV_VALUE = 32768.0

                        if iteration % iters_per_checkpoint == 0:

                            waveglow = torch.load(checkpoint_path)['model']
                            waveglow = waveglow.remove_weightnorm(waveglow)
                            waveglow.cuda().eval()

                            with torch.no_grad():
                                for i, mel in enumerate(test_voc):
                                    print("Input mel feats", mel.shape)
                                    pred_audio = waveglow.infer(mel, sigma=sigma)
                                    # pred_audio = pred_audio_norm * MAX_WAV_VALUE
                                    # pred_audio = pred_audio.squeeze().cpu().numpy().astype('int16')
                                    print("Predicted audio", pred_audio.shape,
                                          torch.min(pred_audio).item(), torch.max(pred_audio).item())
                                    # print("Predicted audio", pred_audio.shape, min(pred_audio), max(pred_audio))
                                    pred_audio = pred_audio.squeeze().detach().cpu()
                                    save_sample(os.path.join(output_directory, "generated_%d.wav" % i), 16000, pred_audio)
                                    print("Predicted audio before permute", pred_audio.shape)
                                    pred_audio = pred_audio.unsqueeze(1)
                                    print("Predicted audio after permute", pred_audio.shape)
                                    # save_sample(os.path.join(output_directory, "generated_%d.wav" % i), 16000, pred_audio)
                                    logger.add_audio(
                                        "generated/sample_%d.wav" % i,
                                        pred_audio,
                                        epoch,
                                        sample_rate=16000,
                                    )
            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a WaveGlow model that is compatible with our SC-VQ-VAE.")
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration',
                        default='')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, **train_config)
