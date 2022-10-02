# WaveGlow: A Flow-based Generative Network for Speech Synthesis

This module contains the code for [WaveGlow](https://arxiv.org/abs/1811.00002). 
The code was obtained from [https://github.com/NVIDIA/waveglow.git](https://github.com/NVIDIA/waveglow.git). 
Pleasure refer to the original repo for more detailed information. 

## Setup
The setup involves two main steps: 

1. Install requirements `pip3 install -r requirements.txt`

2. Install [Apex](https://github.com/nvidia/apex)

## Train Model

2. Make a list of the file names to use as training and test sets:

   ```bash
   ls data/*n.WAV | tail -n+10 > train_files.txt
   ls data/*n.WAV | head -n10 > test_files.txt
   ```

3. Train WaveGlow:
   If you want to train the model on spectrograms that are compatible to 
   the MelGAN system, use:
   
   ```bash
      python3 train_melgan_compat.py -c config_compat.json
   ```

   This is required if you plan on synthesizing speech from spectrograms that come from the SC-VQ-VAE system. 
   Note that we call it "compatible to MelGAN spectrograms", since we use the spectrogram extraction method 
   from the MelGAN system for the VQ-VAEs as well. 

   To train the model in its original configuration, use:
   ```bash
   python3 train.py -c config.json
   ```
   
   For mixed precision training set `"fp16_run": true` on `config.json`.

4. Make test set mel-spectrograms:

   `python mel2samp.py -f test_files.txt -o . -c config.json`

5. Inference:

   ```bash
   ls *.pt > mel_files.txt
   python3 inference.py -f mel_files.txt -w checkpoints/waveglow_10000 -o . --is_fp16 -s 0.6
   ```
