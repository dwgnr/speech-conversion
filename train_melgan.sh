#!/bin/bash

# set -euxo

if [ $# != 3 ]; then
  echo "Train a MelGAN system for whisper to normal speech conversion."
  echo "Usage: $0 <cuda_id> <spk_id> <model_name>"
  echo "e.g.:"
  echo " $0 0 014 my_melgan_model"
  echo "<cuda_id> is the CUDA device you want to use."
  echo "<spk_id> is the speaker id from the wTIMIT corpus. Set it to 'all_spk' for speaker-indepedent training. "
  echo "<model_name> is the name of a directory you want to save the model files at"
  exit 1
fi

cuda_id=$1
spk_id=$2
model_name=$3

# Save model checkpoints
SAVE_PATH=./data/${model_name}
# Path to audio data
DATA_PATH=./data
# Load model from existing checkpoint
LOAD_PATH="--load_path ./data/checkpoint/${model_name}"

export PYTHONPATH=$PWD:$PYTHONPATH && export CUDA_VISIBLE_DEVICES=$cuda_id && python3 -m speech-conversion.melgan.train \
        --save_path "${SAVE_PATH}" \
        --data_path ${DATA_PATH} \
        --epochs 1000 --batch_size 40 --log_interval 10 --save_interval 500 --spk_id "${spk_id}" \
        # $LOAD_PATH
