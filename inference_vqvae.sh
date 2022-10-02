#!/bin/bash

#set -euxo pipefail

if [ $# != 3 ]; then
  echo "Run inference on a trained VQ-VAE system. "
  echo "Usage: $0 <cuda_id> <spk_id> <model_name>"
  echo "e.g.:"
  echo " $0 0 014 my_vqvae_model"
  echo "<cuda_id> is the CUDA device you want to use."
  echo "<spk_id> is the speaker id from the wTIMIT corpus. Set it to 'all_spk' for speaker-indepedent training. "
  echo "<model_name> is the name of a directory where the trained model weights are located"
  exit 1
fi

CUDA_ID=$1
SPK_ID=$2
MODEL_NAME=$3

# We need to set the path to the WaveGlow scripts, otherwise the model can not be loaded
wg_script_path=$(pwd)/speech-conversion/waveglow
echo "Path to WaveGlow Scripts: " $wg_script_path

# Load model from checkpoint
LOAD_DIR=./data/checkpoint/${SPK_ID}/${MODEL_NAME}
# Save results
SAVE_DIR=./data/checkpoint/${MODEL_NAME}
# Path to audio input data
DATA_DIR=./data
# Path to WaveGlow model for speech synthesis
# Note that you don't need this if you plan on using the VQ-VAE system in an end-to-end fashion
WG_PATH=./data/checkpoint/wg_test/waveglow_400000

echo "Run VQ-VAE inference"
export PYTHONPATH=$wg_script_path:$PYTHONPATH && \
export CUDA_VISIBLE_DEVICES=$CUDA_ID && \
python3 -m speech-conversion.vqvae.inference \
  --save_path $SAVE_DIR \
  --load_path $LOAD_DIR \
  --data_path $DATA_DIR \
  --spk_id $SPK_ID \
  --waveglow_path $WG_PATH

exit 0
