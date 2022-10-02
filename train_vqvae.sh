#!/bin/bash


if [ $# != 3 ]; then
  echo "Train a VQ-VAE system for whisper to normal speech conversion."
  echo "Usage: $0 <cuda_id> <spk_id> <model_name>"
  echo "e.g.:"
  echo " $0 1 014 my_vqvae_model"
  echo "<cuda_id> is the CUDA device you want to use."
  echo "<spk_id> is the speaker id from the wTIMIT corpus. Set it to 'all_spk' for speaker-indepedent training. "
  echo "<model_name> is the name of a directory you want to save the model files at"
  exit 1
fi

cuda_id=$1
spk_id=$2
model_name=$3

wg_script_path=$(pwd)/speech-conversion/waveglow
echo "Path to WaveGlow Scripts: " $wg_script_path

# Save model checkpoints
SAVE_PATH=./data/checkpoint/${spk_id}/${model_name}
# Path to audio data
DATA_PATH=./data
# Load model from existing checkpoint
LOAD_PATH=./data/checkpoint/${spk_id}/${model_name}
# Path to WaveGlow model; Only necessary when WaveGlow is used for speech synthesis
WG_PATH=./data/checkpoint/wg_test/waveglow_400000

export PYTHONPATH=$wg_script_path:$PWD:$PYTHONPATH && \
export CUDA_VISIBLE_DEVICES=$cuda_id && \
python3 -m speech-conversion.vqvae.train \
	--save_path $SAVE_PATH \
	--data_path $DATA_PATH \
	--num_embeddings 256 --embedding_dim 128 --commitment_cost 0.50 --use_kaiming_normal \
	--batch_size 48 --log_interval 1000 --save_interval 10000 \
	--use_sonnet_encoder --epochs 10000 --spk_id $spk_id \
	--waveglow_path $WG_PATH || exit 1