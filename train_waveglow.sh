#!/bin/bash


if [ $# != 1 ]; then
  echo "Train a WaveGlow system that is compatible with our VQ-VAE system."
  echo "Usage: $0 <cuda_id>"
  echo "e.g.:"
  echo " $0 1"
  echo "<cuda_id> is the CUDA device you want to use."
  exit 1
fi

cuda_id=$1

export PYTHONPATH=$PWD:$PYTHONPATH && \
export CUDA_VISIBLE_DEVICES=$cuda_id && \
python3 -m speech-conversion.waveglow.train_melgan_compat \
  --config ./speech-conversion/waveglow/config_compat.json  || exit 1
