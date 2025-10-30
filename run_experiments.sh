#!/bin/bash


bkafi_criterion=("feature_importance" "std")
dataset_sizes=("small" "large")
normalizations=("True" "False")
sdr_factors=("False" "true")
eval_mode="blocking"

for criterion in "${bkafi_criterion[@]}"; do
  for size in "${dataset_sizes[@]}"; do
    for normalization in "${normalizations[@]}"; do
      for sdr in "${sdr_factors[@]}"; do
        echo "=================================================================================================="
        # if eval_mode == blocking print the following
        if [[ $eval_mode == "blocking" ]]; then
          echo "Running blocking with: bkafi_criterion=$criterion, size=$size, vector_normalization=$normalization, SDR=$sdr"
        else
          echo "Running matching with: bkafi_criterion=$criterion, size=$size, vector_normalization=$normalization, SDR=$sdr"
        fi

        python main.py \
          --bkafi_criterion $criterion \
          --evaluation_mode $eval_mode \
          --dataset_size_version $size \
          --vector_normalization $normalization \
          --sdr_factor $sdr
      done
    done
  done
done
