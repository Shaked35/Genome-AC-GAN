#!/bin/bash

python genome_ac_gan_training.py \
  --hapt_genotypes_path resource/10K_SNP_1000G_real.hapt \
  --experiment_name Genomoe-AC-GAN_By_Continental \
  --extra_data_path resource/extra_data_1000G.tsv \
  --results_folder experiment_results \
  --latent_size 600 \
  --alph 0.01 \
  --g_learn 0.0001 \
  --d_learn 0.0008 \
  --epochs 10001 \
  --batch_size 256 \
  --class_loss_weights 1.0 \
  --save_number 50 \
  --minimum_samples 50 \
  --first_epoch 1 \
  --target_column Superpopulation_code \
  --d_activation sigmoid \
  --class_loss_function polyloss_ce \
  --validation_loss_function binary_crossentropy