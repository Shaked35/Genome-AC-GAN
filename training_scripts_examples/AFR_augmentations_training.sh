#!/bin/bash

python genome_ac_gan_training.py \
    --hapt_genotypes_path resource/train_AFR_pop.csv \
    --experiment_name sub_set_AFR_aug \
    --extra_data_path resource/extra_data_1000G.tsv \
    --latent_size 800 \
    --alph 0.01 \
    --g_learn 0.00008 \
    --d_learn 0.0004 \
    --epochs 20001 \
    --batch_size 256 \
    --class_loss_weights 0.8 \
    --save_number 200 \
    --minimum_samples 50 \
    --target_column Population_code \
    --d_activation sigmoid \
    --class_loss_function polyloss_ce \
    --validation_loss_function binary_crossentropy \
    --required_populations ACB GWD ESN MSL YRI LWK ASW


