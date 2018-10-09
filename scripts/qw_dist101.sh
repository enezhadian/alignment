#!/usr/bin/env bash

#block(name=weak_delf, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_weak                                                                                                   \
        --model-aff trained_models/best_delf_dist101_smallerlr_strong_30_pascal_affine_101_1000_full_1000_grid_loss.pth.tar \
        --model-tps trained_models/best_delf_dist101_smallerlr_strong_30_pascal_tps_101_1000_full_1000_grid_loss.pth.tar    \
        --feature-extraction-cnn        101_1000_full_1000                                                                  \
        --feature-extraction-last-layer layer3                                                                              \
        --delf-path                     ../selected_models/delf_dist101_converted.pth.tar                                   \
        --training-dataset              pf-pascal                                                                           \
        --dataset-csv-path              training_data/pf-pascal-flip/                                                       \
        --dataset-image-path            datasets/proposal-flow-pascal                                                       \
        --num-epochs                    30                                                                                  \
        --lr                            5e-8                                                                                \
	--result-model-fn               delf_dist101_weak
