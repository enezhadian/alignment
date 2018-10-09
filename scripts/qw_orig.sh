#!/usr/bin/env bash

#block(name=weak_delf, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_weak                                                                                       \
        --model-aff trained_models/best_resnet101_smallerlr_strong_50_pascal_affine_resnet101_grid_loss.pth.tar \
        --model-tps trained_models/best_resnet101_smallerlr_strong_50_pascal_tps_resnet101_grid_loss.pth.tar    \
        --feature-extraction-cnn        resnet101                                                               \
        --feature-extraction-last-layer layer3                                                                  \
        --training-dataset              pf-pascal                                                               \
        --dataset-csv-path              training_data/pf-pascal-flip/                                           \
        --dataset-image-path            datasets/proposal-flow-pascal                                           \
        --num-epochs                    30                                                                      \
        --lr                            5e-8                                                                    \
        --result-model-fn               resnet101_weak
