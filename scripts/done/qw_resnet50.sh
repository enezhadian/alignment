#!/usr/bin/env bash

#block(name=weak_delf, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_weak                                                                                       \
        --model-aff trained_models/best_resnet50_smallerlr_strong_30_pascal_affine_resnet50_grid_loss.pth.tar   \
        --model-tps trained_models/best_resnet50_smallerlr_strong_30_pascal_tps_resnet50_grid_loss.pth.tar      \
        --feature-extraction-cnn        resnet50                                                                \
        --feature-extraction-last-layer layer3                                                                  \
        --training-dataset              pf-pascal                                                               \
        --dataset-csv-path              training_data/pf-pascal-flip/                                           \
        --dataset-image-path            datasets/proposal-flow-pascal                                           \
        --num-epochs                    30                                                                      \
        --lr                            5e-8                                                                    \
        --result-model-fn               resnet50_weak
