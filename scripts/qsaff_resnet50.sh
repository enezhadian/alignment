#!/usr/bin/env bash

#block(name=strong_affine, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_strong                                                           \
        --feature-extraction-cnn        resnet50                                      \
        --feature-extraction-last-layer layer3                                        \
        --training-dataset              pascal                                        \
        --dataset-csv-path              training_data/pascal-random/                  \
        --num-epochs                    30                                            \
        --batch-size                    32                                            \
        --random-sample                 1                                             \
        --result-model-fn               resnet50

    python3 -m train_strong                                                                        \
        --model    trained_models/best_resnet50_strong_30_pascal_affine_resnet50_grid_loss.pth.tar \
        --feature-extraction-cnn        resnet50                                                   \
        --feature-extraction-last-layer layer3                                                     \
        --training-dataset              pascal                                                     \
        --dataset-csv-path              training_data/pascal-random/                               \
        --num-epochs                    30                                                         \
        --batch-size                    32                                                         \
        --random-sample                 1                                                          \
        --lr                            0.0002                                                     \
        --result-model-fn               resnet50_smallerlr
