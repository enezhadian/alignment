#!/usr/bin/env bash

#block(name=strong_affine, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_strong                                 \
        --feature-extraction-cnn        50_1000_full_20     \
        --feature-extraction-last-layer layer3              \
        --delf-path ../delf_ep10_bz32_lr0.001_im224_taken/20180927113241/best.pth.tar \
        --training-dataset              pascal              \
        --dataset-csv-path              data/pascal-random/ \
        --num-epochs                    30                  \
        --batch-size                    32                  \
        --random-sample                 1                   \
        --result-model-fn               delf

    python3 -m train_strong                                 \
        --model trained_models/best_delf_strong_50_pascal_affine_50_1000_full_20_grid_loss.pth.tar \
        --feature-extraction-cnn        50_1000_full_20     \
        --feature-extraction-last-layer layer3              \
        --delf-path ../delf_ep10_bz32_lr0.001_im224_taken/20180927113241/best.pth.tar \
        --training-dataset              pascal              \
        --dataset-csv-path              data/pascal-random/ \
        --num-epochs                    30                  \
        --batch-size                    32                  \
        --random-sample                 1                   \
        --lr                            0.0002              \
        --result-model-fn               delf_smallerlr