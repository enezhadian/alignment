#!/usr/bin/env bash

#block(name=weak_delf, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_weak                                    \
        --model-aff models/best_delf_smallerlr_strong_50_pascal_affine_50_1000_full_20_grid_loss.pth.tar \
        --model-tps models/best_delf_smallerlr_strong_50_pascal_tps_50_1000_full_20_grid_loss.pth.tar \
        --feature-extraction-cnn        50_1000_full_20      \
        --feature-extraction-last-layer layer3               \
        --delf-path ../delf_ep10_bz32_lr0.001_im224_taken/20180927113241/best.pth.tar \
        --training-dataset              pf-pascal            \
        --dataset-csv-path              data/pf-pascal-flip/ \
        --dataset-image-path            data/pfpascal        \
        --num-epochs                    15                   \
        --lr                            5e-8                 \
        --result-model-fn               delf_weak
