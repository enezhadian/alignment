#!/usr/bin/env bash

#block(name=weak_mask_classify50, threads=2, memory=7500, hours=1000, gpus=1)
    python3 -m train_weak                                                                                                   \
        --model-aff trained_models/best_mask_classify50_smallerlr_strong_30_pascal_affine_mask_classify50_grid_loss.pth.tar \
        --model-tps trained_models/best_mask_classify50_smallerlr_strong_30_pascal_tps_mask_classify50_grid_loss.pth.tar    \
        --feature-extraction-cnn        mask_classify50                                                                     \
        --feature-extraction-last-layer layer3                                                                              \
        --delf-path                     ../selected_models/mask_classify/segment50layer3.pth.tar                            \
        --training-dataset              pf-pascal                                                                           \
        --dataset-csv-path              training_data/pf-pascal-flip/                                                       \
        --dataset-image-path            datasets/proposal-flow-pascal                                                       \
        --num-epochs                    30                                                                                  \
        --lr                            5e-8                                                                                \
        --result-model-fn               mask_classify50_weak
