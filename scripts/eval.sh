#!/usr/bin/env bash

#block(name=eval_delf50, threads=2, memory=7500, hours=1000, gpus=1)
#     python eval.py                                                                                                \
#       --feature-extraction-cnn 50_1000_full_20                                                                    \
#       --model                  ../final_evaluations/best_delf_weak_50_1000_full_20_regfact0.2.pth.tar             \
#       --delf-path              ../selected_models/delf_ep10_bz32_lr0.001_im224_taken/20180927113241/best.pth.tar  \
#       --eval-dataset           pf-pascal
#     python eval.py                                                                                                \
#       --feature-extraction-cnn resnet101                                                                          \
#       --model                  ../selected_models/weak_resnet101/resnet101_weak_resnet101_regfact0.2.pth.tar      \
#       --eval-dataset           pf-pascal
    python eval.py                                                                                                \
      --feature-extraction-cnn 101_1000_full_1000                                                                 \
      --model                  ../final_evaluations/best_delf_dist101_weak_101_1000_full_1000_regfact0.2.pth.tar  \
      --delf-path              ../selected_models/delf_dist101_converted.pth.tar                                  \
      --eval-dataset           pf-pascal

