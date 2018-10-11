#!/usr/bin/env bash

#block(name=eval_delf50, threads=2, memory=7500, hours=1000, gpus=1)
    python eval.py                                                                                                      \
      --feature-extraction-cnn 50_1000_full_20                                                                          \
      --model                  "../final_evaluations/best_delf_dist101_gdrive_weak_101_1000_full_20_regfact0.2.pth.tar" \
      --delf-path              "../attention/models/delf_ep10_bz32_lr0.001_im224_taken/20181010215747/best.pth.tar"     \
      --eval-dataset           pf-pascal
