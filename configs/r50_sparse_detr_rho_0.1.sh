#!/usr/bin/env bash

set -x



python -u main.py \
    --output_dir ./outputs \
    --with_box_refine \
    --two_stage \
    --rho 0.5 \
    --use_enc_aux_loss \
