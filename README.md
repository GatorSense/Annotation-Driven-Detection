# Annotation-Driven-Detection
Beyond Task-Driven Features for Object Detection

    OVERVIEW

This document describes the full experimental pipeline used to evaluate
latent space methods (DTL, DTL-hard, MATL, CTL) for object detection
on the AWIR dataset using Faster R-CNN.

The pipeline consists of four stages:

Train projection models from CLIP embeddings

Generate sliding-window latent feature grids

Train Faster R-CNN with latent grid fusion

Evaluate trained detection models

Each stage depends on outputs from the previous stage.

-----------------------------------------------------------------------

STAGE 1: Train Projection Models from CLIP Embeddings

-----------------------------------------------------------------------

Script:
/blue/azare/zhou.m/awir/obj_det_awir_clip_ctl_emb.py

Purpose:

Train a projection head that maps frozen CLIP ViT-B/32 image embeddings
into a structured latent space defined by the selected method:DTL, DTL_HARD, MATL, CTL

CLIP remains frozen. The projection layer learns annotation-driven
structure using the chosen triplet formulation.

Output:

For each fold, the script saves projection checkpoints:

trained_embedding_models_demo/emb_proj/clip/
    dtl_best_fold{fold}.h5
    dtl_hard_best_fold{fold}.h5
    matl_best_fold{fold}.h5
    ctl_best_fold{fold}.h5
    
    These files represent trained projection heads that transform 512-D
CLIP embeddings into the learned latent space.

Important:

The folds correspond to projection model cross-validation.
The dataset split used later for detection remains fixed.

-----------------------------------------------------------------------

STAGE 2: Generate Sliding-Window Latent Feature Grids

-----------------------------------------------------------------------

Notebook:
observing_embeddings.ipynb

Section:
Saving Embedding Grids

Purpose:

For each projection model fold and latent space method:

Load CLIP ViT-B/32.

Slide a window across each AWIR image.

Extract patch-level CLIP embeddings.

Pass embeddings through the trained projection head.

Reshape embeddings into a spatial grid aligned with the image.

Save one compressed .npz file per image.

Each .npz contains:

clip_grid
dtl_grid
dtl_hard_grid
matl_grid
ctl_grid
coords_grid
metadata (image size, grid size, stride, patch size)

awir_sliding_grids_fold{fold_number}/
    train/
        *.npz
    val/
        *.npz
    index_train.csv
    index_val.csv
    
    
Each folder corresponds to one projection fold.
The train/val split is deterministic and shared across folds.

These saved grids are later loaded by the Faster R-CNN training script.

-----------------------------------------------------------------------

STAGE 3: Train Faster R-CNN with Latent Grid Fusion

-----------------------------------------------------------------------

Script:
train_awir_frcnn_grid_multipleFusion.py

Purpose:

Train Faster R-CNN using precomputed sliding-window latent grids.
Different fusion techniques are evaluated to inject latent structure
into the backbone feature hierarchy.

Fusion methods may include:

- Additive fusion
- Concatenation fusion
- FiLM-based modulation
- Spatial mask fusion

Inputs: 

--data_root      AWIR dataset root
--grid_root      awir_sliding_grids_fold{fold_number}
--variants       latent space methods to use
--fusion type    selected fusion mechanism
--hyperparameters (epochs, batch size, learning rate, etc.)


During training:

The detector loads image data.

The corresponding latent grid is loaded from disk.

Latent features are fused into the backbone.

RPN and ROI heads operate normally.

The detector is trained end-to-end.

Output:

Trained Faster R-CNN models are saved under:

fasterrcnn_runs/{run_tag}/

This directory contains:
- Model checkpoints
- Training logs
- Configuration metadata

Each run_tag encodes:
latent space method
fold number
fusion strategy
batch size and learning rate

-----------------------------------------------------------------------

STAGE 4: Evaluate Detection Models

-----------------------------------------------------------------------

Notebook:
faster_r_cnn.ipynb

Section:
Evaluating from Saved Model

Purpose:

Load a trained Faster R-CNN checkpoint and evaluate detection
performance on the validation set.

Evaluation metrics may include:

- mAP@0.5
- Precision
- Recall
- ROC curves
- Spatial IoU comparisons

Procedure:

Load the saved model checkpoint.

Load validation data.

Perform inference.

Compute evaluation metrics.

Save plots and summary results.

This stage produces the final quantitative comparisons across:

- Latent space methods
- Projection folds
- Fusion mechanisms


-----------------------------------------------------------------------

COMPLETE PIPELINE SUMMARY

-----------------------------------------------------------------------

Step 1:
Train projection heads from CLIP embeddings.

Step 2:
Generate sliding-window latent feature grids per fold.

Step 3:
Train Faster R-CNN using grid fusion.

Step 4:
Evaluate trained detection models.

Data flow:

AWIR Images
    → CLIP Embeddings
    → Projection Head (DTL / MATL / CTL)
    → Sliding Window Grid
    → Faster R-CNN Fusion
    → Detection Metrics
    
    
    NOTES

Projection folds apply to the projection head only.

The detection dataset split remains fixed.

Sliding grids are precomputed to avoid recomputation during detection training.

All latent space comparisons use identical detection hyperparameters
for fair evaluation.