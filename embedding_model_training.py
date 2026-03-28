
# Standard libraries
import os
import sys
import time
import random
import argparse
import numpy as np

# Data science and ML
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Project-specific imports
from TripletNetwork_Online import ctl_projection_headv2, multitask_head_v2
from awir_utilities import (
    global_max_normalize, assign_tile_labels, generate_triplet_box_label_normalized
)
from awir_custom_losses import keras_batch_all_triplet_loss

# Silence TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# ===================== Main Training Script =====================

def main():
    """Main function to train and evaluate embedding models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default='rgb')
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--cls_weight", type=float, default=0.5)
    parser.add_argument("--num_comparison", type=int, default=32)
    args = parser.parse_args()

    parameters = {
        "epochs": 3000,
        "iterations": 1,
        "runs": 1,
        "num_comparisons": args.num_comparison,
        "embedding_dimension": args.embedding_dimension,
        "margin": args.margin,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "modality": args.modality,
        "cls_weight": args.cls_weight
    }

    # Load the merged data
    f = np.load("data.npz")
    modality_chosen = args.modality
    print(f'Modality: {modality_chosen}', flush=True)

    # Extract arrays
    rgb = f['rgb']
    thermal = f['thermal']
    y_cls = f['class_label']
    y_box = np.clip(f['box_label'], None, 300)

    y_mask = np.zeros((240, 300, 300, 1))
    for i, (xmin, xmax, ymin, ymax) in enumerate(y_box):
        y_mask[i, ymin:ymax, xmin:xmax, 0] = 1

    # Convert y_box to center, width, height
    image_width, image_height = 300, 300
    num_samples = y_box.shape[0]
    y_box_cenwh = np.zeros((num_samples, 4))
    for i in range(num_samples):
        xmin, xmax, ymin, ymax = y_box[i]
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        y_box_cenwh[i, 0] = xcenter / image_width
        y_box_cenwh[i, 1] = ycenter / image_height
        y_box_cenwh[i, 2] = width / image_width
        y_box_cenwh[i, 3] = height / image_height

    # One-hot encode the class labels
    encoder = LabelEncoder()
    one_hot_encoded_classes = encoder.fit_transform(y_cls)

    y_triplet_box_label, features = generate_triplet_box_label_normalized(y_box, y_cls, n_clusters=3)

    # Composite label for stratification
    if len(one_hot_encoded_classes.shape) > 1:
        one_hot_encoded_classes = np.argmax(one_hot_encoded_classes, axis=1)
    composite_label = [f"{cls}_{triplet}" for cls, triplet in zip(one_hot_encoded_classes, y_triplet_box_label)]
    composite_label_encoded = LabelEncoder().fit_transform(composite_label)

    rgb_normalized = global_max_normalize(rgb)
    thermal_normalized = global_max_normalize(thermal)

    print('rgb_norm min max:', rgb_normalized.min(), rgb_normalized.max())
    print('thermal_norm min max:', thermal_normalized.min(), thermal_normalized.max())
    print('testing size:', args.test_size)

    tile_labels = assign_tile_labels(y_box_cenwh)

    # Load external embeddings
    awir_dinov2_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_dinov2_emb.npy")
    awir_clip_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_clip_emb.npy")
    awir_mae_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_mae_emb.npy")

    # Train/test split
    (
        X_rgb_trainval, X_rgb_test,
        X_thermal_trainval, X_thermal_test,
        y_cls_trainval, y_cls_test,
        y_box_trainval, y_box_test,
        y_mask_trainval, y_mask_test,
        y_triplet_box_trainval, y_triplet_box_test,
        box_feat_trainval, box_feat_test,
        awir_clip_emb_trainval, awir_clip_emb_test,
        awir_dinov2_emb_trainval, awir_dinov2_emb_test,
        awir_mae_emb_trainval, awir_mae_emb_test,
        composite_label_encoded_trainval, composite_label_encoded_test
    ) = train_test_split(
        rgb_normalized, thermal_normalized, y_cls, y_box_cenwh, y_mask, y_triplet_box_label, features,
        awir_clip_emb_loaded, awir_dinov2_emb_loaded, awir_mae_emb_loaded, composite_label_encoded,
        test_size=0.2, stratify=composite_label_encoded, random_state=42
    )

    val_frac = 0.1
    train_frac = 0.9

    # Subsample trainval set
    (
        X_rgb_sub, _, X_thermal_sub, _, y_cls_sub, _, y_box_sub, _, y_mask_sub, _,
        y_triplet_box_sub, _, box_feat_sub, _, awir_clip_emb_sub, _, awir_dinov2_emb_sub, _,
        awir_mae_emb_sub, _, composite_label_encoded_sub, _
    ) = train_test_split(
        X_rgb_trainval, X_thermal_trainval, y_cls_trainval, y_box_trainval, y_mask_trainval,
        y_triplet_box_trainval, box_feat_trainval, awir_clip_emb_trainval, awir_dinov2_emb_trainval,
        awir_mae_emb_trainval, composite_label_encoded_trainval,
        train_size=train_frac, stratify=composite_label_encoded_trainval, random_state=42
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_rgb_sub, composite_label_encoded_sub)):
        print(f"  Fold {fold+1}")
        # Train/val split
        X_rgb_train, X_rgb_val = X_rgb_sub[train_idx], X_rgb_sub[val_idx]
        X_thermal_train, X_thermal_val = X_thermal_sub[train_idx], X_thermal_sub[val_idx]
        y_cls_train, y_cls_val = y_cls_sub[train_idx], y_cls_sub[val_idx]
        y_box_train, y_box_val = y_box_sub[train_idx], y_box_sub[val_idx]
        y_mask_train, y_mask_val = y_mask_sub[train_idx], y_mask_sub[val_idx]
        y_triplet_box_train, y_triplet_box_val = y_triplet_box_sub[train_idx], y_triplet_box_sub[val_idx]
        box_feat_train, box_feat_val = box_feat_sub[train_idx], box_feat_sub[val_idx]
        awir_clip_emb_train, awir_clip_emb_val = awir_clip_emb_sub[train_idx], awir_clip_emb_sub[val_idx]
        awir_dinov2_emb_train, awir_dinov2_emb_val = awir_dinov2_emb_sub[train_idx], awir_dinov2_emb_sub[val_idx]
        awir_mae_emb_train, awir_mae_emb_val = awir_mae_emb_sub[train_idx], awir_mae_emb_sub[val_idx]

        # Encode class labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_cls_train)
        y_val_encoded = label_encoder.transform(y_cls_val)
        y_test_encoded = label_encoder.transform(y_cls_test)

        for experiment_used in ['clip', 'dinov2', 'mae']:
            print('training', experiment_used, 'projection', flush=True)
            awir_fm_emb_loaded = np.load(f"/blue/azare/zhou.m/dissertation_comparisons/awir_{experiment_used}_emb.npy")
            input_dim = awir_fm_emb_loaded.shape[1]
            print("Loaded embeddings shape:", awir_fm_emb_loaded.shape)

            # Select embeddings
            if experiment_used == 'clip':
                train_embeddings, val_embeddings, test_embeddings = awir_clip_emb_train, awir_clip_emb_val, awir_clip_emb_test
            elif experiment_used == 'dinov2':
                train_embeddings, val_embeddings, test_embeddings = awir_dinov2_emb_train, awir_dinov2_emb_val, awir_dinov2_emb_test
            elif experiment_used == 'mae':
                train_embeddings, val_embeddings, test_embeddings = awir_mae_emb_train, awir_mae_emb_val, awir_mae_emb_test

            base_save_path = f"/blue/azare/zhou.m/awir/paper_results/trained_models/exp1_emb_proj/{experiment_used}"
            run_number = fold + 1
            dtl_checkpoint_save_path = f"{base_save_path}/dtl/best_val_model_run_{run_number}.h5"
            os.makedirs(os.path.dirname(dtl_checkpoint_save_path), exist_ok=True)
            dtl_ckpt = ModelCheckpoint(dtl_checkpoint_save_path, monitor='val_loss', save_best_only=True, verbose=0)

            # DTL TRAINING
            dtl_model = ctl_projection_headv2(input_dim)
            dtl_model.compile(optimizer=Adam(0.0001), loss=[keras_batch_all_triplet_loss(margin=parameters['margin'])])
            dtl_start_time = time.time()
            dtl_model.fit(
                train_embeddings,
                y_train_encoded,
                validation_data=(val_embeddings, y_val_encoded),
                epochs=parameters['epochs'],
                batch_size=parameters['batch_size'],
                verbose=0,
                callbacks=[dtl_ckpt]
            )
            dtl_train_duration = time.time() - dtl_start_time
            print(f"DTL training completed in {dtl_train_duration:.2f} seconds.")

            # Load model from checkpoint
            loaded_dtl_model = ctl_projection_headv2(input_dim)
            loaded_dtl_model.load_weights(dtl_checkpoint_save_path)

            # Get projection outputs on test embeddings
            dtl_proj_test = dtl_model.get_layer("ctl_proj").output
            dtl_projection_model = tf.keras.Model(inputs=dtl_model.input, outputs=dtl_proj_test)
            dtl_test_proj = dtl_projection_model.predict(test_embeddings, verbose=0)
            print('dtl_test_proj.shape:', dtl_test_proj.shape)

            # Save projection outputs
            proj_save_dir = f"/blue/azare/zhou.m/awir/paper_results/exp1_test_projections/{experiment_used}/"
            os.makedirs(proj_save_dir, exist_ok=True)
            dtl_proj_save_path = os.path.join(proj_save_dir, f"dtl_proj_val{val_frac}_run_{run_number}.npy")
            np.save(dtl_proj_save_path, dtl_test_proj)
            print(f"Saved DTL test projections to {dtl_proj_save_path}")


if __name__ == "__main__":
    main()

                        
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default='rgb')
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--cls_weight", type=float, default=0.5)
    parser.add_argument("--num_comparison", type=int, default=32)
    args = parser.parse_args()
    
    parameters = {
    "epochs": 3000, 
    "iterations": 1,
    "runs": 1,
    "num_comparisons": args.num_comparison,
    "embedding_dimension": args.embedding_dimension,
    "margin" : args.margin,
    "batch_size": args.batch_size,
    "test_size": args.test_size,
    "modality": args.modality,
    "cls_weight": args.cls_weight
}

    # Load the merged data
    f = np.load("data.npz")
    modality_chosen = args.modality
    print('Modality: ' + modality_chosen, flush=True)

    # Extract the necessary arrays
    rgb = f['rgb']  
    thermal = f['thermal']
    y_cls = f['class_label'] 
    y_box = f['box_label'] 
    y_box = np.clip(y_box, None, 300)
    
    y_mask = np.zeros((240, 300, 300, 1))

    # Iterate over each box in y_box and set the corresponding area in y_mask to 1
    for i, (xmin, xmax, ymin, ymax) in enumerate(y_box):
        y_mask[i, ymin:ymax, xmin:xmax, 0] = 1

    #Converting y_box to center, width, height
    
    # Image size (for normalization)
    image_width = 300
    image_height = 300

    # Initialize an array to hold the converted values
    # Shape: (num_samples, 4)
    num_samples = y_box.shape[0]
    y_box_cenwh = np.zeros((num_samples, 4))

    # Calculate center x, center y, width, height and normalize
    for i in range(num_samples):
        xmin, xmax, ymin, ymax = y_box[i]
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        # Normalize by the image size
        y_box_cenwh[i, 0] = xcenter / image_width      # xcenter normalized
        y_box_cenwh[i, 1] = ycenter / image_height     # ycenter normalized
        y_box_cenwh[i, 2] = width / image_width         # width normalized
        y_box_cenwh[i, 3] = height / image_height       # height normalized
        
    # One-hot encode the class labels
    encoder = LabelEncoder()
    one_hot_encoded_classes = encoder.fit_transform(y_cls)

    # y_triplet_box_label, _ = generate_triplet_box_label_one_hot(y_box, y_cls, n_clusters=3)
    
    y_triplet_box_label, features = generate_triplet_box_label_normalized(y_box, y_cls, n_clusters=3)
    
    
    
    # Dictionary to map integers to box descriptions
    box_label_mapping = {
        0: "small elongated box",
        1: "large elongated box",
        2: "small square box"
    }
    
    # Ensure one-hot encoded classes are converted to integer labels
    if len(one_hot_encoded_classes.shape) > 1:
        one_hot_encoded_classes = np.argmax(one_hot_encoded_classes, axis=1)

    # Create a composite label by combining class and triplet labels
    composite_label = [f"{cls}_{triplet}" for cls, triplet in zip(one_hot_encoded_classes, y_triplet_box_label)]

    # Optionally encode the composite labels into integers
    composite_label_encoded = LabelEncoder().fit_transform(composite_label)
        
    rgb_normalized = global_max_normalize(rgb)
    thermal_normalized = global_max_normalize(thermal)
    
    print('rgb_norm min max: ', rgb_normalized.min(), rgb_normalized.max())
    print('thermal_norm min max: ', thermal_normalized.min(), thermal_normalized.max())
  
    test_size = args.test_size
    print('testing size: ' + str(test_size))
    
    tile_labels = assign_tile_labels(y_box_cenwh)
    
    awir_dinov2_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_dinov2_emb.npy")
    awir_clip_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_clip_emb.npy")
    awir_mae_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_mae_emb.npy")



    (
        X_rgb_trainval, X_rgb_test,
        X_thermal_trainval, X_thermal_test,
        y_cls_trainval, y_cls_test,
        y_box_trainval, y_box_test,
        y_mask_trainval, y_mask_test,
        y_triplet_box_trainval, y_triplet_box_test,
        box_feat_trainval, box_feat_test,
        awir_clip_emb_trainval, awir_clip_emb_test,
        awir_dinov2_emb_trainval, awir_dinov2_emb_test,
        awir_mae_emb_trainval, awir_mae_emb_test,
        composite_label_encoded_trainval, composite_label_encoded_test
    ) = train_test_split(
        rgb_normalized, thermal_normalized, y_cls, y_box_cenwh, y_mask, y_triplet_box_label,
        features, awir_clip_emb_loaded, awir_dinov2_emb_loaded, awir_mae_emb_loaded,
        composite_label_encoded,
        test_size=0.2, stratify=composite_label_encoded, random_state=42
    )
    
    val_frac = 0.1
    train_frac = 0.9

    # Subsample the trainval set to keep only the fraction requested
    (
        X_rgb_sub, _,
        X_thermal_sub, _,
        y_cls_sub, _,
        y_box_sub, _,
        y_mask_sub, _,
        y_triplet_box_sub, _,
        box_feat_sub, _,
        awir_clip_emb_sub, _,
        awir_dinov2_emb_sub, _,
        awir_mae_emb_sub, _,
        composite_label_encoded_sub, _
    ) = train_test_split(
        X_rgb_trainval, X_thermal_trainval, y_cls_trainval, y_box_trainval, y_mask_trainval,
        y_triplet_box_trainval, box_feat_trainval,
        awir_clip_emb_trainval, awir_dinov2_emb_trainval, awir_mae_emb_trainval,
        composite_label_encoded_trainval,
        train_size=train_frac, stratify=composite_label_encoded_trainval, random_state=42
    )

    # Cross-validation on the subsample
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    # To store timing statistics per model
    ctl_times_dict = {m: [] for m in ['clip', 'dinov2', 'mae']}
    mtl_times_dict = {m: [] for m in ['clip', 'dinov2', 'mae']}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_rgb_sub, composite_label_encoded_sub)):
        print(f"  Fold {fold+1}")

        # Train/val split
        X_rgb_train, X_rgb_val = X_rgb_sub[train_idx], X_rgb_sub[val_idx]
        X_thermal_train, X_thermal_val = X_thermal_sub[train_idx], X_thermal_sub[val_idx]
        y_cls_train, y_cls_val = y_cls_sub[train_idx], y_cls_sub[val_idx]
        y_box_train, y_box_val = y_box_sub[train_idx], y_box_sub[val_idx]
        y_mask_train, y_mask_val = y_mask_sub[train_idx], y_mask_sub[val_idx]
        y_triplet_box_train, y_triplet_box_val = y_triplet_box_sub[train_idx], y_triplet_box_sub[val_idx]
        box_feat_train, box_feat_val = box_feat_sub[train_idx], box_feat_sub[val_idx]
        awir_clip_emb_train, awir_clip_emb_val = awir_clip_emb_sub[train_idx], awir_clip_emb_sub[val_idx]
        awir_dinov2_emb_train, awir_dinov2_emb_val = awir_dinov2_emb_sub[train_idx], awir_dinov2_emb_sub[val_idx]
        awir_mae_emb_train, awir_mae_emb_val = awir_mae_emb_sub[train_idx], awir_mae_emb_sub[val_idx]

        print('Number of training samples:', X_rgb_train.shape[0])
        print('Number of validation samples:', X_rgb_val.shape[0])

        # Encode class labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_cls_train)
        y_val_encoded = label_encoder.transform(y_cls_val)
        y_test_encoded = label_encoder.transform(y_cls_test)

        y_train_class_categorical = tf.keras.utils.to_categorical(y_train_encoded)
        y_val_class_categorical = tf.keras.utils.to_categorical(y_val_encoded)

        box_loc_train = y_box_train[:, :2]
        box_loc_val = y_box_val[:, :2]   
        box_loc_test = y_box_test[:, :2]

        train_features_dict = {"box_features": box_feat_train, "class_labels": y_train_class_categorical}
        val_features_dict = {"box_features": box_feat_val, "class_labels": y_val_class_categorical}
        weights_dict = {"box_features": 1.0, "class_labels": 0.5}
        normalize_dict = {"box_features": False, "class_labels": False}

        # combined_train = weigh_features_multi(train_features_dict, weights_dict, normalize_dict)
        # combined_val = weigh_features_multi(val_features_dict, weights_dict, normalize_dict)

        # print('combined_train.shape:', combined_train.shape)
        # print('combined_val.shape:', combined_val.shape)

        for experiment_used in ['clip', 'dinov2', 'mae']:

            print('training', experiment_used, 'projection', flush=True)

            awir_fm_emb_loaded = np.load(f"/blue/azare/zhou.m/dissertation_comparisons/awir_{experiment_used}_emb.npy")
            input_dim = awir_fm_emb_loaded.shape[1]
            print("Loaded embeddings shape:", awir_fm_emb_loaded.shape)

            # Select embeddings
            if experiment_used == 'clip':
                train_embeddings, val_embeddings, test_embeddings = awir_clip_emb_train, awir_clip_emb_val, awir_clip_emb_test
            elif experiment_used == 'dinov2':
                train_embeddings, val_embeddings, test_embeddings = awir_dinov2_emb_train, awir_dinov2_emb_val, awir_dinov2_emb_test
            elif experiment_used == 'mae':
                train_embeddings, val_embeddings, test_embeddings = awir_mae_emb_train, awir_mae_emb_val, awir_mae_emb_test

            base_save_path = f"/blue/azare/zhou.m/awir/paper_results/trained_models/exp1_emb_proj/{experiment_used}"
            run_number = fold + 1

            dtl_checkpoint_save_path = f"{base_save_path}/dtl/best_val_model_run_{run_number}.h5"
            # ctl_checkpoint_save_path = f"{base_save_path}/ctl/best_val_model_run_{run_number}.h5"
            # mtl_checkpoint_save_path = f"{base_save_path}/mtl/best_val_model_run_{run_number}.h5"

            os.makedirs(os.path.dirname(dtl_checkpoint_save_path), exist_ok=True)
            # os.makedirs(os.path.dirname(ctl_checkpoint_save_path), exist_ok=True)
            # os.makedirs(os.path.dirname(mtl_checkpoint_save_path), exist_ok=True)

            dtl_ckpt = ModelCheckpoint(dtl_checkpoint_save_path, monitor='val_loss', save_best_only=True, verbose=0)
            # ctl_ckpt = ModelCheckpoint(ctl_checkpoint_save_path, monitor='val_loss', save_best_only=True, verbose=0)
            # mtl_ckpt = ModelCheckpoint(mtl_checkpoint_save_path, monitor='val_loss', save_best_only=True, verbose=0)

#             # ------------------- CTL TRAINING -------------------
#             ctl_model = ctl_projection_headv2(input_dim)
#             ctl_model.compile(optimizer=Adam(0.0001), loss=[keras_batch_all_triplet_continuous_loss_final()])

#             ctl_start_time = time.time()
#             ctl_model.fit(
#                 train_embeddings,
#                 combined_train,
#                 validation_data=(val_embeddings, combined_val),
#                 epochs=parameters['epochs'],
#                 batch_size=parameters['batch_size'],
#                 verbose=0,
#                 callbacks=[ctl_ckpt]
#             )
#             ctl_train_duration = time.time() - ctl_start_time
#             print(f"CTL training completed in {ctl_train_duration:.2f} seconds.")
#             ctl_times_dict[experiment_used].append(ctl_train_duration)
            
            # ------------------- DTL TRAINING -------------------
            dtl_model = ctl_projection_headv2(input_dim)
            dtl_model.compile(optimizer=Adam(0.0001), loss=[keras_batch_all_triplet_loss(margin = parameters['margin'])])

            dtl_start_time = time.time()
            dtl_model.fit(
                train_embeddings,
                y_train_encoded,
                validation_data=(val_embeddings, y_val_encoded),
                epochs=parameters['epochs'],
                batch_size=parameters['batch_size'],
                verbose=0,
                callbacks=[dtl_ckpt]
            )
            dtl_train_duration = time.time() - dtl_start_time
            print(f"DTL training completed in {dtl_train_duration:.2f} seconds.")
            # dtl_times_dict[experiment_used].append(dtl_train_duration)
            

#             # ------------------- MTL TRAINING -------------------
#             mtl_model = multitask_head_v2(train_embeddings.shape[1], num_classes=3, output_dim=box_feat_train.shape[1])
#             mtl_model.compile(optimizer=Adam(0.0001), loss=['categorical_crossentropy', 'mse'])

#             mtl_start_time = time.time()
#             mtl_model.fit(
#                 train_embeddings, [y_train_class_categorical, box_feat_train],
#                 validation_data=(val_embeddings, [y_val_class_categorical, box_feat_val]),
#                 epochs=parameters['epochs'],
#                 batch_size=parameters['batch_size'],
#                 verbose=0,
#                 callbacks=[mtl_ckpt]
#             )
#             mtl_train_duration = time.time() - mtl_start_time
#             print(f"MTL training completed in {mtl_train_duration:.2f} seconds.")
#             mtl_times_dict[experiment_used].append(mtl_train_duration)
            
            # Load models from saved checkpoints
            

            loaded_dtl_model = ctl_projection_headv2(input_dim)
            loaded_dtl_model.load_weights(dtl_checkpoint_save_path)
            
#             loaded_ctl_model = ctl_projection_headv2(input_dim)
#             loaded_ctl_model.load_weights(ctl_checkpoint_save_path)
            
#             loaded_mtl_model = multitask_head_v2(train_embeddings.shape[1], num_classes=3, output_dim=box_feat_train.shape[1])
#             loaded_mtl_model.load_weights(mtl_checkpoint_save_path)
                
                
            # ctl_model = tf.keras.models.load_model(ctl_checkpoint_save_path, compile=False)
            # mtl_model = tf.keras.models.load_model(mtl_checkpoint_save_path, compile=False)

            # print(f"Reloaded CTL model from {ctl_checkpoint_save_path}")
            # print(f"Reloaded MTL model from {mtl_checkpoint_save_path}")
            
            # ===== Get CTL and MTL projection outputs on test embeddings =====
            dtl_proj_test = dtl_model.get_layer("ctl_proj").output
            # ctl_proj_test = ctl_model.get_layer("ctl_proj").output
            # mtl_proj_test = mtl_model.get_layer("ctl_proj").output

            # Compute the actual projections
            dtl_projection_model = tf.keras.Model(inputs=dtl_model.input, outputs=dtl_proj_test)
            # ctl_projection_model = tf.keras.Model(inputs=ctl_model.input, outputs=ctl_proj_test)
            # mtl_projection_model = tf.keras.Model(inputs=mtl_model.input, outputs=mtl_proj_test)

            dtl_test_proj = dtl_projection_model.predict(test_embeddings, verbose=0)
            # ctl_test_proj = ctl_projection_model.predict(test_embeddings, verbose=0)
            # mtl_test_proj = mtl_projection_model.predict(test_embeddings, verbose=0)

            print('dtl_test_proj.shape: ', dtl_test_proj.shape)
            # print('ctl_test_proj.shape: ', ctl_test_proj.shape)
            # print('mtl_test_proj.shape: ', mtl_test_proj.shape)

            # ===== Save projection outputs =====
            proj_save_dir = f"/blue/azare/zhou.m/awir/paper_results/exp1_test_projections/{experiment_used}/"
            os.makedirs(proj_save_dir, exist_ok=True)

            dtl_proj_save_path = os.path.join(proj_save_dir, f"dtl_proj_val{val_frac}_run_{run_number}.npy")
            # ctl_proj_save_path = os.path.join(proj_save_dir, f"ctl_proj_val{val_frac}_run_{run_number}.npy")
            # mtl_proj_save_path = os.path.join(proj_save_dir, f"mtl_proj_val{val_frac}_run_{run_number}.npy")

            np.save(dtl_proj_save_path, dtl_test_proj)
            # np.save(ctl_proj_save_path, ctl_test_proj)
            # np.save(mtl_proj_save_path, mtl_test_proj)

            print(f"Saved DTL test projections to {dtl_proj_save_path}") 
            # print(f"Saved CTL test projections to {ctl_proj_save_path}")
            # print(f"Saved MTL test projections to {mtl_proj_save_path}")

#     # ------------------- SAVE AVG AND STD -------------------
#     for experiment_used in ['clip', 'dinov2', 'mae']:
#         ctl_times = np.array(ctl_times_dict[experiment_used])
#         mtl_times = np.array(mtl_times_dict[experiment_used])

#         ctl_mean, ctl_std = np.mean(ctl_times), np.std(ctl_times)
#         mtl_mean, mtl_std = np.mean(mtl_times), np.std(mtl_times)

#         timing_save_dir = f"/blue/azare/zhou.m/awir/paper_results/training_times/{experiment_used}/"
#         os.makedirs(timing_save_dir, exist_ok=True)

#         np.save(os.path.join(timing_save_dir, f"ctl_time_stats_val{val_frac}.npy"), np.array([ctl_mean, ctl_std]))
#         np.save(os.path.join(timing_save_dir, f"mtl_time_stats_val{val_frac}.npy"), np.array([mtl_mean, mtl_std]))

#         print(f"Saved CTL timing stats for {experiment_used}: mean={ctl_mean:.2f}s, std={ctl_std:.2f}s")
#         print(f"Saved MTL timing stats for {experiment_used}: mean={mtl_mean:.2f}s, std={mtl_std:.2f}s")