import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss
from dual_triplet_loss_clf_aspect import batch_all_triplet_double_loss
from continuous_triplet_loss import (
    batch_all_continuous_triplet_loss_final_lambda1,
    batch_all_continuous_triplet_loss_final_adjustable_comparisons,
)


def compute_selection_mask_tf(labels_clf, sample_metric,
                              top_percent=0.3, random_percent=0.1, seed=42):
    """
    TensorFlow version of top-percent + next-percent sample selection.
    No randomness. No tf.sets. No RNG.

    Args:
        labels_clf: (B,) int32
        sample_metric: (B,)
    Returns:
        mask: boolean tensor (B,)
    """
    labels_clf = tf.cast(labels_clf, tf.int32)
    sample_metric = tf.cast(sample_metric, tf.float32)

    # Unique classes in batch (dynamic)
    classes = tf.unique(labels_clf).y
    num_classes = tf.shape(classes)[0]

    # Initialize mask = all False
    N = tf.shape(labels_clf)[0]
    mask = tf.zeros(N, dtype=tf.bool)

    def process_class(c, mask):
        # indices for class c
        class_mask = tf.equal(labels_clf, c)
        class_idx = tf.where(class_mask)[:, 0]
        class_vals = tf.gather(sample_metric, class_idx)

        num_samples = tf.shape(class_idx)[0]
        num_samples_f = tf.cast(num_samples, tf.float32)

        # numbers to select
        n_top  = tf.maximum(1, tf.cast(num_samples_f * top_percent, tf.int32))
        n_more = tf.cast(num_samples_f * random_percent, tf.int32)

        # Sort values descending (largest first)
        sorted_vals, sorted_order = tf.sort(class_vals, direction="DESCENDING"), tf.argsort(class_vals, direction="DESCENDING")
        sorted_idx = tf.gather(class_idx, sorted_order)

        # Top group
        top_idx = sorted_idx[:n_top]

        # Next 10% (no randomness)
        remaining_idx = sorted_idx[n_top:]
        n_more = tf.minimum(n_more, tf.shape(remaining_idx)[0])
        more_idx = remaining_idx[:n_more]

        # Combine selection
        selected = tf.concat([top_idx, more_idx], axis=0)

        update = tf.tensor_scatter_nd_update(
            mask,
            tf.expand_dims(selected, 1),
            tf.ones_like(selected, dtype=tf.bool)
        )
        return update

    # Loop per-class
    def loop_body(i, mask):
        return process_class(classes[i], mask)

    mask = tf.foldl(lambda m, i: loop_body(i, m),
                    tf.range(num_classes),
                    initializer=mask)

    return mask


##################### NORMAL ##############


def keras_batch_all_triplet_loss(margin=0.0):
    def compute_keras_batch_all_triplet_loss(labels, embeddings):
        labels = K.squeeze(labels, axis=-1)
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin)
        return loss
    return compute_keras_batch_all_triplet_loss


def keras_batch_all_triplet_loss_hard(margin=0.0):
    def compute_keras_batch_all_triplet_loss_hard(labels, embeddings):
        labels = K.squeeze(labels, axis=-1)
        loss = batch_hard_triplet_loss(labels, embeddings, margin)
        return loss
    return compute_keras_batch_all_triplet_loss_hard


##################### DOUBLE ##############


def keras_batch_all_triplet_double_loss(margin=0.0, box_weight=0.5):
    def compute_keras_batch_all_triplet_double_loss(labels, embeddings):
        labels_clf = labels[:, 0]
        labels_AR = labels[:, 1]

        loss = batch_all_triplet_double_loss(labels_clf, labels_AR, embeddings, margin, box_weight)
        return loss
    return compute_keras_batch_all_triplet_double_loss


def keras_batch_all_triplet_double_loss_top_random(margin=0.0, box_weight=0.5,
                                                   top_percent=0.3, random_percent=0.1):
    def compute_loss(labels, embeddings):
        labels_clf = tf.cast(labels[:, 0], tf.int32)
        labels_AR = tf.cast(labels[:, 1], tf.int32)
        sample_metric = tf.cast(labels[:, 2], tf.float32)

        # Compute selection mask inside TF graph
        mask = compute_selection_mask_tf(
            labels_clf, sample_metric,
            top_percent=top_percent,
            random_percent=random_percent
        )

        # Apply mask
        embeddings_sel = tf.boolean_mask(embeddings, mask)
        labels_clf_sel = tf.boolean_mask(labels_clf, mask)
        labels_AR_sel = tf.boolean_mask(labels_AR, mask)

        loss = batch_all_triplet_double_loss(labels_clf_sel, labels_AR_sel, embeddings_sel, margin, box_weight)

        return loss

    return compute_loss


##################### CONTINUOUS ##############


def keras_batch_all_triplet_continuous_loss_final(margin=0.01):
    def compute_keras_batch_all_triplet_continuous_loss_final(box_features, embeddings):
        loss = batch_all_continuous_triplet_loss_final_lambda1(box_features, embeddings, margin, squared=False)
        return loss
    return compute_keras_batch_all_triplet_continuous_loss_final


def keras_batch_all_continuous_triplet_loss_final_adjustable_comparisons(num_comparisons):
    print('inside loss, num_comparisons: ', num_comparisons)
    def compute_keras_batch_all_continuous_triplet_loss_final_adjustable_comparisons(box_features, embeddings):
        loss = batch_all_continuous_triplet_loss_final_adjustable_comparisons(box_features, embeddings, num_comparisons, squared=False)
        return loss
    return compute_keras_batch_all_continuous_triplet_loss_final_adjustable_comparisons


##################### BOX REGRESSION ##############


def ciou_loss(y_true, y_pred, alpha=0.5):
    """
    CIoU loss for bounding box prediction.

    Parameters:
    - y_true: Ground truth bounding box coordinates (xcenter, ycenter, width, height) (normalized)
    - y_pred: Predicted bounding box coordinates (xcenter, ycenter, width, height) (normalized)
    - alpha: Trade-off parameter for aspect ratio term (default 0.5)

    Returns:
    - CIoU loss
    """
    # Extract coordinates
    x_center_true, y_center_true, width_true, height_true = tf.split(y_true, 4, axis=-1)
    x_center_pred, y_center_pred, width_pred, height_pred = tf.split(y_pred, 4, axis=-1)

    # Calculate corners of the boxes
    x1_true = x_center_true - (width_true / 2)
    x2_true = x_center_true + (width_true / 2)
    y1_true = y_center_true - (height_true / 2)
    y2_true = y_center_true + (height_true / 2)

    x1_pred = x_center_pred - (width_pred / 2)
    x2_pred = x_center_pred + (width_pred / 2)
    y1_pred = y_center_pred - (height_pred / 2)
    y2_pred = y_center_pred + (height_pred / 2)

    ### CIoU Loss Calculation ###

    # Calculate intersection
    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)

    # Calculate union
    true_area = width_true * height_true
    pred_area = width_pred * height_pred
    union_area = true_area + pred_area - inter_area

    # Calculate IoU
    iou = inter_area / tf.maximum(union_area, 1e-10)

    # Calculate center points and diagonal of the smallest enclosing box
    center_true = tf.concat([x_center_true, y_center_true], axis=-1)
    center_pred = tf.concat([x_center_pred, y_center_pred], axis=-1)

    # Distance between centers
    center_distance = tf.reduce_sum(tf.square(center_true - center_pred), axis=-1)

    # Calculate enclosing box diagonal
    c_x1 = tf.minimum(x1_true, x1_pred)
    c_x2 = tf.maximum(x2_true, x2_pred)
    c_y1 = tf.minimum(y1_true, y1_pred)
    c_y2 = tf.maximum(y2_true, y2_pred)

    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    c_diagonal_squared = tf.square(c_x2 - c_x1) + tf.square(c_y2 - c_y1)

    # Aspect ratio term
    v = (4 / (tf.square(tf.constant(np.pi)))) * tf.square(tf.atan(height_true / width_true) - tf.atan(height_pred / width_pred))

    # CIoU loss
    ciou_loss_value = 1 - iou + (center_distance / tf.maximum(c_diagonal_squared, 1e-10)) + alpha * v

    return tf.reduce_mean(ciou_loss_value)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for binary segmentation.

    Args:
        y_true (tensor): Ground truth binary mask.
        y_pred (tensor): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        dice_loss (float): Dice loss value.
    """
    # Flatten to simplify calculation
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Calculate intersection and union
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    # Dice coefficient and Dice loss
    dice_coefficient = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_coefficient

    return dice_loss


def ssim_loss(alpha=0.1):
    """
    Custom loss function that combines SSIM and MSE losses.
    """
    def inner_ssim_loss(y_true, y_pred):
        loss = 1 - tf.image.ssim(y_true, y_pred, 1.0, filter_size=7)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=(-3, -2, -1))
        combined_loss = tf.reduce_mean(alpha * loss + (1 - alpha) * mse_loss)
        return combined_loss

    return inner_ssim_loss
