import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, f_classif


def compute_per_sample_metric(y_cls_encoded, y_box_features, metric='mi'):
    """
    Compute per-sample MI or ANOVA between classification labels and box features.
    
    Args:
        y_cls_encoded: numpy array, shape (N,)
        y_box_features: numpy array, shape (N, 4)
        metric: 'mi' or 'anova'
    
    Returns:
        sample_metric: numpy array, shape (N,)
    """
    # Normalize box features to [0,1] for stability
    scaler = MinMaxScaler()
    X = scaler.fit_transform(y_box_features)
    y = y_cls_encoded

    if metric == 'mi':
        # MI per feature
        mi_per_feature = mutual_info_classif(X, y, discrete_features=False)
        # Per-sample metric: weighted sum over features
        sample_metric = np.sum(X * mi_per_feature, axis=1)
    elif metric == 'anova':
        fvals, _ = f_classif(X, y)
        sample_metric = np.sum(X * fvals, axis=1)
    else:
        raise ValueError("metric must be 'mi' or 'anova'")
    
    return sample_metric


def generate_triplet_box_label_normalized(
    y_box,
    y_cls,
    n_clusters=3,
    eps=1e-12,
    random_state=42,
):
    """
    Keeps original prediction targets:
        - norm_width
        - norm_height
        - norm_area
        - norm_symmetric_squareness_deviation

    Returns:
        y_box_triplet_label : (N,)
        features            : (N, K)
            First 4 columns are:
                [norm_width, norm_height, norm_area, norm_symmetric_squareness_deviation]
            Remaining columns are additional continuous features.
    """

    # -------------------------------------------------
    # Basic geometry
    # -------------------------------------------------
    x0, x1, y0, y1 = y_box[:, 0], y_box[:, 1], y_box[:, 2], y_box[:, 3]

    w = np.maximum(x1 - x0, 0.0)
    h = np.maximum(y1 - y0, 0.0)

    area = w * h
    perimeter = 2.0 * (w + h)
    diagonal = np.sqrt(w**2 + h**2)

    aspect_ratio = w / (h + eps)
    log_aspect = np.log(aspect_ratio + eps)

    symmetric_squareness_deviation = 1.0 - np.minimum(
        w / (h + eps),
        h / (w + eps),
    )

    thinness_ratio = (4.0 * np.pi * area) / (perimeter**2 + eps)

    # Second-moment quantities
    Ixx = (w * h**3) / 12.0
    Iyy = (h * w**3) / 12.0
    log_inertia_ratio = np.log((Ixx + eps) / (Iyy + eps) + eps)
    polar_moment = Ixx + Iyy
    radius_gyration_sq = polar_moment / (area + eps)

    log_area = np.log(area + eps)
    shape_angle = np.arctan2(h, w + eps)

    # -------------------------------------------------
    # DataFrame
    # -------------------------------------------------
    df = pd.DataFrame({
        "class": y_cls,
        "width": w,
        "height": h,
        "area": area,
        "perimeter": perimeter,
        "diagonal": diagonal,
        "aspect_ratio": aspect_ratio,
        "log_aspect": log_aspect,
        "symmetric_squareness_deviation": symmetric_squareness_deviation,
        "thinness_ratio": thinness_ratio,
        "log_inertia_ratio": log_inertia_ratio,
        "radius_gyration_sq": radius_gyration_sq,
        "log_area": log_area,
        "shape_angle": shape_angle,
    })

    # -------------------------------------------------
    # Min–max normalize all numeric columns except class
    # -------------------------------------------------
    numeric_cols = [c for c in df.columns if c != "class"]

    for col in numeric_cols:
        v = df[col].to_numpy(dtype=np.float64)
        vmin, vmax = v.min(), v.max()
        df[f"norm_{col}"] = (v - vmin) / (vmax - vmin + eps)

    # -------------------------------------------------
    # Build features matrix
    # First 4 MUST stay the same for compatibility
    # -------------------------------------------------
    base_feature_order = [
        "norm_width", #0
        "norm_height", #1
        "norm_area", #2
        "norm_symmetric_squareness_deviation", #3
    ]

    # Additional continuous features
    extra_feature_order = [
        "norm_perimeter", #4
        "norm_diagonal", #5
        "norm_aspect_ratio", #6
        "norm_log_aspect", #7
        "norm_thinness_ratio", #8
        "norm_log_inertia_ratio", #9
        "norm_radius_gyration_sq", #10
        "norm_log_area", #11
        "norm_shape_angle", #12
    ]

    features = df[base_feature_order + extra_feature_order].to_numpy(dtype=np.float32)

    # -------------------------------------------------
    # Clustering still based on the original 4
    # -------------------------------------------------
    clustering_features = df[base_feature_order].to_numpy(dtype=np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df["y_box_triplet_label"] = kmeans.fit_predict(clustering_features)

    return df["y_box_triplet_label"].to_numpy(), features


def global_max_normalize(data):
    """
    Perform Max Normalization using the global maximum value of the input data.

    Parameters:
    - data (numpy.ndarray): Input array of shape (n_samples, height, width, channels)

    Returns:
    - normalized_data (numpy.ndarray): Max normalized array with the same shape as input

    """
    # Find the global maximum value across the entire dataset
    global_max = np.max(data)
    print(global_max)

    # Perform Max normalization by dividing by the global maximum
    normalized_data = data / global_max

    # Clip values to ensure they stay in the range [0, 1]
    # normalized_data = np.clip(normalized_data, 0, 1)

    return normalized_data


def assign_tile_labels(function_y_box_cenwh):
    """
    Assign discrete labels to bounding boxes based on their (x_center, y_center) positions 
    in a 3x3 grid.

    Parameters:
        y_box_cenwh (array): Array of bounding boxes with shape (num_boxes, 4), where each row 
                             contains [x_center, y_center, width, height] (normalized by image size).

    Returns:
        array: An array of discrete tile labels for each bounding box.
    """
    # Extract x_center and y_center from y_box_cenwh
    x_center = function_y_box_cenwh[:, 0]
    y_center = function_y_box_cenwh[:, 1]
    
    # Compute tile indices for the x and y positions
    x_tile = np.floor(x_center * 3).astype(int)  # 0, 1, or 2
    y_tile = np.floor(y_center * 3).astype(int)  # 0, 1, or 2

    # Clip values to ensure they are within the range [0, 2] (in case of edge cases like 1.0)
    x_tile = np.clip(x_tile, 0, 2)
    y_tile = np.clip(y_tile, 0, 2)
    
    # Compute the tile label as (row_index * 3 + column_index)
    tile_labels = y_tile * 3 + x_tile

    return tile_labels
