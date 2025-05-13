import torch
import json
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from hydra import initialize, compose
from sklearn.utils import class_weight as sklearn_class_weight


def load_df(file_path, raw_type):
    if raw_type == "parquet":
        return pd.read_parquet(file_path)
    elif raw_type == "csv":
        return pd.read_csv(file_path)


# def load_config(config_name="exp1"):
#     with initialize(version_base=None, config_path="../conf"):
#         cfg = compose(config_name=config_name)
#         return cfg


def load_config(config_name: str = "exp1"):
    # Split “group/subgroup/filename” into parts
    parts = config_name.split("/")
    # If there's a group path, prepend it to "../conf"
    if len(parts) > 1:
        subfolder = "/".join(parts[:-1])          # e.g. "experiment_type"
        filename = parts[-1]                     # e.g. "baseline"
        config_path = f"../conf/{subfolder}"      # relative path
    else:
        filename = parts[0]
        config_path = "../conf"

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=filename)
        return cfg


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalized=False,
                          file_path=None,
                          show_figure=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalized:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    if file_path:
        plt.savefig(file_path)
    if show_figure:
        plt.show()
    return fig


def calculate_fpr_fnr_with_global(cm):
    """
    Calculate FPR and FNR for each class and globally for a multi-class confusion matrix.

    Parameters:
        cm (numpy.ndarray): Confusion matrix of shape (num_classes, num_classes).

    Returns:
        dict: A dictionary containing per-class and global FPR and FNR.
    """
    num_classes = cm.shape[0]
    results = {"per_class": {}, "global": {}}

    # Initialize variables for global calculation
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    # Per-class calculation
    for class_idx in range(num_classes):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Calculate FPR and FNR for this class
        FPR = FP / (FP + TN) if (FP + TN) != 0 else None
        FNR = FN / (TP + FN) if (TP + FN) != 0 else None

        # Store per-class results
        results["per_class"][class_idx] = {"FPR": FPR, "FNR": FNR}

        # Update global counts
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Global calculation
    global_FPR = total_FP / \
        (total_FP + total_TN) if (total_FP + total_TN) != 0 else None
    global_FNR = total_FN / \
        (total_FN + total_TP) if (total_FN + total_TP) != 0 else None

    results["global"]["FPR"] = global_FPR
    results["global"]["FNR"] = global_FNR

    return results


def compute_class_weights(targets: pd.Series,
                          classes: np.ndarray,
                          version: str = 'v4',
                          device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Compute class weights for CrossEntropyLoss, selectable by `version`.

    Parameters
    ----------
    targets : pd.Series
        Series of integer class labels.
    classes : ndarray of shape (num_classes,)
        Array of all possible class labels (e.g. np.arange(num_classes)).
    version : str, one of {'v1', 'v2', 'v3', 'v4'}
        Which weighting strategy to use:
          - v1: total/(num_classes*count)  (naïve “balanced”)
          - v2: normalize(1/(count + eps))
          - v3: raw inverse (1/count)
          - v4: sklearn compute_class_weight('balanced')
    device : torch.device
        Where to put the resulting tensor.

    Returns
    -------
    torch.Tensor of shape (num_classes,)
        Float tensor of weights.
    """
    # get counts per class label
    counts = targets.value_counts().to_dict()
    num_classes = len(classes)
    counts_arr = np.zeros(num_classes, dtype=float)
    for lbl, cnt in counts.items():
        counts_arr[int(lbl)] = cnt

    if version == 'v1':
        # v1: total samples divided equally across classes
        total = counts_arr.sum()
        weights_arr = np.zeros_like(counts_arr)
        mask = counts_arr > 0
        weights_arr[mask] = total / (num_classes * counts_arr[mask])
        weight_tensor = torch.tensor(
            weights_arr, dtype=torch.float, device=device)

    elif version == 'v2':
        # v2: normalized inverse-frequency with epsilon
        class_counts = torch.tensor(
            counts_arr, dtype=torch.float, device=device)
        weight_tensor = 1.0 / (class_counts + 1e-6)
        weight_tensor = weight_tensor / weight_tensor.sum()

    elif version == 'v3':
        # v3: raw inverse-frequency
        counts_tensor = torch.tensor(
            counts_arr, dtype=torch.float, device=device)
        # avoid division by zero
        inv = torch.zeros_like(counts_tensor)
        mask = counts_tensor > 0
        inv[mask] = 1.0 / counts_tensor[mask]
        weight_tensor = inv

    elif version == 'v4':
        # v4: sklearn compute_class_weight
        present = np.array(list(counts.keys()), dtype=int)
        w_present = sklearn_class_weight.compute_class_weight(
            class_weight='balanced', classes=present, y=targets.values
        )
        weights_arr = np.zeros_like(counts_arr)
        for cls, w in zip(present, w_present):
            weights_arr[int(cls)] = w
        weight_tensor = torch.tensor(
            weights_arr, dtype=torch.float, device=device)

    else:
        raise ValueError(
            f"Unknown version '{version}', choose one of {{'v1','v2','v3','v4'}}")

    return weight_tensor


# def compute_class_weights(targets, classes):

#     counts = targets.value_counts().to_dict()
#     counts_arr = np.zeros(len(classes), dtype=float)
#     for lbl, cnt in counts.items():
#         counts_arr[lbl] = cnt

#     # version 1 - wrong
#     # total = counts_arr.sum()
#     # weights_arr = np.zeros_like(counts_arr)
#     # mask = counts_arr > 0
#     # weights_arr[mask] = total / (num_classes * counts_arr[mask])
#     # weight_tensor = torch.tensor(weights_arr).float().to(device)

#     # version 2
#     class_counts = torch.tensor(counts_arr, dtype=torch.float)
#     weights = 1.0 / (class_counts + 1e-6)
#     weights = weights / weights.sum()
#     weight_tensor = torch.FloatTensor(weights)

#     # version 3
#     # weight = 1. / counts_arr
#     # weight_tensor = torch.tensor(weight).float().to(device)

#     # version 4

#     # present = np.array(list(counts.keys()))

#     # weights_present = class_weight.compute_class_weight(
#     #     'balanced', classes=present, y=targets)
#     # weights = np.zeros(len(classes), dtype=float)
#     # for cls, w in zip(present, weights_present):
#     #     weights[cls] = w
#     # weight_tensor = torch.FloatTensor(weights)

#     return weight_tensor
