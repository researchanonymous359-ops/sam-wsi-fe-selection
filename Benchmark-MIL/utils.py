# utils.py
import os
import json
import random
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.subplots as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import MetricCollection, Accuracy, AUROC, Precision, Recall
from torchmetrics import F1Score as F1

def seed_everything(seed=42):
    pl.seed_everything(seed, workers=True) # ADDED
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_parameters(args, save_dir):
    print(args)
    args_dict = vars(args)
    with open(os.path.join(save_dir, "parameters.json"), "w") as f:
        json.dump(
            {n: str(args_dict[n]) for n in args_dict},
            f,
            indent=4
        )

class switch_dim(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.transpose(x, 2, 1)
        return x

def get_metrics(num_classes, task):
    metrics = MetricCollection({
        "ACC": Accuracy(num_classes=num_classes, task=task), # default: average="micro" -> might be misleading if classes are imbalanced
        # "Balanced_ACC": Accuracy(num_classes=num_classes, average="macro", task=task), # does not take label imbalance -> helping in situations where each class's prediction is equally important.
        "AUROC": AUROC(num_classes=num_classes, task=task),
        "Precision": Precision(num_classes=num_classes, task=task),
        "Recall" : Recall(num_classes=num_classes, task=task),
        "F1": F1(num_classes=num_classes, task=task)
    })

    return metrics

def get_loss_weight(args, data_module):
    if args.loss_weight is not None:
        loss_weight = args.loss_weight
    elif args.auto_loss_weight: # automatically calculate the weight of each class
        data_module.setup()
        loss_weight = data_module.dataset_train.get_weight_of_class()
    else:
        loss_weight = None
    if loss_weight is not None:
        print("Using loss weight:", loss_weight)
        loss_weight = torch.tensor(loss_weight)

    return loss_weight

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


def save_yaml(path, yaml_file:dict):
    assert ('.yaml' in path) or ('.yml' in path), 'Error: Attempting to save a file that is not a yaml(yml) file.'
    with open(fr'{path}', 'w') as p:
        yaml.dump(yaml_file, p)


def calculate_predictive_entropy(predictions):
    probabilities = torch.softmax(predictions, dim=-1)
    mean_probabilities = probabilities.mean(dim=0)  # Average over augmentations
    log_mean_probabilities = torch.log(mean_probabilities + 1e-12)  # Avoid log(0)
    entropy = -torch.sum(mean_probabilities * log_mean_probabilities)
    return entropy.item()


def scale_entropy(entropies, args):
    """
    Scales the predictive entropy values between 0 and args.smoothing_factor.

    Args:
        entropies (list[float]): List of entropy values.
        args: Arguments containing smoothing_factor.

    Returns:
        list[float]: Scaled entropy values.
    """
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    scaled_entropies = [
        (e - min_entropy) / (max_entropy - min_entropy + 1e-12) * args.smoothing_factor
        for e in entropies
    ]
    return scaled_entropies

def save_attention_map(slide_name, label_name, pred_name, coords, attention_map, patch_size, downsample, patch_path, save_path):
    """
    Generates and saves an attention map as an image.
    Args:
        slide_name (str): Name of the slide.
        label_name (str): True label name.
        pred_name (str): Predicted label name.
        coords (list): List of coordinates for patches.
        attention_map (np.ndarray): Attention map values.
        patch_size (int): Size of each patch.
        downsample (int): Downsampling factor.
        patch_path (str): Path to the folder containing patch images.
        save_path (Path): Path to save the attention map image.
    """
    # Ensure the attention_map is a numpy array
    if not isinstance(attention_map, np.ndarray):
        attention_map = np.array(attention_map)
    scale_factor = 1.0 / downsample
    patch_size = patch_size // downsample  # Adjust patch size for downsampling
    # convert coords into integer pairs
    int_pairs = [tuple(map(int, s[0].split('-'))) for s in coords]
    canvas_height = (max(coord[0] for coord in int_pairs)+1)*patch_size  # find max for x in coords
    canvas_width = (max(coord[1] for coord in int_pairs)+1)*patch_size  # find max for y in coords
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255 # create white canvas
    # add jet color bar
    cmap = plt.get_cmap('jet')    
    # Normalize attention map to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    for (y, x), coord, attn in zip(int_pairs, coords, attention_map):
        p_path = Path(patch_path) / label_name / slide_name / f"{slide_name}_{coord[0]}.png"
        # check if the path exists
        if not Path(p_path).exists():
            print(f"no patch img in {p_path}")
            continue
        with Image.open(p_path) as pil_img:
            # Resize the image to DOWNSAMPLED_PATCH_SIZE and convert to numpy array
            img = np.array(pil_img.resize((patch_size, patch_size), Image.LANCZOS))

        color = np.array(cmap(attn)[:3])  # drop alpha
        overlay = (color * 255).astype(np.uint8)  # constant RGB
        alpha = 0.4   # how “strong” the tint is; tweak between 0 (no tint) and 1 (solid color)
        tinted = (img.astype(float) * (1-alpha) + overlay * alpha).astype(np.uint8)

        x0, y0 = x * patch_size, y * patch_size
        canvas[y0:y0+patch_size, x0:x0+patch_size, :] = tinted
        del img, tinted, overlay, color
    
    fig, ax = plt.subplots(figsize=(canvas_width/100, canvas_height/100))
    ax.imshow(canvas)
    ax.axis('off')

    # Scale figure elements according to downsample factor
    colorbar_fraction = max(0.020, 0.046 * scale_factor)  # Scale the colorbar width
    label_fontsize = max(10, int(64 * scale_factor))  # Scale font but keep minimum size
    title_fontsize = max(14, int(128 * scale_factor))  # Scale title font

    title_str = f"Slide: {slide_name}\nTrue: {label_name}, Predicted: {pred_name}"
    ax.set_title(title_str, fontsize=title_fontsize) 

    # add color bar
    # Create a ScalarMappable for the color bar using the same colormap and normalization as for the attention map
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the color bar to the figure
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=colorbar_fraction, pad=max(0.01, 0.04 * scale_factor))
    cbar.set_label("Attention", rotation=270, labelpad=max(10, 40*scale_factor), fontsize=label_fontsize) 
    # Add "High" and "Low" text to the colorbar
    # Adjust ha, va, and coordinates as needed for desired placement
    cbar.ax.text(0.5, 1.02, "High", transform=cbar.ax.transAxes, 
                    ha='center', va='bottom', fontsize=label_fontsize) # Text above top-center
    cbar.ax.text(0.5, -0.02, "Low", transform=cbar.ax.transAxes, 
                    ha='center', va='top', fontsize=label_fontsize)    # Text below bottom-center
    plt.tight_layout(pad=0.5) # You can adjust the pad value

    # save canvas
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / f"attention_map_{slide_name}_{label_name}_resize{downsample}.jpg", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    del canvas, fig, ax, sm, cbar, cmap, attention_map, norm
    return