from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import resize_segmentation, pad_nd_image
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter, SlimDataLoaderBase
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms import AbstractTransform
import torchvision.transforms as transforms
import numpy as np


params = {
    'patch_size': (96, 192, 192),
    "do_elastic": False,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0,
    
    "do_scaling": True,
    "scale_range": (0.8, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis":1,
    "p_scale": 0.2,

    "do_rotation": False,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (0 / 360 * 2. * np.pi, 0 / 360 * 2. * np.pi),
    "rotation_z": (0 / 360 * 2. * np.pi, 0 / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,
    
    "do_gamma": False,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": False,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
}
tr_transforms=[]
tr_transforms.append(SpatialTransform(
    params.get("patch_size"), patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"), 
    alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
    do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
    angle_z=params.get("rotation_z"),
    do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
    border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3,
    border_cval_seg=0, order_seg=1, 
    random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
    p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
    independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
))

tr_transforms.append(NumpyToTensor(['data'], 'float'))
tr_transforms = Compose(tr_transforms)



val_transforms=[]
val_transforms.append(NumpyToTensor(['data'], 'float'))
val_transforms = Compose(val_transforms)