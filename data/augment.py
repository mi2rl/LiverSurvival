from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import resize_segmentation, pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from custom_augment import Convert3DTo2DTransform, Convert2DTo3DTransform

import torchvision.transforms as transforms
import numpy as np

params = {

    'patch_size': (64, 180, 240),
    #'patch_size': (64, 360, 480),
    
    "do_elastic": False,
    #"elastic_deform_alpha": (0., 900.),
    "elastic_deform_alpha": (0., 200.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.3,
    
    "do_scaling": True,
    #"scale_range": (0.7, 1.3),
    "scale_range": (0.8, 1.2),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis":1,
    "p_scale": 0.2,

    "do_rotation": True,
    #"rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi),
    "rotation_z": (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.5,

    "random_crop": False,
    "random_crop_dist_to_border": None,
    
    "do_gamma": True,
    "gamma_retain_stats": True,
    #"gamma_range": (0.5, 1.6),
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": False,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": True,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    #"do_additive_brightness": True,
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.3,
    "additive_brightness_p_per_channel": 1,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.2,
}
#-2.2531
tr_transforms=[]

if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    tr_transforms.append(Convert3DTo2DTransform())
    patch_size_spatial = params.get("patch_size")[1:]

tr_transforms.append(SpatialTransform(
    patch_size_spatial, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"), 
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

if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    tr_transforms.append(Convert2DTo3DTransform())

tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
#tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
#                                           p_per_channel=0.5))
#tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.8, 1.2), p_per_sample=0.15))

if params.get("do_additive_brightness"):
    tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                             params.get("additive_brightness_sigma"),
                                             True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                             p_per_channel=params.get("additive_brightness_p_per_channel")))

tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
#tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
#                                                    p_per_channel=0.5,
#                                                    order_downsample=0, order_upsample=3, p_per_sample=0.25,
#                                                    ignore_axes=None))

tr_transforms.append(
    GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                   p_per_sample=0.1))  # inverted gamma

if params.get("do_gamma"):
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=params["p_gamma"]))

#if params.get("do_mirror") or params.get("mirror"):
#    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

#tr_transforms.append(RenameTransform('seg', 'target', True))
#tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
tr_transforms.append(NumpyToTensor(['data'], 'float'))
tr_transforms = Compose(tr_transforms)



val_transforms=[]
#val_transforms.append(RenameTransform('seg', 'target', True))
#val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
val_transforms.append(NumpyToTensor(['data'], 'float'))
val_transforms = Compose(val_transforms)