import os
import sys
import h5py

import numpy as np
import SimpleITK as sitk


###########################################################
# Path Process
###########################################################
def file_name_path( file_dir, type='file' ):
    """
    get root path,sub_dirs,all_sub_files
    :param type: string, dir or file
    :param file_dir: string, target directory path
    :return:
    """
    assert type in ['file', 'dir']
    for root, dirs, files in os.walk(file_dir):
        if type == 'dir':
            print("sub_dirs:", dirs)
            return dirs
        elif type == 'file':
            print("files:", files)
            return files
        else:
            raise NotImplementedError('Type not implemented')


def train_val_split( image_items, mask_items, ratio ):
    """

    :param image_items:
    :param mask_items:
    :param ratio:
    :return:
    """
    assert len(image_items) == len(mask_items)
    indexes = np.arange(len(image_items))
    np.random.shuffle(indexes)

    length = int(len(image_items) * ratio)
    train_indexes = indexes[:length]
    val_indexes = indexes[length:]

    train_image_items = [image_items[index] for index in train_indexes]
    val_image_items = [image_items[index] for index in val_indexes]
    train_mask_items = [mask_items[index] for index in train_indexes]
    val_mask_items = [mask_items[index] for index in val_indexes]

    return train_image_items, train_mask_items, val_image_items, val_mask_items


###########################################################
# SimpleITK Based
###########################################################
def load_image( path, type=sitk.sitkFloat64 ):
    """
    Load NIFTI Image by SimpleITK
    :param path:
    :param type:
    :return:
    """
    image = sitk.Cast(sitk.ReadImage(path), type)
    return image


def truncation( image, lower=-200, upper=200 ):
    """
    Truncation Nifti Image
    :param image: sitk.Image, target image
    :param lower: float, lower boundary
    :param upper: float, upper boundary
    :return: sitk.Image instance
    """
    image = sitk.IntensityWindowing(image,
                                    windowMinimum=lower,
                                    windowMaximum=upper,
                                    outputMinimum=0,
                                    outputMaximum=255)
    return image


def resize( image, new_spacing, origin_spacing, method, dtype=sitk.sitkFloat32 ):
    """
    Resize NIFTI Image
    :param image:
    :param new_spacing:
    :param origin_spacing:
    :param method:
    :param dtype:
    :return:
    """
    image = sitk.Cast(image, dtype)

    origin_spacing = np.array(origin_spacing, np.float)
    new_spacing = np.array(new_spacing, np.float)

    origin_size = image.GetSize()
    new_size = np.array(origin_size * origin_spacing / new_spacing, dtype=np.int)

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetReferenceImage(image)
    resample_filter.SetOutputSpacing(new_spacing.tolist())
    resample_filter.SetSize(new_size.tolist())
    resample_filter.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resample_filter.SetInterpolator(method)
    resample_image = resample_filter.Execute(image)

    return resample_image


###########################################################
# Patch Process Function
###########################################################
def sample( image, mask, size=(128, 128, 32), overlap=(6, 6, 6) ):
    """
    Sample Image by numpy
    :param image:
    :param mask:
    :param size:
    :param overlap:
    :return:
    """
    assert np.shape(image) == np.shape(mask)
    depth, height, width = np.shape(image)
    patch_depth, patch_height, patch_width = size
    patch_depth_overlap, patch_height_overlap, patch_width_overlap = overlap

    assert depth >= patch_depth and height >= patch_height and width >= patch_width

    stride_depth = patch_depth - patch_depth_overlap
    stride_height = patch_height - patch_depth_overlap
    stride_width = patch_width - patch_width_overlap

    image_patchs = list()
    mask_patchs = list()
    locations = list()

    for z in range(0, depth, stride_depth):
        for y in range(0, height, stride_height):
            for x in range(0, width, stride_width):
                over_width = False
                over_height = False
                over_depth = False
                if x + patch_width >= width:
                    start_x = width - patch_width
                    over_width = True
                else:
                    start_x = x
                if y + patch_height >= height:
                    start_y = height - patch_height
                    over_height = True
                else:
                    start_y = y
                if z + patch_depth >= depth:
                    start_z = depth - patch_depth
                    over_depth = True
                else:
                    start_z = z

                image_patchs.append(image[
                                    start_z:start_z + patch_depth,
                                    start_y:start_y + patch_height,
                                    start_x:start_x + patch_width
                                    ])

                mask_patchs.append(mask[
                                   start_z:start_z + patch_depth,
                                   start_y:start_y + patch_height,
                                   start_x:start_x + patch_width
                                   ])
                locations.append(
                    [start_z, start_y, start_x]
                )

                if over_width:
                    break
            if over_height:
                break
        if over_depth:
            break
    return image_patchs, mask_patchs, locations


def patch_filter( image_patchs, mask_patchs, background, rate, locations=None ):
    assert len(image_patchs) == len(mask_patchs) == len(locations) and len(image_patchs) > 0

    width, height, depth = np.shape(image_patchs[0])
    volume = width * height * depth

    filter_image_patchs = list()
    filter_mask_patchs = list()
    filter_locations = list()

    for index in range(len(image_patchs)):
        mask_volumn = np.sum(mask_patchs[index] > background)
        rate_passed = ((mask_volumn * 1.0 / volume * 1.0) > rate)
        if rate_passed:
            filter_image_patchs.append(image_patchs[index])
            filter_mask_patchs.append(mask_patchs[index])
            if locations is not None:
                filter_locations.append(locations[index])
    if locations is not None:
        return filter_image_patchs, filter_mask_patchs, filter_locations
    else:
        return filter_image_patchs, filter_mask_patchs


def assemble( patchs, shapes, locations ):
    assert len(shapes) == 3
    assert len(patchs) == len(locations)
    empty_array = np.empty(shape=shapes)
    patch_depth, patch_height, patch_width = np.shape(patchs[0])
    for i in range(len(patchs)):
        start_z, start_y, start_x = locations[i]
        empty_array[
        start_z:start_z + patch_depth,
        start_y:start_y + patch_height,
        start_x:start_x + patch_width
        ] = patchs[i]
    return empty_array


def assemble_channels( patchs, shapes, locations ):
    channel = patchs[0].shape[0]
    new_shapes = [channel] + list(shapes)
    patch_depth, patch_height, patch_width = np.shape(patchs[0])[1:]

    empty_array = np.empty(shape=new_shapes)
    for i in range(len(patchs)):
        start_z, start_y, start_x = locations[i]
        empty_array[
        :,
        start_z:start_z + patch_depth,
        start_y:start_y + patch_height,
        start_x:start_x + patch_width
        ] = patchs[i][0, :]

    return empty_array
