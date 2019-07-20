import os
import logging

import h5py
import SimpleITK as sitk
import numpy as np

from tools.utils import load_image, truncation, resize, sample, patch_filter
from tools.utils import train_val_split, file_name_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###########################################################
# Data2HDF5
###########################################################
def write_data_hdf5( image_items,
                     mask_items,
                     hdf5_output_file,
                     process_filter_rate,
                     write_per_num=500,
                     process_resize_spacing=None,
                     sample_size=(128, 128, 32),
                     sample_overlap=(6, 6, 6),
                     window_lower=-200,
                     window_upper=300 ):
    # setting-up HDF5 file
    h5file = h5py.File(name=hdf5_output_file, mode='w')
    data = {}
    ds_shape = tuple([write_per_num] + list(sample_size))
    max_shape = tuple([None] + list(sample_size))
    data['image'] = h5file.create_dataset(name='image',
                                          shape=ds_shape,
                                          compression='gzip',
                                          dtype=np.float32,
                                          maxshape=max_shape,)
    data['mask'] = h5file.create_dataset(name='mask',
                                         shape=ds_shape,
                                         compression='gzip',
                                         dtype=np.uint8,
                                         maxshape=max_shape,)
    data['pid'] = h5file.create_dataset(name='patient_id',
                                        shape=tuple([write_per_num]),
                                        compression='gzip',
                                        dtype=h5py.special_dtype(vlen=np.str),
                                        maxshape=tuple([None]),
                                        )
    data['location'] = h5file.create_dataset(name='location',
                                             shape=tuple([write_per_num, 3]),
                                             compression='gzip',
                                             dtype=np.uint8,
                                             maxshape=tuple([None, 3]))

    # process one patient by one patient
    index = 1
    patch_num = 0
    patch_writed = 0
    image_patch_container = list()
    mask_patch_container = list()
    location_container = list()
    pid_container = list()

    for image_path, mask_path in zip(image_items, mask_items):
        logging.info('-' * 100)
        logging.info('start process %d patient' % index)

        # step 1: load image
        image = load_image(image_path, sitk.sitkFloat32)
        mask = load_image(mask_path, sitk.sitkUInt8)

        # step 2: truncate image with fixed window
        image = truncation(image, lower=window_lower, upper=window_upper)

        # step 3: resize z-axis
        # set z-axis spacing to 1.0
        original_spacing = image.GetSpacing()

        if process_resize_spacing is 'standard':
            resize_spacing = [1.0, 1.0, 1.0]
        elif process_resize_spacing == 'origin':
            resize_spacing = original_spacing
        elif process_resize_spacing == 'z-axis':
            resize_spacing = [1.0] + []
        else:
            raise NotImplementedError('Not support for this resize spacing, please check again.')

        image = resize(image, resize_spacing, original_spacing, sitk.sitkLinear, dtype=sitk.sitkFloat32)
        mask = resize(mask, resize_spacing, original_spacing, sitk.sitkUInt8, dtype=sitk.sitkUInt8)

        image = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(mask)

        # step 4: sample patch
        image_patchs, mask_patchs, locations = sample(image, mask, size=sample_size, overlap=sample_overlap)
        logging.info('sample from {} with {} patchs'.format(image_path, str(len(image_patchs))))

        # step 5: filter patch
        image_patchs, mask_patchs,locations = patch_filter(image_patchs, mask_patchs, background=0, rate=process_filter_rate,locations=locations)
        logging.info('filter form {} with {} patchs'.format(image_path, str(len(image_patchs))))

        length = len(image_patchs)
        if length == 0:
            index += 1
            continue

        image_patch_container.extend(image_patchs)
        mask_patch_container.extend(mask_patchs)
        location_container.extend(locations)
        pid_container.extend(len(image_patchs)*[image_path])

        if len(image_patch_container) >= write_per_num:
            for i in range(len(image_patch_container) // write_per_num):
                image_patch_to_save = np.stack(image_patch_container[:write_per_num])
                mask_patch_to_save = np.stack(mask_patch_container[:write_per_num])
                location_to_save = np.stack(mask_patch_container[:write_per_num])
                pid_to_save = np.stack(pid_container[:write_per_num])

                data['image'][patch_writed:patch_writed + write_per_num, :] = image_patch_to_save
                data['mask'][patch_writed:patch_writed + write_per_num, :] = mask_patch_to_save
                data['location'][patch_writed:patch_writed + write_per_num, :] = location_to_save
                data['pid'][patch_writed:patch_writed + write_per_num, :] = pid_to_save

                patch_writed += write_per_num

                image_patch_container = image_patch_container[write_per_num:]
                mask_patch_container = mask_patch_container[write_per_num:]
                location_container = location_container[write_per_num:]
                pid_container = pid_container[write_per_num:]

        patch_num += length
        index += 1

    if len(image_patch_container) != 0:
        length = len(image_patch_container)
        image_patch_to_save = np.stack(image_patch_container)
        mask_patch_to_save = np.stack(mask_patch_container)
        location_to_save = np.stack(location_container)
        pid_to_save = np.stack(pid_container)

        data['image'].resize(size=patch_writed + length, axis=0)
        data['mask'].resize(size=patch_writed + length, axis=0)
        data['pid'].resize(size=patch_writed + length, axis=0)
        data['location'].resize(size=patch_writed + length, axis=0)

        data['image'][patch_writed:patch_writed + length, :] = image_patch_to_save
        data['mask'][patch_writed:patch_writed + length, :] = mask_patch_to_save
        data['pid'][patch_writed:patch_writed + length] = pid_to_save
        data['location'][patch_writed:patch_writed + length, :] = location_to_save
        patch_writed += length

    logging.info('-' * 100)
    logging.info('Preprocess complete, total processed %d patchs' % (patch_num))
    h5file.close()


if __name__ == '__main__':
    original_path = r'/Users/tanjiale/PycharmProjects/KiTS19-Beta/data/original'
    target_path = r'/Users/tanjiale/PycharmProjects/KiTS19-Beta/data/processed'

    train_folder = r'train'
    val_folder = r'val'

    image_name = r'imaging.nii.gz'
    mask_name = r'segmentation.nii.gz'
    ratio = 0.85

    patients = file_name_path(original_path, 'dir')
    image_items = [os.path.join(original_path, patient, image_name) for patient in patients]
    mask_items = [os.path.join(original_path, patient, mask_name) for patient in patients]

    train_image_items, train_mask_items, \
    val_image_items, val_mask_items = train_val_split(image_items, mask_items, ratio=ratio)

    if not os.path.exists(os.path.join(target_path, train_folder)):
        os.makedirs(os.path.join(target_path, train_folder))
    if not os.path.exists(os.path.join(target_path, val_folder)):
        os.makedirs(os.path.join(target_path, val_folder))

    write_data_hdf5(train_image_items,
                    train_mask_items,
                    os.path.join(target_path, train_folder, 'train.hdf5'),
                    process_filter_rate=0.05,
                    write_per_num=500,
                    process_resize_spacing='origin',
                    sample_size=(128, 128, 32),
                    sample_overlap=(64, 64, 16),
                    window_lower=-200,
                    window_upper=300)

    write_data_hdf5(val_image_items,
                    val_mask_items,
                    os.path.join(target_path, val_folder, 'val.hdf5'),
                    process_filter_rate=0.05,
                    write_per_num=500,
                    process_resize_spacing='origin',
                    sample_size=(128, 128, 32),
                    sample_overlap=(64, 64, 16),
                    window_lower=200,
                    window_upper=300)
