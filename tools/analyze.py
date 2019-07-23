import sys
#sys.path.append(r'E:/ISICDN2019')
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from ISICDM2019.tools.utils import file_name_path, truncation
import json
# 将数据信息写入一个txt文本中
def write_information( path,label_path, name,data):
    _, size, spacing, origin, direction = read_nii(path)
    print(path)
    information={'name':name,'image':path,'label':label_path,'size':size,'spacing':spacing,'origin':origin,'direction':direction,'describe':"WHD"}
    data.append(information)

#get image information
def read_nii(path):
    image = sitk.ReadImage(path)
    return image, image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()

# pancreas information write into json file
def write_pancreas_train(path, save_path):
    dir = os.listdir(path)
    data=[]
    with open(save_path,'w') as file:
        for i in dir:
            if(len(i.split('_'))==1):
                write_information(path + '/' + i,path + '/' + i.split('.')[0] + '_seg.nii', i,data)
        json.dump(data, file, indent=4)
        file.close()

# liver information write into json file
def write_liver_train(path, save_path):
    dir = os.listdir(path)
    data = []
    with open(save_path, 'w') as file:
        for i in dir:
            write_information(path + '/' + i+ '/liver.nii',path + '/' + i+'/liver_nid.nii;'+path + '/' + i+'/liver_seg.nii', i, data)
        json.dump(data, file, indent=4)
        file.close()

# draw a hist
def draw_hist(save_path, draw, name):
    plt.hist(draw, bins=40, facecolor='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Z number')
    plt.ylabel('frequency')
    plt.title(name)
    plt.grid(True, linestyle=':', color='r', alpha=0.6)
    plt.savefig(save_path)
    plt.show()

# draw liver Z axis hist
def draw_liver(path, save_path):
    draw = []
    for i in os.listdir(path):
        image, size, _, _, _ = read_nii(path + '/' + i + '/liver.nii')
        draw.append(size[2])
    draw_hist(save_path, draw, 'liver')

# draw pancreas Z axis hist
def draw_pancreas(path, save_path):
    dir = os.listdir(path)
    draw = []
    for i in dir:
        temp = i.split('_')
        if (len(temp) == 1):
            print(i)
            image, size, _, _, _ = read_nii(path + '/' + i)
            draw.append(size[2])
    draw_hist(save_path, draw, 'thick_pancreas')


# 数据直方图
def draw_liver_data(path, save_path):
    data = []
    for i in file_name_path(path, 'dir'):
        image, _, _, _, _ = read_nii(path + '/' + i + '/liver.nii')
        array = sitk.GetArrayFromImage(image).flatten()
        # array=array[array<255]
        # array=array[array>-255]
        data.append(array)
    draw_hist(save_path, data, 'liver_seg')


def draw_pancreas_data(path, save_path):
    data = []
    for i in file_name_path(path, 'file'):
        temp = i.split('_')
        if (len(temp) == 1):
            image, size, _, _, _ = read_nii(path + '/' + i)
            array = sitk.GetArrayFromImage(image).flatten()
            array = array[array < 255]
            array = array[array > -255]
            data.append(array)
    draw_hist(save_path, data, 'filter_pancreas_data')


# 计算数据总体均值mean value 方差variance

def mean_value(data):
    mean = data.mean()
    return mean


def variance(data):
    return data.std()


def pancreas_data_mean_value(path):
    data_mean = 0
    count = 0
    data_variance = 0
    for i in file_name_path(path, 'file'):
        temp = i.split('_')
        if (len(temp) == 1):
            image, size, _, _, _ = read_nii(path + '/' + i)
            image = truncation(image, -255, 255)
            data = sitk.GetArrayFromImage(image)
            data_mean += mean_value(data)
            data_variance += variance(data)
            count += 1
    print(' pancreas_data 均值和方差： ',data_mean / count, data_variance / count)
    return data_mean / count, data_variance / count


def liver_data_mean_value(path):
    data_mean = 0
    count = 0
    data_variance = 0
    for i in file_name_path(path, 'dir'):
        image, size, _, _, _ = read_nii(path + '/' + i + '/liver.nii')
        image = truncation(image, -255, 255)
        data = sitk.GetArrayFromImage(image)
        data_mean += mean_value(data)
        data_variance += variance(data)
        count += 1
    print('liver_data 均值和方差：',data_mean / count, data_variance / count)
    return data_mean / count, data_variance / count


# 目标区域直方图，均值，方差

def draw_liver_label_data(path, save_path):
    data = []
    data_2 = []
    for i in file_name_path(path, 'dir'):
        array_label, _, _ = get_image_label_comment(path + '/' + i + '/liver.nii',
                                                    path + '/' + i + '/liver_seg.nii')
        array_label_2, _, _ = get_image_label_comment(path + '/' + i + '/liver.nii',
                                                      path + '/' + i + '/liver_nid.nii')
        print(array_label.sum(), array_label_2.sum())
        data.append(array_label)
        data_2.append(array_label_2)
    draw_hist(save_path + '/liver_seg_data.jpg', data, 'liver_seg')
    draw_hist(save_path + '/liver_nid_data.jpg', data_2, 'liver_nid_tumour')


def draw_pancreas_label_data(path, save_path):
    draw = []
    for i in file_name_path(path, 'file'):
        temp = i.split('_')
        if (len(temp) == 1):
            print(i.split('.')[0] + '_seg.nii')
            array_label, _, _ = get_image_label_comment(path + '/' + i,
                                                        path + '/' + i.split('.')[0] + '_seg.nii')
            draw.append(array_label)
    draw_hist(save_path + '/thick_pancreas_label_data.jpg', draw, 'thick_pancreas_label')


def get_image_label_comment(image_path, label_path):
    image, _, _, _, _ = read_nii(image_path)
    image = truncation(image, -255, 255)
    label, _, _, _, _ = read_nii(label_path)
    image = sitk.GetArrayFromImage(image).flatten()
    label = sitk.GetArrayFromImage(label).flatten().astype(np.int).astype(bool)
    img = image[label]
    mean = mean_value(img)
    varian = variance(img)
    return img, mean, varian


def liver_label_data_mean_value(path):
    data_mean_seg = 0
    count = 0
    data_variance_seg = 0
    data_mean_nid = 0
    data_variance_nid = 0
    for i in file_name_path(path, 'dir'):
        _, mean_seg, variance_seg = get_image_label_comment(path + '/' + i + '/liver.nii',
                                                            path + '/' + i + '/liver_seg.nii')
        _, mean_nid, variance_nid = get_image_label_comment(path + '/' + i + '/liver.nii',
                                                            path + '/' + i + '/liver_nid.nii')
        data_mean_seg += mean_seg
        data_variance_seg += variance_seg
        data_mean_nid += mean_nid
        data_variance_nid += variance_nid
        count += 1
    print('liver 目标区域均值和方差：',data_mean_seg / count, data_variance_seg / count, data_mean_nid / count, data_variance_nid / count)
    return data_mean_seg / count, data_variance_seg / count, data_mean_nid / count, data_variance_nid / count


def pancreas_label_data_mean_value(path):
    data_mean = 0
    data_variance = 0
    count = 0
    for i in file_name_path(path, 'file'):
        temp = i.split('_')
        if (len(temp) == 1):
            _, mean_label, variance_label = get_image_label_comment(path + '/' + i,
                                                                    path + '/' + i.split('.')[0] + '_seg.nii')
            data_mean += mean_label
            data_variance += variance_label
            count += 1
    print('pancreas 目标区域均值和方差： ',data_mean / count, data_variance / count)
    return data_mean / count, data_variance / count


if __name__ == '__main__':
    #write_panceas_train(r'E:/ISICDN2019/data/pancreas/pancreas_train/thick',
     #                   r'E:/ISICDN2019/data/pancreas/pancreas_train/pancreas.json')
    #write_liver_train(r'E:/ISICDN2019/data/liver/liver_train',r'E:/ISICDN2019/data/liver/liver.json')
    # draw_liver(r'E:/ISICDN2019/data/liver/liver_train','E:/ISICDN2019/data/liver/liver.jpg')
    # draw_pancreas(r'E:/ISICDN2019/data/pancreas/pancreas_train/thick',
    #          r'E:/ISICDN2019/data/pancreas/pancreas_train/thick_pancreas.jpg')
    # draw_liver_data(r'E:/ISICDN2019/data/liver/liver_train', 'E:/ISICDN2019/data/liver/liver_seg_data.jpg')
    # draw_pancreas_data(r'E:/ISICDN2019/data/pancreas/pancreas_train/thick',
    #               r'E:/ISICDN2019/data/pancreas/pancreas_train/filter_thick_pancreas_data.jpg')
    #pancreas_mean=pancreas_data_mean_value(r'E:\ISICDN2019\data\pancreas\pancreas_train\thick')
    # liver_mean = liver_data_mean_value(r'E:/ISICDN2019/data/liver/liver_train')
    # draw_liver_label_data(r'E:/ISICDN2019/data/liver/liver_train', 'E:/ISICDN2019/data/liver')
    # liver_label_data_mean_value(r'E:/ISICDN2019/data/liver/liver_train')
    # draw_pancreas_label_data(r'E:/ISICDN2019/data/pancreas/pancreas_train/thick',
    #                          r'E:/ISICDN2019/data/pancreas/pancreas_train')
    pancreas_label_data_mean_value(r'E:/ISICDN2019/data/pancreas/pancreas_train/thick')