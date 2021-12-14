import os
import re
import cv2
from numpy import core
import torch
import random
import argparse

import numpy as np
import Polygon as plg

from PIL import Image
from collections import OrderedDict
from matplotlib import patches
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

def read_txt(txt_path):
    txt_contents = []
    f = open(txt_path, 'r')
    while True:
        line = f.readline()
        if not line: break
        x1, y1, x2, y2, x3, y3, x4, y4, label = line.split(',')
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, [x1, y1, x2, y2, x3, y3, x4, y4])
        label = label.split()[0]
        polygon = polygon_from_points([x1, y1, x2, y2, x3, y3, x4, y4])
        txt_contents.append([polygon, label, f"{x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}"])
        
    f.close()
    return txt_contents


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except:
        return 0


def make_recognition_pred_gt(detection_output_dir, detection_data_dir, iou_threshold, cropped_data_dir):
    count_dict = dict()
    if not os.path.isdir(cropped_data_dir):
        os.makedirs(cropped_data_dir, exist_ok = True)
    for detection_result in os.listdir(detection_output_dir):
        if 'txt' in detection_result:
            detection_path = os.path.join(detection_output_dir, detection_result)
            gt_path = os.path.join(detection_data_dir, detection_result)
            eval_data_path = os.path.join(cropped_data_dir, detection_result)
            gt_polygon_label_coordinate = read_txt(gt_path)
            pred_polygon_confidence_coordinate = read_txt(detection_path)
            
            img_label_count = 0
            img_det_count = 0
            img_name = detection_result.split('.txt')[0]

            for gt_polygon, gt_label, gt_coordinate in gt_polygon_label_coordinate:
                if gt_label not in ['*', '###', 'syeom']:
                    img_label_count += 1
            
            f = open(eval_data_path, 'w')
            for pred_polygon, pred_confidence, pred_coordinate in pred_polygon_confidence_coordinate:
                keep_info = []
                for gt_polygon, gt_label, gt_coordinate in gt_polygon_label_coordinate:
                    if gt_label in ['*', '###', 'syeom']:
                        pass
                    else:
                        intersection_over_union = get_intersection_over_union(gt_polygon, pred_polygon)
                        if intersection_over_union > iou_threshold:
                            keep_info.append([pred_coordinate, gt_label, intersection_over_union])
                if keep_info:
                    img_det_count += 1
                    keep_info = sorted(keep_info, key = lambda x : -x[2])
                    data = f"{keep_info[0][0]}, {keep_info[0][1]}\n"
                    f.write(data)
            count_dict[img_name] =[img_det_count, img_label_count]
            f.close()
    return count_dict


def coordinate_process(points_confidence):
    x1, y1, x2, y2, x3, y3, x4, y4, confidence = points_confidence
    x_points = [int(x1), int(x2), int(x3), int(x4)]
    y_points = [int(y1), int(y2), int(y3), int(y4)]
    min_x = int(min(x_points))
    max_x = int(max(x_points))
    min_y = int(min(y_points))
    max_y = int(max(y_points))
    return min_x, max_x, min_y, max_y


def coordinate_process_pred(points_confidence):
    x1, y1, x2, y2, x3, y3, x4, y4, label = points_confidence
    x_points = [int(x1), int(x2), int(x3), int(x4)]
    y_points = [int(y1), int(y2), int(y3), int(y4)]
    min_x = int(min(x_points))
    max_x = int(max(x_points))
    min_y = int(min(y_points))
    max_y = int(max(y_points))
    label = label.split( )
    return min_x, max_x, min_y, max_y, label
    

def polygon_crop(src_img, points_confidence, args):
    is_vertical = False
    x1, y1, x2, y2, x3, y3, x4, y4, label = points_confidence
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, [x1, y1, x2, y2, x3, y3, x4, y4])
    min_x, max_x, min_y, max_y = coordinate_process(points_confidence)
    if (max_y - min_y) / (max_x - min_x) >= 1.5:
        is_vertical = True
    temp_points = np.float32([ [x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    sm = temp_points.sum(axis = 1)
    dif = np.diff(temp_points, axis  = 1)
    tl = temp_points[np.argmin(sm)]
    br = temp_points[np.argmax(sm)]
    tr = temp_points[np.argmin(dif)]
    bl = temp_points[np.argmax(dif)]

    if is_vertical:
        H, W = args.imgW, args.imgH
    else:
        H, W = args.imgH, args.imgW
    
    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[0,0], [W, 0], [0, H], [W, H]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(src_img, M, (W, H))
    if is_vertical:
        res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return res, [x1, y1, x2, y2, x3, y3, x4, y4], label.split( )[0], is_vertical


def crop_polygon_with_persp(detection_data_dir, detection_output_dir, args):
    detection_results = []
    for detection_result in os.listdir(detection_output_dir):
        if 'txt' in detection_result:
            detection_results.append([f'{detection_output_dir}{detection_result}', f"{detection_data_dir}{detection_result.split('.txt')[0]}.jpg"])
    crop_imgs = dict()
    for txt_path, img_path in detection_results:
        if not os.path.isfile(img_path):        
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src_img = img
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        crop_imgs[img_name] = []
        crop_imgs_h = []
        crop_imgs_v = []
        with open(txt_path, 'r') as f:
            count = 1
            while True:
                line = f.readline().strip('\n').strip(' ')
                if not line:
                    break
                res, coordinate, label, is_vertical = polygon_crop(src_img, line.split(','), args)
                if is_vertical:
                    crop_imgs_v.append([res, coordinate, label])
                else:
                    crop_imgs_h.append([res, coordinate, label])
                
            crop_imgs[img_name].append([crop_imgs_h, crop_imgs_v])
    return crop_imgs


def crop_polygon_with_persp_from_gt(gt_data_dir, args):
    gt_file_names = []
    for gt_file_name in os.listdir(gt_data_dir):
        if 'txt' in gt_file_name:
            gt_file_names.append([f'{gt_data_dir}{gt_file_name}', f"{gt_data_dir}{gt_file_name.split('.txt')[0]}.jpg"])
    crop_imgs = dict()
    for txt_path, img_path in gt_file_names:
        if not os.path.isfile(img_path):        
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src_img = img
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        crop_imgs[img_name] = []

        with open(txt_path, 'r') as f:
            while True:
                line = f.readline().strip('\n').strip(' ')
                if not line:
                    break
                res, coordinate, label, is_vertical = polygon_crop(src_img, line.split(','), args)
                if label in ['*', '###', 'syeom']:
                    continue
                crop_imgs[img_name].append([res, coordinate, label])
    return crop_imgs


def crop_img(detection_data_dir, detection_output_dir, hv_ratio):
    detection_results = []
    for detection_result in os.listdir(detection_output_dir):
        if 'txt' in detection_result:
            detection_results.append([f'{detection_output_dir}{detection_result}', f"{detection_data_dir}{detection_result.split('.txt')[0]}.jpg"])
    crop_imgs = dict()
    for txt_path, img_path in detection_results:
        if not os.path.isfile(img_path):        
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_name = img_path.split('/')[-1].split('.')[0]
        crop_imgs[img_name] = []
        
        f = open(txt_path, 'r')
        
        while True:
            line = f.readline()
            if not line: break
            min_x, max_x, min_y, max_y= coordinate_process(line.split(','))
            crop = img[min_y:max_y, min_x:max_x]
            crop_imgs[img_name].append([crop, [min_x, min_y, max_x, max_y]])
        f.close()
    return crop_imgs


def crop_img_pred(detection_data_dir, recognition_eval_data_dir, hv_ratio):
    detection_results = []
    for detection_result in os.listdir(recognition_eval_data_dir):
        if 'txt' in detection_result:
            detection_results.append([f'{recognition_eval_data_dir}{detection_result}', f"{detection_data_dir}{detection_result.split('.txt')[0]}.jpg"])
    crop_imgs = dict()
    for txt_path, img_path in detection_results:
        if not os.path.isfile(img_path):        
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_name = img_path.split('/')[-1].split('.')[0]
        crop_imgs[img_name] = []
        
        f = open(txt_path, 'r')
        while True:
            line = f.readline()
            if not line: break
            min_x, max_x, min_y, max_y, label = coordinate_process_pred(line.split(','))
            crop = img[min_y:max_y, min_x:max_x]
            crop_imgs[img_name].append([crop, [min_x, min_y, max_x, max_y], label])
        f.close()
    return crop_imgs


def change_checkpoint(detection_checkpoint):
    config = torch.load('./detection/output/DBNet_MobileNetV3_FPN_DBHead/checkpoint/model_best.pth')
    torch.save(config['state_dict'], detection_checkpoint)


def visualization_poly(ocr_output_dir, detection_data_dir):
    fm.get_fontconfig_fonts()
    font_path = "./NanumFont/NanumGothicBold.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    for txt_name in os.listdir(ocr_output_dir):
        if 'txt' in txt_name:
            img_name = f"{txt_name.split('.txt')[0]}.jpg"
            img_path = os.path.join(detection_data_dir, img_name)
            txt_path = os.path.join(ocr_output_dir, txt_name)

            img = Image.open(img_path)
            plt.imshow(img)
            
            ax = plt.gca()
            with open(txt_path, 'r') as f:
                while True:
                    line = f.readline().strip('\n').strip(' ')
                    if not line:
                        break
                    else:
                        p4 = re.compile(r"([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, (.*)")
                        p8 = re.compile(r"([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, (.*)")
                        
                        if m:= p8.search(line):
                            det = [int(m.group(i)) for i in range(1, 9)]
                            rec = m.group(9)
                        
                        elif m := p4.search(line):
                            det = [int(m.group(i)) for i in range(1, 4)]
                            rec = m.group(4)
                            
                        x = [det[i] for i in range(len(det)) if i % 2 == 0]
                        y = [det[i] for i in range(len(det)) if i % 2 == 1]
                        xy = np.array([ [x_i, y_i] for x_i, y_i in zip(x, y)])

                        poly = patches.Polygon(xy = xy,
                                                fill = False,
                                                linewidth = 2,
                                                edgecolor = 'cyan')
                        ax.add_patch(poly)
                        plt.text(min(x), min(y), rec, fontproperties=font_prop)
            plt.axis('off')
            plt.savefig(os.path.join(ocr_output_dir, img_name), bbox_inches = 'tight', pad_inches = 0)
            plt.clf()
                    

def to_tensor(src):
    src = src.transpose((2, 0 ,1))
    src = src/255.
    src = src[np.newaxis, :, :, :]
    src_tensor = torch.Tensor(src)
    src_tensor.sub_(0.5).div_(0.5)
    return src_tensor


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict



def gt_list_num_except_ignore(gt_list, ignore):
    gt_list_num = 0
    for gt in gt_list:
        _, _, _, _, _, _, _, _, gt_label = gt
        if gt_label not in ignore:
            gt_list_num += 1

    return gt_list_num

def evaluation(pred_list, gt_list, iou_threshold):
    ignore = ['*', '###']
    gt_num = gt_list_num_except_ignore(gt_list, ignore)
    if gt_num == 0:
        return 0, 0
    correct_num = 0
    for pred in pred_list:
        x1, y1, x2, y2, x3, y3, x4, y4, pred_label = pred 
        pred_polygon = polygon_from_points([x1, y1, x2, y2, x3, y3, x4, y4])
        
        keep_info = []
        for gt in gt_list:
            x1, y1, x2, y2, x3, y3, x4, y4, gt_label = gt
            if gt_label not in ignore:
                gt_polygon = polygon_from_points([x1, y1, x2, y2, x3, y3, x4, y4])
                intersection_over_union = get_intersection_over_union(pred_polygon, gt_polygon)
                if intersection_over_union > iou_threshold:
                    keep_info.append([gt_label, intersection_over_union])
                
        if keep_info:
            keep_info = sorted(keep_info, key = lambda x : -x[1])[0]
            if pred_label == gt_label:
                correct_num += 1
    return correct_num, gt_num




def visualization(img_path, predict, output_dir):
    fm.get_fontconfig_fonts()
    font_path = "./NanumFont/NanumGothicBold.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    img_name = img_path.split('/')[-1]
    img = Image.open(img_path)
    plt.imshow(img)
    ax = plt.gca()
    for label, coordinate in predict.items():
        x = [coordinate[i] for i in range(len(coordinate)) if i % 2 == 0]
        y = [coordinate[i] for i in range(len(coordinate)) if i % 2 == 1]
        xy = np.array([ [x_i, y_i] for x_i, y_i in zip(x, y)])
        poly = patches.Polygon(xy = xy,
                                fill = False,
                                linewidth = 2,
                                edgecolor = 'cyan')
        ax.add_patch(poly)
        plt.text(min(x), min(y), label, fontproperties=font_prop)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, img_name), bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
            