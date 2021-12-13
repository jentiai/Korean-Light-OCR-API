import os
import json
import torch
import random
import pathlib
import argparse

import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tools import *
from detection.tools.detector import Detector
from detection.utils.util import show_img, draw_bbox, save_result, get_file_list

from recognition.model import Model as Recognizer
from recognition.utils import CTCLabelConverter, AttnLabelConverter

os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'   
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = 1111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    dash_line = '-'*84
    print('\n' + '-'*28 + f' JENTI KOREAN OCR EVALUATION ' + '-'*27 + '\n')
    ocr_checkpoint_h = args.ocr_checkpoint[:-4] + f"_{args.Prediction}_{args.imgH}_{args.imgW}_st_{args.stride}_h.pth"
    ocr_checkpoint_v = args.ocr_checkpoint[:-4] + f"_{args.Prediction}_{args.imgH}_{args.imgW}_st_{args.stride}_v.pth"

    if not os.path.isdir(args.detection_output_dir):
        os.makedirs(args.detection_output_dir, exist_ok = True)
    if not os.path.isdir(args.ocr_output_dir):
        os.makedirs(args.ocr_output_dir, exist_ok = True)
    if not os.path.isfile(args.detection_checkpoint):
        change_checkpoint(args.detection_checkpoint)
    if not os.path.isdir(args.cropped_data_dir):
        os.makedirs(args.cropped_data_dir, exist_ok = True)
    if not os.path.isdir(args.evaluation_data_dir):
        os.makedirs(args.evaluation_data_dir, exist_ok = True)

    print(f'[Ready] Device')
    print(f'        - {"Type":<12s}: {device}')
    if torch.cuda.is_available():
        print(f'        - {"# of GPUs":<12s}: {torch.cuda.device_count()}')
        print(f'        - {"Name":<12s}: {torch.cuda.get_device_name()}')
    print(f'[Ready] Output directories')
    print(f'        - {"Detection":<12s}: {args.detection_output_dir}')
    print(f'        - {"Recognition":<12s}: {args.ocr_output_dir}')
    img_len = len(get_file_list(args.detection_data_dir, p_postfix=['.jpg']))
    print(f'[Ready] Dataset')
    print(f'        - {"Directory":12s}: {args.detection_data_dir}')
    print(f'        - {"# of images":12s}: {img_len}')
    
    detector = Detector(model_path = args.detection_checkpoint,
                        detection_config_path = args.detection_config_file,
                        device = device, 
                        post_p_thre = args.detection_threshold)
    print(f'[Ready] Detector')
    print(f'        - {"config":<12s}: {os.path.basename(args.detection_config_file)}')
    print(f'        - {"checkpoint":<12s}: {os.path.basename(args.detection_checkpoint)}')

    with open(args.charset_path, 'r') as t:
        charset = t.read().strip('\n').strip(' ')
        args.character = charset
        if 'CTC' in args.Prediction:
            converter = CTCLabelConverter(args.character, device = device)
            cc = 'CTC'
        else:
            converter = AttnLabelConverter(args.character, device = device)
            cc = 'Attn'
        args.num_class = len(converter.character)

    recognizer_h = Recognizer(args).to(device)
    recognizer_h.load_state_dict(copyStateDict(torch.load(ocr_checkpoint_h)))
    recognizer_h.eval()
    
    if os.path.isfile(ocr_checkpoint_v):
        recognizer_v = Recognizer(args).to(device)
        recognizer_v.load_state_dict(copyStateDict(torch.load(ocr_checkpoint_v)))
        recognizer_v.eval()
    else:
        recognizer_v = recognizer_h
        ocr_checkpoint_v = ocr_checkpoint_h

    print(f'[Ready] Recognizer')
    print(f'        - {"config":<12s}: NoTPS-MobileNetV3({args.output_channel})-BiLSTM({args.hidden_size})-{cc}')
    print(f'        - {"checkpoint":<12s}: [{"HORIZONTAL":^12s}] {os.path.basename(ocr_checkpoint_h)}')
    print(f'                        [{"VERTICAL":^12s}] {os.path.basename(ocr_checkpoint_v)}')
    print(f'        - {"# of chars":<12s}: {args.num_class}')
    print('\n'+dash_line+'\n')

    print(f'[Evaluation] Now evaluate detector...')
    for img_path in tqdm(get_file_list(args.detection_data_dir, p_postfix=['.jpg']), bar_format= '{percentage:3.0f}%|{bar:49}{r_bar}'):
        _, boxes_list, score_list, t = detector.predict(img_path)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        os.makedirs(args.detection_output_dir, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.detection_output_dir, img_path.stem + '_result.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, False)
    
    count_dict = make_recognition_pred_gt(args.detection_output_dir, args.detection_data_dir, args.iou_threshold, args.cropped_data_dir)
    det_pred = crop_polygon_with_persp(args.detection_data_dir, args.cropped_data_dir, args)
    print(f'[Evaluation] Detection is done:')
    print(f'       - # of detected texts: {len(det_pred.values())}\n')

    print('\n' + ' '*26 + "** End-to-End Style Evaluation **" + ' '*26)
    head = f'{" #":^8s} | {"Image Name":^37s} | {"Det.":^8s} | {"Rec.":^8s} | {"E2E":^8s}'
    head = f'{dash_line}\n{head}\n{dash_line}'
    print(head)
    i = 1
    e2e_total = 0
    with torch.no_grad():
        for k, v in det_pred.items():
            img_name = str(k)
            if os.path.isfile(os.path.join(args.ocr_output_dir, str(k) + '.txt')):
                os.remove(os.path.join(args.ocr_output_dir, str(k) + '.txt'))
            with open(os.path.join(args.evaluation_data_dir, img_name + '.json'), 'w', encoding = 'UTF-8-sig') as json_res:
                dict_res = {}
                with open(os.path.join(args.evaluation_data_dir, img_name + '.txt'), 'wb') as txt_res:
                    count_img = 0
                    res_h, res_v = v[0]
                    if len(res_h) >= 1 and res_h[0] != []:
                        for crop, bbox, label in res_h:
                            label = label.lower()
                            crop_tensor = to_tensor(crop).to(device)
                            length_for_pred = torch.IntTensor([args.batch_max_length] * crop_tensor.size(0)).to(device)
                            text_for_pred = torch.LongTensor(crop_tensor.size(0), args.batch_max_length + 1).fill_(0).to(device)
                            
                            if 'CTC' in args.Prediction:
                                rec_pred = recognizer_h(crop_tensor, text = text_for_pred, is_train = False)
                                pred_size = torch.IntTensor([rec_pred.size(1)] * 1)
                                _, preds_index = rec_pred.max(2)
                                pred_str = converter.decode(preds_index.data, pred_size.data)[0]

                            elif 'Attn' in args.Prediction:
                                rec_pred = recognizer_h(crop_tensor, text = text_for_pred, is_train = False)
                                _, preds_index = rec_pred.max(2)
                                pred_str = converter.decode(preds_index, length_for_pred)[0]
                                pred_EOS = pred_str.find('[s]')
                                pred_str = pred_str[:pred_EOS]

                            ocr_res = f'{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}, {bbox[5]}, {bbox[6]}, {bbox[7]}, {pred_str}\n'.encode('utf-8')
                            txt_res.write(ocr_res)
                            dict_res[pred_str] = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]]
                            if pred_str == label:
                                count_img += 1
                    
                    if len(res_v) >= 1 and res_v[0] != []:
                        for crop, bbox, label in res_v:
                            label = label.lower()
                            crop_tensor = to_tensor(crop).to(device)
                            length_for_pred = torch.IntTensor([args.batch_max_length] * crop_tensor.size(0)).to(device)
                            text_for_pred = torch.LongTensor(crop_tensor.size(0), args.batch_max_length + 1).fill_(0).to(device)
                            
                            if 'CTC' in args.Prediction:
                                rec_pred = recognizer_h(crop_tensor, text = text_for_pred, is_train = False)
                                pred_size = torch.IntTensor([rec_pred.size(1)] * 1)
                                _, preds_index = rec_pred.max(2)
                                pred_str = converter.decode(preds_index.data, pred_size.data)[0]
                                
                                pred_soft = F.Softmax(rec_pred, dim = 2)
                                pred_max_prob, _ = pred_soft.max(dim = 2)
                                pred_max_prob = pred_max_prob[0]
                                
                                try:
                                    h_conf = pred_max_prob.cumprod(dim = 0)[-1]
                                    h_conf = h_conf.item()
                                except:
                                    h_conf = 0
                                
                                if h_conf <= 0.1:
                                    rec_pred_v = recognizer_v(crop_tensor, text = text_for_pred, is_train = False)
                                    _, preds_index = rec_pred.max(2)
                                    pred_str = converter.decode(preds_index.data, pred_size.data)[0]
                                    
                            elif 'Attn' in args.Prediction:
                                rec_pred = recognizer_h(crop_tensor, text = text_for_pred, is_train = False)
                                _, preds_index = rec_pred.max(2)
                                pred_str = converter.decode(preds_index, length_for_pred)[0]
                                pred_EOS = pred_str.find('[s]')
                                pred_str = pred_str[:pred_EOS]
                                
                                pred_soft = F.Softmax(rec_pred, dim = 2)
                                pred_max_prob, _ = pred_soft.max(dim = 2)
                                pred_max_prob = pred_max_prob[0]
                                pred_max_prob = pred_max_prob[:pred_EOS]

                                try:
                                    h_conf = pred_max_prob.cumprod(dim = 0)[-1]
                                    h_conf = h_conf.item()
                                except:
                                    h_conf = 0

                                if h_conf <= 0.1:
                                    rec_pred_v = recognizer_v(crop_tensor, text = text_for_pred, is_train = False)
                                    _, preds_index = rec_pred.max(2)
                                    pred_str = converter.decode(preds_index.data, pred_size.data)[0]
                                    pred_EOS = pred_str.find('[s]')
                                    pred_str = pred_str[:pred_EOS]

                            ocr_res = f'{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}, {bbox[5]}, {bbox[6]}, {bbox[7]}, {pred_str}\n'.encode('utf-8')
                            txt_res.write(ocr_res)
                            dict_res[pred_str] = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]]
                            if pred_str == label:
                                count_img += 1

                json.dump(dict_res, json_res, ensure_ascii = False, indent = '\t')
            d_correct, box_count = count_dict[img_name]
            det_score, rec_score, e2e_score = 0, 0, 0

            if box_count != 0: # gt box가 존재할 때
                if d_correct != 0: # detection을 하나라도 했을 때
                    det_score = d_correct/box_count
                    rec_score = count_img/d_correct
                    e2e_score = count_img/box_count
                else: # detection을 하나도 못했을 때
                    det_score, rec_score, e2e_score = 0, 0, 0
            else: # gt box가 없을 때
                continue
            
            print(f" {i:^8d}| {img_name:^37s} | {det_score:^8.4f} | {rec_score:^8.4f} | {e2e_score:^8.4f}")
            i += 1
            e2e_total += e2e_score
        print(dash_line)
        print(f'{"Evaluation_results":80s}')
        print(f"- # of Images: {i-1}\n- [Macro] E2E Accuracy: {e2e_total/(i-1):3.3f}\n- [Micro] E2E Accuracy: {e2e_total/(i-1):3.3f}\n")
        ## ToDos: Micro level accuracy 수정 필요


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_data_dir', type=str, default = './sample/', help='detection data path')
    parser.add_argument('--detection_output_dir', type=str, default = './detection/output/predict/', help='detection output path')
    parser.add_argument('--detection_checkpoint', type=str, default = './checkpoint/detection_best.pth', help='detection_checkpoint')
    parser.add_argument('--detection_config_file', type=str, default='./detection/config/icdar2015_dcn_mobilenetv3_FPN_DBhead_polyLR.yaml')
    parser.add_argument('--detection_threshold', type=str, default=0.5 , help='detection_threshold')
    parser.add_argument('--cropped_data_dir', type=str, default='./cropped_data_dir/' , help='eval_data_dir')
    parser.add_argument('--evaluation_data_dir', type = str, default = './evaluation_data_dir', help = 'final result directory')
    
    parser.add_argument('--ocr_output_dir', type = str, default = './ocr_output')
    parser.add_argument('--ocr_checkpoint', type = str, default = './checkpoint/best_accuracy.pth')
    parser.add_argument('--charset_path', type = str, default = './recognition/charset.txt')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=48, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=192, help='the width of the input image')
    parser.add_argument('--rgb', default = True, help='use rgb input')
    parser.add_argument('--stride', type = int, default = 1, help = 'stride for second building block')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    
    parser.add_argument('--Transformation', default = "None", type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', default = 'Mobile', type=str,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', default = 'BiLSTM', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default = 'Attn', help='Prediction stage. CTC|Attn')
    
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=576,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=48, help='the size of the LSTM hidden state')
    parser.add_argument('--iou_threshold', type=float, default=0.5 , help='detection_threshold')
    args = parser.parse_args()

    main(args)