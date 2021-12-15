import os
import json
import random
import argparse

import numpy as np

import torch
import torch.nn.functional as F

from recognition.model import Model as Recognizer
from recognition.utils import CTCLabelConverter, AttnLabelConverter
from tools import *

import pathlib

from tqdm import tqdm
from detection.tools.detector import Detector
from detection.utils.util import show_img, draw_bbox, save_result, get_file_list

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
    print('\n' + '-'*33 + f' JENTI KOREAN OCR ' + '-'*33 + '\n')
    ocr_checkpoint_h = args.ocr_checkpoint[:-4] + f"_{args.Prediction}_{args.imgH}_{args.imgW}_st_{args.stride}_h.pth"
    ocr_checkpoint_v = args.ocr_checkpoint[:-4] + f"_{args.Prediction}_{args.imgH}_{args.imgW}_st_{args.stride}_v.pth"

    if not os.path.isdir(args.detection_output_dir):
        os.makedirs(args.detection_output_dir, exist_ok = True)
    if not os.path.isdir(args.ocr_output_dir):
        os.makedirs(args.ocr_output_dir, exist_ok = True)
    if not os.path.isfile(args.detection_checkpoint):
        change_checkpoint(args.detection_checkpoint)

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
    
    # compose Detector
    detector = Detector(model_path = args.detection_checkpoint,
                        detection_config_path = args.detection_config_file,
                        device = device,
                        post_p_thre = args.detection_threshold)
    print(f'[Ready] Detector')
    print(f'        - {"config":<12s}: {os.path.basename(args.detection_config_file)}')
    print(f'        - {"checkpoint":<12s}: {os.path.basename(args.detection_checkpoint)}')

    # compose Recognizer
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
    
    recognizer_h = Recognizer(args)
    recognizer_h.load_state_dict(copyStateDict(torch.load(ocr_checkpoint_h)))
    recognizer_h = recognizer_h.to(device)
    recognizer_h.eval()
    
    if os.path.isfile(ocr_checkpoint_v):
        recognizer_v = Recognizer(args)
        recognizer_v.load_state_dict(copyStateDict(torch.load(ocr_checkpoint_v)))
        recognizer_v = recognizer_v.to(device)
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

    print(f'[Prediction] Now detect texts...')
    for img_path in tqdm(get_file_list(args.detection_data_dir, p_postfix=['.jpg']), bar_format= '{percentage:3.0f}%|{bar:49}{r_bar}'):
        _, boxes_list, score_list, t = detector.predict(img_path)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        os.makedirs(args.detection_output_dir, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.detection_output_dir, img_path.stem + '_result.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, False)
    det_pred = crop_polygon_with_persp(args.detection_data_dir, args.detection_output_dir, args)
    print(f'[Prediction] Detection is done:')
    print(f'       - # of detected texts: {len(det_pred.values())}\n')
    
    print(f'[Prediction] Now recognize texts...')    
    with torch.no_grad():
        for k, v in tqdm(det_pred.items(), bar_format= '{percentage:3.0f}%|{bar:49}{r_bar}'):
            if os.path.isfile(os.path.join(args.ocr_output_dir, str(k) + '.txt')):
                os.remove(os.path.join(args.ocr_output_dir, str(k) + '.txt'))
            with open(os.path.join(args.ocr_output_dir, str(k)+'.json'), 'w', encoding = 'UTF-8-sig') as json_res:    
                dict_res = {}
                with open(os.path.join(args.ocr_output_dir, str(k) + '.txt'), 'wb') as txt_res:
                    res_h, res_v = v[0]
                    if len(res_h) >= 1 and res_h[0] != []:
                        for crop, bbox, conf in res_h:
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

                    if len(res_v) >= 1 and res_v[0] != []:
                        for crop, bbox, conf in res_v:
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

                json.dump(dict_res, json_res, ensure_ascii=False, indent = "\t")

    print(f"[Prediction] Recognition is done\n")
    visualization_poly(args.ocr_output_dir, args.detection_data_dir)
    print(f"[Prediction] Visualization is done: {args.ocr_output_dir}")
    print('\n'+f'-'*32 + f' PREDICTION DONE !! ' + '-'*32 + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--detection_data_dir', type=str, default = './sample/', help='detection data path')
    parser.add_argument('--detection_output_dir', type=str, default = './detection/output/predict/', help='detection output path')
    parser.add_argument('--detection_checkpoint', type=str, default = './checkpoint/detection_best.pth', help='detection_checkpoint')
    parser.add_argument('--detection_config_file', type=str, default='./detection/config/icdar2015_dcn_mobilenetv3_FPN_DBhead_polyLR.yaml')
    
    parser.add_argument('--ocr_output_dir', type = str, default = './ocr_output')
    parser.add_argument('--ocr_checkpoint', type = str, default = './checkpoint/best_accuracy.pth')
    parser.add_argument('--charset_path', type = str, default = './recognition/charset.txt')
    
    # detection
    parser.add_argument('--detection_threshold', type=str, default=0.5 , help='detection_threshold')
    parser.add_argument('--hv_ratio', type=str, default=2 , help='hv_ratio')
    
    # recognition
    parser.add_argument('--imgH', type=int, default=48, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=192, help='the width of the input image')
    parser.add_argument('--stride', type = int, default = 1, help = 'stride for second building block')
    parser.add_argument('--rgb', default = True, help='use rgb input')
    parser.add_argument('--character', type=str, default=' ', help='character label')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
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
    args = parser.parse_args()

    main(args)