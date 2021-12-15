# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

# project = 'DBNet.pytorch'  # 工作项目根目录
# sys.path.append(os.getcwd().split(project)[0] + project)
import time
import cv2
import torch
from tqdm import tqdm
from data_loader import get_transforms, get_dataloader
from models import build_model
from post_processing import get_post_processing
from utils import get_metric
import torchsummary
import anyconfig


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Detector:
    def __init__(self, model_path, detection_config_path, device, post_p_thre=0.7):

        self.device = device
        config = anyconfig.load(open(detection_config_path, 'rb'))
        checkpoint = torch.load(model_path, map_location=self.device)
        config['arch']['backbone']['pretrained'] = False
        config['distributed'] = False

        config['dataset']['type'] = "ICDAR2015Dataset"
        self.model = build_model(config['arch'])
        self.model.load_state_dict(checkpoint)
        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        
        self.model.to(self.device)        
        self.model.eval()

        self.transform = []
        for t in [{'type': 'ToTensor', 'args': {}}, {'type': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}]:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def eval(self):
        self.config['dataset']['validate']['dataset']['args']['data_path'] = ['./datasets/predict.txt']
        self.validate_loader = get_dataloader(config['dataset']['validate'], self.config['distributed'])

        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds,is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t
    

def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument('--detection_config_path', type=str, default='./config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml')
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.3,type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    # 初始化网络

    model = Detector(args.model_path, args.detection_config_path, post_p_thre=args.thre, gpu_id=0)

    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=args.polygon)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)
