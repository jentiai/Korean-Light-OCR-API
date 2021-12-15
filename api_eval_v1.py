import os
import json
import argparse
from tools import evaluation

class ReadPredGT:
    def __init__(self, args):
        self.api_res_json = args.api_res_json
        self.txt_gt_dir = args.txt_gt_dir

    def read_pred_json(self):
        with open(self.api_res_json, 'r', encoding = 'UTF-8-sig') as j:
            pred_json = json.load(j)
        return pred_json

    def read_naver_pred(self, v_pred):
        self.pred_list = []
        images = v_pred["images"][0]

        fields = images["fields"]
        for field in fields:
            bounding_polys = field["boundingPoly"]["vertices"]
            inferText = field["inferText"]
            line = []
            for pts in bounding_polys:
                line.extend(map(int, [pts["x"], pts["y"]]))
            line.append(inferText)
            self.pred_list.append(line)
        
    def read_kakao_pred(self, v_pred):
        self.pred_list = []
        if "result" in v_pred.keys():
                result = v_pred["result"]
                for res in result:
                    line = []
                    for pts in res["boxes"]:
                        line.extend(map(int, pts))
                    line.append(res["recognition_words"][0])
                    self.pred_list.append(line)

    def read_jenti_pred(self, v_pred):
        self.pred_list = []
        for rec, det in v_pred.items():
            line = det + [rec]
            self.pred_list.append(line)

    def read_pred(self, v_pred):
        if 'naver' in os.path.basename(self.api_res_json):
            self.read_naver_pred(v_pred)
        elif 'kakao' in os.path.basename(self.api_res_json):
            self.read_kakao_pred(v_pred)
        elif 'jenti' in os.path.basename(self.api_res_json):
            self.read_jenti_pred(v_pred)
        return self.pred_list

    def read_txt_gt(self, k_img_name):
        gt_list = []
        with open(os.path.join(self.txt_gt_dir, k_img_name + '.txt'), 'r') as t:
            while True:
                line = t.readline().strip('\n').strip(' ')
                if not line:
                    break
                else:
                    line_set = line.split(',')
                    det = list(map(int, line_set[:8]))
                    rec = line_set[8].strip(' ')
                    if line.endswith(','):
                        rec += ','
                    temp = [det[0], det[1], det[2], det[3], det[4], det[5], det[6], det[7], rec]
                    gt_list.append(temp)
        return gt_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_res_json', required = True, type = str,
                        help = 'path to pred_all.json')
    parser.add_argument('--txt_gt_dir', required = True, type = str,
                        help = 'path to gt dir')
    args = parser.parse_args()

    Reader = ReadPredGT(args)    
    pred_json = Reader.read_pred_json()

    total_num = 0 # 총 데이터 수
    macro = 0
    micro_numerator = 0
    micro_denominator = 0

    for i, (k_img_name, v_pred) in enumerate(pred_json.items()):
        pred_list = Reader.read_pred(v_pred)
        gt_list = Reader.read_txt_gt(k_img_name)
        iou_threshold = 0.1
        correct_num, gt_num = evaluation(pred_list, gt_list, iou_threshold)

        if gt_num:
            total_num += 1
            macro += correct_num / gt_num
            micro_numerator += correct_num
            micro_denominator += gt_num

    print(f'marcro: {macro / total_num}')
    print(f'mircro: {micro_numerator/ micro_denominator}')         
