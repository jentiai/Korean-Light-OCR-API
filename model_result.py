import os
import requests
import argparse

from tools import visualization


def main():

    output_dir = "./output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    url = "http://27.255.77.102:5000/evaluation"
    img_path = './1.jpg'
    files = {'file': open(img_path, 'rb').read()}
    r = requests.post(url, files = files)

    visualization(img_path,  r.json(), output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_res_json', type = str,
                        help = 'path to pred_all.json')
    parser.add_argument('--txt_gt_dir', type = str,
                        help = 'path to gt dir')
    args = parser.parse_args()

    Reader = ReadPredGT(args)    
    pred_json = Reader.read_pred_json()