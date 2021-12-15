import os
import requests
import argparse
import json

from tools import visualization


def main(args):
    img_dir = args.img_dir
    output_dir = args.output_dir
    url = args.url

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    
    with open('./jenti.json', 'w', encoding = 'UTF-8-sig') as json_res:
        res_dict = {}
        for img_name in list(filter(lambda x: x.find('.jpg') != -1 or x.find('.png') != -1, os.listdir(img_dir))):
            img_path = os.path.join(img_dir, img_name)
            files = {'file': open(img_path, 'rb').read()}
            r = requests.post(url, files = files)
            visualization(img_path,  r.json(), output_dir)
            res_dict[os.path.splitext(img_name)[0]] = r.json()
        
        json.dump(res_dict, json_res, ensure_ascii = False, indent = '\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type = str,
                        help = 'input image directory')
    parser.add_argument('--output_dir', type = str,
                        help = 'output image directory')
    parser.add_argument('--url', type = str, default="http://27.255.77.102:5000/evaluation", help = 'output image directory')

    args = parser.parse_args()

    main(args)    
    