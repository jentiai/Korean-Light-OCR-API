import os
import json
import sys
import argparse

def evaluate_jenti(args): 
    with open(args.api_res_json, 'r', encoding = 'UTF-8-sig') as t:
        json_data = json.load(t)

def evaluate_naver(args):
    with open(args.api_res_json, 'r', encoding = 'UTF-8-sig') as t:
        json_data = json.load(t)

def evaluate_kakao(args):
    with open(args.api_res_json, 'r', encoding = 'UTF-8-sig') as t:
        json_data = json.load(t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--api_res_json', type = str
                        help = 'api to evaluation')
    parser.add_argument('--gt_path', type = str,
                        help = 'path to gt')
    args = parser.parse_args()

    if 'naver' in args.api_res_json:
        evaluate_naver(args)
    elif 'kakao' in args.api_res_json:
        evaluate_kakao(args)

    elif 'jenti' in args.pai_res_json:
        evaluate_jenti(args)
