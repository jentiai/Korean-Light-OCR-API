# Real-time Scene Text Detection with Differentiable Binarization



![network](imgs/paper/db.jpg)

## 



## Requirements
* pytorch 1.4+

* torchvision 0.5+

* gcc 4.9+

  original repo에 위와 같이 나와있는데 pytorch와 torchvision은 cuda 11.1에 맞춰서 설치하셔도 동작이 됩니다.

  ```
  pip install -r requirements.txt
  ```

  


## Train
config file이 config 폴더에 있습니다. mobilenetv3를 백본으로 하므로 지금은 아래의 config 파일을 만들어 사용했습니다.

``` 
config/icdar2015_dcn_mobilenetv3_FPN_DBhead_polyLR.yaml
```

* single gpu train
```bash
bash singlel_gpu_train.sh
```
* . Multi-gpu training
```bash
bash multi_gpu_train.sh
```


## Test

[eval.py](tools/eval.py) is used to test model on test dataset

1. config `model_path` in [eval.sh](eval.sh)
2. use following script to test
```bash
bash eval.sh
```



## Predict 

>  inference는 아래의 명령어를 실행 시키면 됩니다.

```
bash predict.sh
```


predict.sh를 보면 아래와 같이 돼있는데 경로에 맞게 수정하시면 됩니다,

```
CUDA_VISIBLE_DEVICES=0 python tools/predict.py --model_path "./output/DBNet_MobileNetV3_FPN_DBHead/checkpoint/model_best.pth" --input_folder "./datasets/test/img" --output_folder ./output --thre 0.7 --polygon --show 
```

