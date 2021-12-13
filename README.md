# Light-OCR-API

## 1. 경량화 OCR 모델 구조
저희 경량화 OCR 모델은 Detector (텍스트 탐지 모듈) 과  Recognizer (텍스트 인식 모듈) 로 구분됩니다. 아래는 각 모듈에 대한 요약입니다. 
### (1) Detector
- 텍스트 탐지 모듈 그림

    <center>
    <img src= "https://user-images.githubusercontent.com/55676509/145774041-72dc110e-d5e7-464f-93c9-5d927f8c65ba.PNG" width = "80%" height = "80%">
    </center>

- 텍스트 탐지 모듈 요약
<center>

| Model | FeatureExtraction | SequenceModel | Prediction | 파라미터 수 | size(MB) | 정확도 |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: |
| BASE (H) | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 88.524% |
| **BEST** (H) | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **90.709%** |
| BASE (V) | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 87.234% |
| **BEST** (V) | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **89.821%** |

</center>


### (2) Recognizer
- 텍스트 인식 모듈 그림
    <center><img src ="https://user-images.githubusercontent.com/55676509/145774122-6d2cf8b4-e701-46d3-a725-44b59f2b790f.PNG" width = "100%" height = "10%"></center>

- 텍스트 인식 모듈 요약
<center>

| Model | FeatureExtraction | SequenceModel | Prediction | 파라미터 수 | size (MB) | 정확도 (%)|
| :---: | :---: | :---: |:---: | :---: | :---: | :---: |
| BASE (H) | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 88.524 |
| **BEST** (H) | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **90.709** |
| BASE (V) | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 87.234 |
| **BEST** (V) | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **89.821** |

</center>

텍스트 인식 모듈은 TPS를 사용하지 않고, ``MobileNetV3 - BiLSTM - {Attention, CTC}`` 를 이용하는 구조입니다. MobileNetV3와 BiLSTM의 경우 각각 차원을 576, 48로 사용하여 기존보다 파라미터 수를 대폭 감소시킬 수 있었습니다. 모델의 입력은 가로와 세로의 길이가 각각 192, 48인 RGB 이미지입니다. 한글의 경우 가로 글씨 (H) 와 세로 글씨 (V) 를 각각 인식하기 위해 구조가 동일한 두 개의 모델을 각각의 데이터에 대해 학습하였습니다. 마지막 추론 모듈을 CTC에서 Attention으로 교환할 시에 파라미터 수는 약 3만개 증가하고, size는 0.13 (MB) 증가한 반면 정확도를 1-2% 개선시킬 수 있기에 전체 OCR 과정의 기본값을 Attention을 이용한 추론 모듈로 하였습니다. ``TPS - ResNet50 - BiLSTM - Attention``을 이용한 모델 대비, 전체 모델의 size는 약 38배 감소하였으며, 정확도는 2-3% 정도 떨어진 성능을 기록하고 있습니다.

## 2. API 사용법
- 사용법

## 3. 참고 자료

[1] PP-OCR

[2] Differentiable Binarization

[3] Text detection benchmark
