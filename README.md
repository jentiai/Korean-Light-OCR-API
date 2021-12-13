# Light-OCR-API

## 1. 경량화 OCR 모델 구조
저희 경량화 OCR 모델은 Detector (텍스트 탐지 모듈) 과  Recognizer (텍스트 인식 모듈) 로 구분됩니다. 아래는 각 모듈에 대한 요약입니다. 
### (1) Detector
- 텍스트 탐지 모듈 그림

    <center><img src= "https://user-images.githubusercontent.com/55676509/145774041-72dc110e-d5e7-464f-93c9-5d927f8c65ba.PNG" width = "50%" height = "50%"></center>

- 텍스트 탐지 모듈 요약
<center>

| Model | 가로x세로 | RGB | TPS | FeatureExtraction | SequenceModel | Prediction | 파라미터 수 | size(MB) | 정확도 |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BASE (H) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 88.524% |
| **BEST** (H) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **90.709%** |
| BASE (V) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 87.234% |
| **BEST** (V) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **89.821%** |

</center>


### (2) Recognizer
- 텍스트 인식 모듈 그림
    <center><img src ="https://user-images.githubusercontent.com/55676509/145774122-6d2cf8b4-e701-46d3-a725-44b59f2b790f.PNG" width = "50%" height = "50%"></center>

- 텍스트 인식 모듈 요약
<center>

| Model | 가로x세로 | RGB | TPS | FeatureExtraction | SequenceModel | Prediction | 파라미터 수 | size(MB) | 정확도 |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BASE (H) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 88.524% |
| **BEST** (H) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **90.709%** |
| BASE (V) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | CTC | 1,216,100 | 4.86 | 87.234% |
| **BEST** (V) | 192 x 48 | True | None | MobileNetV3 (576) | BiLSTM (48) | Attention | 1,246,532 | 4.99 | **89.821%** |

</center>

## 2. API 사용법


## 3. 참고 자료

[1] PP-OCR

[2] Differentiable Binarization

[3] Text detection benchmark