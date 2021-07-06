# Artificial-intelligence-based-malicious-code-analysis

## 프로젝트 명
> 악성 파일 분석 및 기계 학습을 통한 악성코드 분류

## 프로젝트 목적
> -	Cuckoo sandbox를 통해서 악성파일과 정상 파일 분석
> -	기계학습을 기반으로 악성파일과 정상파일을 분류하는 모델 생성

## 순서도
> ![image](https://user-images.githubusercontent.com/69044270/122741988-7bf2de00-d2c0-11eb-8cba-265077f4f07c.png)


## 기계학습을 통한 악성코드 분류
> ![기계학습을 통한 악성코드 분류](https://user-images.githubusercontent.com/69952073/124561862-5269b300-de79-11eb-8855-b7a481070baa.PNG)

## 환경
> - 우분투 16.04 
> - Cuckoo sandbox 2.0.4
> - Tensorflow 2.5.0
> - Python 3.7.10

## 탐지 알고리즘
> - KNN알고리즘: K 1로 설정 -> accuracy: 83.85%
> - CNN알고리즘: Convolutional layer는 하나, max pool 사용, Adam optimizer를 사용, loss함수로 binary cross entropy를 사용. epoch을 100으로 설정하여 전체 데이터를 100번 학습 -> accuracy: 93.03%
> - Xgboost알고리즘: Bayesian Optimization을 활용하여 Xgboost의 튜닝 값 조절 -> accuracy: 91.58%
> - SVM 알고리즘: 커널 값 선형 사용, C값은 1000으로 설정 -> accuracy: 87%

## 최종 모델
> 4개의 알고리즘을 합친 앙상블 모델 제작.
> 
> 4개의 알고리즘 중 3개 이상의 모델이 1이 나오면 악성파일로 판단 -> accuracy: 86.77%
