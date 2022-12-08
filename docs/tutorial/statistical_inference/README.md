원문: [A tutorial on statistical-learning for scientific data processing](https://scikit-learn.org/stable/tutorial/index.html)

# 과학적인 데이터 처리를 위한 통계적 학습 튜토리얼

**통계적 학습**

[기계 학습(machine learning)](https://en.wikipedia.org/wiki/Machine_learning)은 실험 과학이 마주하고 있는 데이터셋(datasets)의 크기가 빠르게 커져감에 따라, 그 중요성이 커지고 있는 기술입니다. 기계 학습이 부딪히는 문제는 서로 다른 관측들을 연결하는 예측 함수를 구축하는 것부터, 관측들을 분류하거나, 레이블되지 않은 데이터셋 속의 구조를 배워가는 것까지 다양합니다.

이 튜토리얼은 [통계적 추론(statistical inference)](https://en.wikipedia.org/wiki/Statistical_inference)를 목표로 기계 학습 기술들을 사용하는 방법론인 *통계적 학습*을 탐구할 것입니다: 손에 쥔 데이터에 대한 결론을 만들어가는 것이죠.

사이킷런은 과학 파이썬(Python) 패키지들([넘파이(NumPy)](https://www.numpy.org/), [사이파이(SciPy)](https://scipy.org/), [맷플롯립(matplotlib)](https://matplotlib.org/))의 밀도 있게 짜여진 세계에서 고전적인 기계 학습 알고리즘들을 통합한 파이썬 모듈(module)입니다.

### - [통계적 학습: 사이킷런에서의 설정과 추정기 객체](settings)

- [데이터셋](settings#데이터셋)
- [추정기 객체](settings#추정기-객체)

### - [지도 학습: 고차원 관측으로부터 출력 변수 예측하기](supervised_learning)

- [최근접 이웃과 차원의 저주](supervised_learning#최근접-이웃과-차원의-저주)
- [선형 모델: 회귀부터 희소성까지](supervised_learning#선형-모델-회귀부터-희소성까지)
- [서포트 벡터 머신들(SVMs)](supervised_learning#서포트-벡터-머신들(SVMs))

### - [모델 선택: 추정기와 매개변수 선택하기](model_selection)

- [점수, 그리고 교차 검증된 점수](model_selection#점수,-그리고-교차-검증된-점수)
- [교차 검증 생성기](model_selection#교차-검증-생성기)
- [격자 탐색과 교차 검증된 추정기](model_selection#격자-탐색과-교차-검증된-추정기)

### - [비지도 학습: 데이터 표현법 찾기](unsupervised_learning)

- [군집화: 관측을 함께 묶기](unsupervised_learning#군집화-관측을-함께-묶기)
- [분해: 신호부터 성분과 부하량까지](unsupervised_learning#분해-신호부터-성분과-부하량까지)

### - [모두 모으기](putting_together)

- [파이프라이닝](putting_together#파이프라이닝)
- [고유얼굴로 얼굴 인식](putting_together#고유얼굴로-얼굴-인식)
- [열린 문제: 주식 시장 구조](putting_together#열린-문제-주식-시장-구조)
