원문: [User Guide](https://scikit-learn.org/stable/user_guide.html)

# 사용자 안내

## [1. 지도 학습](supervised_learning)

### [1.1. 선형 모델](modules/linear_model)

- [1.1.1. 최소제곱법](modules/linear_model#최소제곱법)
- [1.1.2. 릿지 회귀와 분류](modules/linear_model#릿지-회귀와-분류)
- [1.1.3. 라쏘](modules/linear_model#라쏘)
- [1.1.4. 다중작업 라쏘](modules/linear_model#다중작업-라쏘)
- [1.1.5. 엘라스틱넷](modules/linear_model#엘라스틱넷)
- [1.1.6. 다중작업 엘라스틱넷](modules/linear_model#다중작업-엘라스틱넷)
- [1.1.7. 최소 각도 회귀](modules/linear_model#최소-각도-회귀)
- [1.1.8. LARS 라쏘](modules/linear_model#LARS-라쏘)
- [1.1.9. 직교 정합 추구(OMP)](modules/linear_model#직교-정합-추구)
- [1.1.10. 베이지안 회귀](modules/linear_model#베이지안-회귀)
- [1.1.11. 로지스틱 회귀](modules/linear_model#로지스틱-회귀)
- [1.1.12. 일반화 선형 모델](modules/linear_model#일반화-선형-모델)
- [1.1.13. 확률적 경사 하강 - SGD](modules/linear_model#확률적-경사-하강-SGD)
- [1.1.14. 퍼셉트론](modules/linear_model#퍼셉트론)
- [1.1.15. 수동 적대적 알고리즘](modules/linear_model#수동-적대적-알고리즘)
- [1.1.16. 강건성 회귀: 이상값과 모델링 오류](modules/linear_model#강건성-회귀-이상값과-모델링-오류)
- [1.1.17. 분위수 회귀](modules/linear_model#분위수-회귀)
- [1.1.18. 다항 회귀: 기저 함수로 선형 모델 확장](modules/linear_model#다항-회귀-기저-함수로-선형-모델-확장)

### [1.2. 선형과 이차 판별 분석](modules/lda_qda)

- [1.2.1. 선형 판별 분석을 사용한 차원 축소](modules/lda_qda#선형-판별-분석을-사용한-차원-축소)
- [1.2.2. LDA와 QDA 분류기의 수학 공식](modules/lda_qda#LDA와-QDA-분류기의-수학-공식)
- [1.2.3. LDA 차원 축소의 수학 공식](modules/lda_qda#LDA-차원-축소의-수학-공식)
- [1.2.4. 수축과 공분산 추정기](modules/lda_qda#수축과-공분산-추정기)
- [1.2.5. 추정 알고리즘](modules/lda_qda#추정-알고리즘)

### [1.3. 커널 릿지 회귀](modules/kernel_ridge)

### [1.4. 서포트 벡터 머신](modules/svm)

- [1.4.1. 분류](modules/svm#분류)
- [1.4.2. 회귀](modules/svm#회귀)
- [1.4.3. 밀도 추정, 특이값 탐지](modules/svm#밀도-추정-특이값-탐지)
- [1.4.4. 복잡도](modules/svm#복잡도)
- [1.4.5. 실제 사용 팁](modules/svm#실제-사용-팁)
- [1.4.6. 커널 함수](modules/svm#커널-함수)
- [1.4.7. 수학 공식](modules/svm#수학-공식)
- [1.4.8. 구현 상세](modules/svm#구현-상세)

### [1.5. 확률적 경사 하강](modules/sgd)

- [1.5.1. 분류](modules/sgd#분류)
- [1.5.2. 회귀](modules/sgd#회귀)
- [1.5.3. 온라인 단일클래스 SVM](modules/sgd#온라인-단일클래스-SVM)
- [1.5.4. 희소 데이터를 위한 확률적 경사 하강](modules/sgd#희소-데이터를-위한-확률적-경사-하강)
- [1.5.5. 복잡도](modules/sgd#복잡도)
- [1.5.6. 정지 기준](modules/sgd#정지-기준)
- [1.5.7. 실제 사용 팁](modules/sgd#실제-사용-팁)
- [1.5.8. 수학 공식](modules/sgd#수학-공식)
- [1.5.9. 구현 상세](modules/sgd#구현-상세)

### [1.6. 최근접 이웃](modules/neighbors)

- [1.6.1. 비지도 최근접 이웃](modules/neighbors#비지도-최근접-이웃)
- [1.6.2. 최근접 이웃 분류](modules/neighbors#최근접-이웃-분류)
- [1.6.3. 최근접 이웃 회귀](modules/neighbors#최근접-이웃-회귀)
- [1.6.4. 최근접 이웃 알고리즘](modules/neighbors#최근접-이웃-알고리즘)
- [1.6.5. 최근접 중심 분류기](modules/neighbors#최근접-중심-분류기)
- [1.6.6. 최근접 이웃 변환기](modules/neighbors#최근접-이웃-변환기)
- [1.6.7. 주변 성분 분석](modules/neighbors#주변-성분-분석)

### [1.7. 가우시안 과정](modules/gaussian_process)

- [1.1.1. 가우시안 과정 회귀(GPR)](modules/gaussian_process#가우시안-과정-회귀(GPR))
- [1.1.2. GPR 예시](modules/gaussian_process#GPR-예시)
- [1.1.3. 가우시안 과정 분류(GPC)](modules/gaussian_process#가우시안-과정-분류(GPC))
- [1.1.4. GPC 예시](modules/gaussian_process#GPC-예시)
- [1.1.5. 가우시안 과정을 위한 커널](modules/gaussian_process#가우시안-과정을-위한-커널)

### [1.8. 교차 분해](modules/cross_decomposition)

- [1.1.1. PLSCanonical](modules/cross_decomposition#PLSCanonical)
- [1.1.2. PLSSVD](modules/cross_decomposition#PLSSVD)
- [1.1.3. PLSRegression](modules/cross_decomposition#PLSRegression)
- [1.1.4. 정준 상관 분석](modules/cross_decomposition#정준-상관-분석)

### [1.9. 나이브 베이즈](modules/naive_bayes)

- [1.9.1. 가우시안 나이브 베이즈](modules/naive_bayes#가우시안-나이브-베이즈)
- [1.9.2. 다항 나이브 베이즈](modules/naive_bayes#다항-나이브-베이즈)
- [1.9.3. 보완 나이브 베이즈](modules/naive_bayes#보완-나이브-베이즈)
- [1.9.4. 베르누이 나이브 베이즈](modules/naive_bayes#베르누이-나이브-베이즈)
- [1.9.5. 범주형 나이브 베이즈](modules/naive_bayes#범주형-나이브-베이즈)
- [1.9.6. 아웃-오브-코어 나이브 베이즈 모델 적합](modules/naive_bayes#아웃-오브-코어-나이브-베이즈-모델-적합)

### [1.10. 의사결정나무](modules/tree)

- [1.10.1. 분류](modules/tree#분류)
- [1.10.2. 회귀](modules/tree#회귀)
- [1.10.3. 다중출력 문제](modules/tree#다중출력-문제)
- [1.10.4. 복잡도](modules/tree#복잡도)
- [1.10.5. 실제 사용 팁](modules/tree#실제-사용-팁)
- [1.10.6. 트리 알고리즘: ID3, C4.5, C5.0 그리고 CART](modules/tree#트리-알고리즘-ID3-C45-C50-그리고-CART)
- [1.10.7. 수학 공식](modules/tree#수학-공식)
- [1.10.8. 최소 비용-복잡도 가지치기](modules/tree#최소-비용-복잡도-가지치기)

### [1.11. 앙상블 방식](modules/ensemble)

- [1.11.1. 배깅 메타추정기](modules/ensemble#배깅-메타추정기)
- [1.11.2. 무작위 나무의 숲](modules/ensemble#무작위-나무의-숲)
- [1.11.3. 에이다부스트](modules/ensemble#에이다부스트)
- [1.11.4. 경사 트리 부스팅](modules/ensemble#경사-트리-부스팅)
- [1.11.5. 히스토그램 기반 경사 부스팅](modules/ensemble#히스토그램-기반-경사-부스팅)
- [1.11.6. 보팅 분류기](modules/ensemble#보팅-분류기)
- [1.11.7. 보팅 회귀자](modules/ensemble#보팅-회귀자)
- [1.11.8. 스택 일반화](modules/ensemble#스택-일반화)

### [1.12. 다중클래스와 다중출력 알고리즘](modules/multiclass)

- [1.12.1. 다중클래스 분류](modules/multiclass#다중클래스-분류)
- [1.12.2. 다중레이블 분류](modules/multiclass#다중레이블-분류)
- [1.12.3. 다중클래스-다중출력 분류](modules/multiclass#다중클래스-다중출력-분류)
- [1.12.4. 다중출력 회귀](modules/multiclass#다중출력-회귀)

### [1.13. 특성 선택](modules/feature_selection)

- [1.13.1. 저분산 특성 제거](modules/feature_selection#저분산-특성-제거)
- [1.13.2. 단변량 특성 선택](modules/feature_selection#단변량-특성-선택)
- [1.13.3. 재귀적 특성 제거](modules/feature_selection#재귀적-특성-제거)
- [1.13.4. SelectFromModel로 특성 선택](modules/feature_selection#SelectFromModel로-특성-선택)
- [1.13.5. 순차적 특성 선택](modules/feature_selection#순차적-특성-선택)
- [1.13.6. 파이프라인의 일부로 특성 선택](modules/feature_selection#파이프라인의-일부로-특성-선택)

### [1.14. 준지도 학습](modules/semi_supervised)

- [1.14.1. 자기 훈련](modules/semi_supervised#자기-훈련)
- [1.14.2. 레이블 전파](modules/semi_supervised#레이블-전파)

### [1.15. 등위 회귀](modules/isotonic)

### [1.16. 확률 교정](modules/calibration)

- [1.16.1. 교정 곡선](modules/calibration#교정-곡선)
- [1.16.2. 분류기 교정](modules/calibration#분류기-교정)
- [1.16.3. 사용법](modules/calibration#사용법)

### [1.17. 신경망 모델(지도)](modules/neural_networks_supervised)

- [1.17.1. 다층 퍼셉트론](modules/neural_networks_supervised#다층-퍼셉트론)
- [1.17.2. 분류](modules/neural_networks_supervised#분류)
- [1.17.3. 회귀](modules/neural_networks_supervised#회귀)
- [1.17.4. 정규화](modules/neural_networks_supervised#정규화)
- [1.17.5. 알고리즘](modules/neural_networks_supervised#알고리즘)
- [1.17.6. 복잡도](modules/neural_networks_supervised#복잡도)
- [1.17.7. 수학 공식](modules/neural_networks_supervised#수학-공식)
- [1.17.8. 실제 사용 팁](modules/neural_networks_supervised#실제-사용-팁)
- [1.17.9. warm_start로 더 조절](modules/neural_networks_supervised#warm_start로-더-조절)

## [2. 비지도 학습](unsupervised_learning)

### [2.1. 가우시안 혼합 모델](modules/mixture)

### [2.2. 매니폴드 학습](modules/manifold)

### [2.3. 군집화](modules/clustering)

### [2.4. 이중군집화](modules/biclustering)

### [2.5. 성분 신호 분해(행렬 분해 문제)](modules/decomposition)

### [2.6. 공분산 추정](modules/covariance)

### [2.7. 특이값과 이상값 탐지](modules/outlier_detection)

### [2.8. 밀도 추정](modules/density)

### [2.9. 신경망 모델(비지도)](modules/neural_networks_unsupervised)

## [3. 모델 선택과 평가](model_selection)

### [3.1. 교차 검증: 추정기 성능 평가](modules/cross_validation)

### [3.2. 추정기 초매개변수 조정](modules/grid_search)

### [3.3. 측정과 채점: 예측 품질 수량화](modules/model_evaluation)

### [3.4. 검증 곡선: 모델 평가를 위한 점수 도표화](modules/learning_curve)

## [4. 검사](inspection)

### [4.1. 부분 의존도와 개별 조건부 기대치 도표](modules/partial_dependence)

### [4.2. 순열 특성 중요도](modules/permutation_importance)

## [5. 시각화](visualizations)

### [5.1. 사용 가능한 도표화 유틸리티](visualizations#사용-가능한-도표화-유틸리티)

## [6. 데이터셋 변환](data_transforms)

### [6.1. 파이프라인과 복합 추정기](modules/compose)

### [6.2. 특성 추출](modules/feature_extraction)

### [6.3. 데이터 전처리](modules/preprocessing)

### [6.4. 결측값 대체](modules/impute)

### [6.5. 비지도 차원 축소](modules/unsupervised_reduction)

### [6.6. 무작위 투영](modules/random_projection)

### [6.7. 커널 근사](modules/kernel_approximation)

### [6.8. 쌍별 측정, 근연성 그리고 커널](modules/metrics)

### [6.9. 예측 목표(y) 변환하기](modules/preprocessing_targets)

## [7. 데이터셋 불러오기 유틸리티](datasets)

### [7.1. 장난감 데이터셋](datasets/toy_dataset)

### [7.2. 현실 세계 데이터셋](datasets/real_world)

### [7.3. 생성된 데이터셋](datasets/sample_generators)

### [7.4. 다른 데이터셋 불러오기](datasets/loading_other_datasets)

## [8. 사이킷런으로 계산하기](computing)

### [8.1. 계산적으로 스케일링하기 위한 전략: 더 큰 데이터](computing/scaling_strategies)

### [8.2. 계산 성능](computing/computational_performance)

### [8.3. 병렬성, 자원 관리, 그리고 설정](computing/parallelism)

## [9. 모델 영속성](model_persistence)

### [9.1. 파이썬 특이적 직렬화](model_persistence#파이썬-특이적-직렬화)

### [9.2. 상호 운용 가능한 형식](model_persistence#상호-운영-가능한-형식)

## [10. 일반적인 함정과 추천 사례](common_pitfalls)

### [10.1. 일관성 없는 전처리](common_pitfalls#일관성-없는-전처리)

### [10.2. 데이터 누수](common_pitfalls#데이터-누수)

### [10.3. 무작위성 제어하기](common_pitfalls#무작위성-제어하기)

## [11. 특보](dispatching)

### [11.1. (실험적인) 어레이 API 지원](modules/array_api)
