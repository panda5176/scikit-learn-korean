원문: [An introduction to machine learning with scikit-learn](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

# 사이킷런과 함께 하는 기계 학습 소개

**섹션 내용**

이 섹션에서는, 사이킷런에서 전반적으로 사용하는 [기계 학습(machine learning)](https://en.wikipedia.org/wiki/Machine_learning) 어휘를 소개하고 간단한 학습 예시를 제공합니다.

## 기계 학습: 문제 설정

일반적으로, 학습 문제는 n개 [표본(samples)](https://en.wikipedia.org/wiki/Sample_(statistics))의 데이터 집합을 고려하여 알려지지 않은 데이터의 속성을 예측하고자 합니다. 만약 각 표본이 하나보단 많고, 예를 들어, 다차원 항목(일명 [다변량(multivariate)](https://en.wikipedia.org/wiki/Multivariate_random_variable) 데이터)일 때, 이를 여러 속성(attributes) 또는 **특성(features)**이 있다고 말합니다.

학습 문제는 몇 가지 범주로 나뉩니다:

- [지도 학습(supervised learning)](https://en.wikipedia.org/wiki/Supervised_learning)에서는, 우리가 예측하고자 하는 추가적인 속성이 데이터에 함께 제공됩니다(사이킷런 지도 학습 페이지로 가시려면 [여기](supervised_learning)를 클릭하세요). 이 문제는 다음 중 하나일 수 있습니다:

  - [분류(classification)](https://en.wikipedia.org/wiki/Classification_in_machine_learning): 표본은 둘 또는 그 이상의 클래스에 속하며 우리는 이미 레이블된(labeled) 데이터에서 어떻게 레이블되지 않은(unlabeled) 데이터의 클래스를 예측하는지 배우고 싶어합니다. 분류 문제의 예시로는 손글씨 숫자 인식이 있습니다, 이 때 목표는 각 입력 벡터(vector)를 유한한 숫자의 이산형 범주(discrete categories) 중 하나에 할당하는 것입니다. 분류를 생각해볼 수 있는 또다른 방법으로는 제한된 숫자의 범주가 있고 각각 n개의 표본이 제공되며, 표본들에 올바른 범주나 클래스를 레이블하고자 하는 이산형(연속형(continuous)의 반대) 형태의 지도 학습입니다.
  - [회귀(regression)](https://en.wikipedia.org/wiki/Regression_analysis): 만약 원하는 출력이 하나 이상의 연속형 변수라면, 이 작업은 *회귀*라고 부릅니다. 회긔 문제의 예시로는 연어의 길이를 나이와 무게의 함수로 예측하는 것이 있습니다.

- [비지도 학습](https://en.wikipedia.org/wiki/Unsupervised_learning)은, 훈련 데이터를 구성하는 입력 벡터 x가 대응하는 목표값이 전혀 없는 경우입니다. 이런 문제의 목표는 데이터 안에서 비슷한 예시들의 그룹을 찾는 것, 즉 [군집화(clustering)](https://en.wikipedia.org/wiki/Cluster_analysis)거나, 입력 공간(space) 속에서 데이터의 분포를 결정, 즉 [밀도 추정(density estimation)](https://en.wikipedia.org/wiki/Density_estimation)이거나, 또는 *시각화(visualization)* 목적으로 데이터를 고차원(high-dimensional) 공간에서 이차원 또는 삼차원 공간으로 투사(project)하는 것입니다(사이킷런 비지도 학습 페이지로 가시려면 [여기](unsupervised_learning)를 클릭하세요).

**훈련 세트(training set)와 테스팅 세트(testing set)**

기계 학습은 데이터 세트의 일부 속성을 학습한 다음 이 속성들을 다른 데이터 세트에 대해 테스트하는 것입니다. 기계 학습의 일반적인 관행은 데이터 세트를 둘로 분할해 알고리즘을 평가하는 것입니다. 우리는 이 세트 중 하나를 **훈련 세트**라 부르는데, 이는 일부 속성을 학습하는 곳입니다; 다른 세트는 **테스팅 세트**라 부르며, 학습한 속성을 테스트하는 곳입니다.

## 예시 데이터셋 로딩

## 학습과 예측

## 컨벤션
