---
layout: default
title: Home
---

원문: [Supervised learning: predicting an output variable from high-dimensional observations](https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)

# 지도 학습: 고차원 관측으로부터 출력 변수 예측하기

**지도 학습에서 해결하는 문제**

[지도 학습(supervised learning)](../../supervised_learning)은 두 데이터셋(datasets) 사이의 연결고리에 대한 학습으로 이루어집니다: 두 데이터셋은 관측(observed) 데이터 `X`와 목표값(target) 또는 레이블(labels)이라고 보통 부르는, 예측하고자 하는 외부 변수 `y`입니다. 대부분, `y`는 `n_samples`(표본 수) 길이의 1차원 배열(1D array)입니다.

사이킷런의 모든 지도 [`추정기(estimators)`](https://en.wikipedia.org/wiki/Estimator)는 모델(model)에 적합(fit)하기 위한 `fit(X, y)` 메서드(method)와 레이블 없는 관측 `X`을 받아 예측 레이블 `y`를 반환하는 `predict(X)` 메서드를 구현합니다.

**어휘: 분류와 회귀**

만약 예측 작업이 관측값을 유한한 레이블 집합 안에서 분류하는 것이라면, 다른 말로 관측된 개체에 "이름을 붙이는(name)" 것이라면, 작업은 **분류(classification)** 작업이라고 부릅니다. 반면에, 목표가 연속적인 목표 변수를 예측하는 거라면, **회귀(regression)** 작업이라고 부릅니다.

사이킷런에서 분류를 할 때, `y`는 정수(integers) 또는 문자열(strings)의 벡터(vector)입니다.

참고: 사이킷런에서 사용하는 기초 기계 학습 어휘에 대해 빠르게 훑어보려면 [사이킷런과 함께 하는 기계 학습 소개](../basic/tutorial)를 보세요.

## 최근접 이웃과 차원의 저주

**붓꽃 분류하기**

붓꽃(iris) 데이터셋은 붓꽃의 꽃잎(petal)과 꽃받침(sepal)의 길이(length)와 너비(width)에서 붓꽃의 3가지 유형(Setosa, Versicolour, 그리고 Verginica)을 식별하는 것으로 이루어지는 분류 작업입니다.

```python
>>> import numpy as np
>>> from sklearn import datasets
>>> iris_X, iris_y = datasets.load_iris(return_X_y=True)
>>> np.unique(iris_y)
array([0, 1, 2])
```

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dataset_001.png)

### k-최근접 이웃(k-nearest neighbors) 분류기

가장 간단한 가능한 분류기는 [최근접 이웃(nearest neighbor)](https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm)입니다: 새로운 관측 `X_test`이 주어지면, 훈련 세트(training set)(즉 추정기를 훈련하기 위해 사용한 데이터)에서 가장 가까운 특성 벡터(feature vector)가 있는 관측값을 찾는 것입니다. (이 유형의 분류기에 대한 더 많은 정보를 원하신다면 온라인 사이킷런 문서의 [최근접 이웃 섹션](../../modules/neighbors)을 보세요.)

**훈련 세트와 테스트 세트**

학습 알고리즘을 실험할 때, 추정기를 적합할 때 사용한 데이터로 추정기의 예측력을 테스트하지 않는 것이 중요합니다, 왜냐하면 이게 **새 데이터(new data)**에 대한 추정기의 성능을 평가하지는 못할 것이기 때문입니다. 이것이 종종 데이터셋을 *훈련(train)*과 *테스트(test)* 데이터로 나누는 이유입니다.

**KNN(k 최근접 이웃) 분류 예제:

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png)

```python
>>> # 붓꽃 데이터를 훈련과 테스트 세트로 나눕니다
>>> # 데이터를 무작위로 나누기 위한 무작위 순열(random permutation)
>>> np.random.seed(0)
>>> indices = np.random.permutation(len(iris_X))
>>> iris_X_train = iris_X[indices[:-10]]
>>> iris_y_train = iris_y[indices[:-10]]
>>> iris_X_test = iris_X[indices[-10:]]
>>> iris_y_test = iris_y[indices[-10:]]
>>> # 최근접-이웃 분류기를 만들고 적합합니다
>>> from sklearn.neighbors import KNeighborsClassifier
>>> knn = KNeighborsClassifier()
>>> knn.fit(iris_X_train, iris_y_train)
KNeighborsClassifier()
>>> knn.predict(iris_X_test)
array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
>>> iris_y_test
array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])
```

### 차원의 저주(the curse of dimensionality)

추정기가 효과적이기 위해서는, 문제에 따라 달라지는 어떤 값 $d$보다, 이웃하는 점들 사이의 거리가 작도록 해야 합니다. 일차원에서는, 평균적으로 $n \sim 1/d$의 점이 필요합니다. 위 $k$-NN 예제의 맥락에서는, 만약 데이터가 0부터 1까지의 범위이며 $n$개의 훈련 관측값이 있는 딱 하나의 특성으로 설명된다면, 새로운 데이터는 $1/n$보다 멀리 떨어져있으면 안됩니다. 따라서 최근접 이웃 결정 규칙은 $1/n$이 클래스간 특성 다양성(feature variations)의 규모에 비해 작을수록 효율적일 것입니다.

특성의 개수가 $p$라면, 여러분은 $n \sim 1/{d}^p$의 점이 필요합니다. 우리가 일차원에 10개의 점이 필요하다고 해봅시다: 이제 $[0, 1]$ 공간(space)을 덮으려면 $p$ 차원에서 ${10}^p$개의 점이 필요합니다. $p$가 커질수록, 좋은 추정기를 위해서 필요한 훈련 점의 개수는 지수적(exponentially)으로 증가합니다.

예를 들어, 각 점이 그냥 단일 숫자(8 바이트(bytes))라면, 소소한 $p \sim 20$ 차원에서의 효과적인 $k$-NN 추정기조차 전체 인터넷의 현재 추정 크기(±1000 엑사바이트(Exabytes)쯤)보다 더 많은 훈련 데이터를 필요로 할 것입니다.

이것을 [차원의 저주](https://en.wikipedia.org/wiki/Curse_of_dimensionality)라 부르고 기계 학습이 다루는 핵심 문제입니다.

## 선형 모델: 회귀부터 희소성까지


## 서포트 벡터 머신들(SVMs)
