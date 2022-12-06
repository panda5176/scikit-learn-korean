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

**훈련 세트(training set)와 테스트 세트(testing set)**

기계 학습은 데이터 세트의 일부 속성을 학습한 다음 이 속성들을 다른 데이터 세트에 대해 테스트하는 것입니다. 기계 학습의 일반적인 관행은 데이터 세트를 둘로 분할해 알고리즘을 평가하는 것입니다. 우리는 이 세트 중 하나를 **훈련 세트**라 부르는데, 이는 일부 속성을 학습하는 곳입니다; 다른 세트는 **테스트 세트**라 부르며, 학습한 속성을 테스트하는 곳입니다.

## 예시 데이터셋 불러오기

`Scikit-learn`에는 몇 가지 표준적인 데이터셋이 함께 들어있는데, 예시로 분류를 위한 [붓꽃(iris)](https://en.wikipedia.org/wiki/Iris_flower_data_set)과 [숫자(digits)](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) 데이터셋과 회귀를 위한 [당뇨병(diabetes)](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) 등이 있습니다.

다음에서는, 우리의 셸(shell)에서 파이썬 인터프리터(Python interpreter)를 시작해 `iris`과 `digits` 데이터셋을 불러옵니다. 우리의 표기 규칙에서는 `$`이 셸 프롬프트(prompt)를 나타내고 `>>>`이 파이썬 인터프리터 프롬프트를 나타냅니다:

```python
$ python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> digits = datasets.load_digits()
```

데이터셋은 모든 데이터와 데이터에 대한 일부 메타데이터(metadata)를 담고 있는 사전과 같은(dictionary-like) 객체입니다. 이 데이터는 `.data` 멤버(member)에 저장되는데, 이는 `n_samples, n_features` 배열입니다. 지도 문제의 경우, 하나 이상의 반응 변수가 `.target` 멤버에 저장됩니다. 다양한 데이터셋에 대한 자세한 내용은 [전용 섹션](datasets)에서 찾아볼 수 있습니다.

예를 들어, 숫자 데이터셋의 경우, `digits.data`는 숫자 표본을 분류하는데 사용할 수 있는 특성에 접근하게 해줍니다:

```python
>>> print(digits.data)
[[ 0.   0.   5. ...   0.   0.   0.]
 [ 0.   0.   0. ...  10.   0.   0.]
 [ 0.   0.   0. ...  16.   9.   0.]
 ...
 [ 0.   0.   1. ...   6.   0.   0.]
 [ 0.   0.   2. ...  12.   0.   0.]
 [ 0.   0.  10. ...  12.   1.   0.]]
```

그리고 `digits.target`은 숫자 데이터셋에 대한 실제 정답을 제공하는데, 이는 우리가 배우고자 하는 각 숫자 이미지에 대응하는 숫자입니다:

```python
>>> digits.target
array([0, 1, 2, ..., 8, 9, 8])
```

**데이터 배열의 형태**

데이터는 항상 2차원(2D) 배열, `(n_samples, n_features)` 형태입니다, 원본 데이터가 다른 형태를 가졌더라도 말입니다. 숫자의 경우, 각 원본 표본은 `(8, 8)` 형태의 이미지이며 다음처럼 접근할 수 있습니다:

```python
>>> digits.images[0]
array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
       [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
       [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
       [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
       [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
       [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
       [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
       [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])
```

[이 데이터셋의 간단한 예제](auto_examples/classification/plot_digits_classification)는 사이킷런에서 어떻게 원래 문제에서 시작해서 사용할만한 데이터의 형태를 만드는지 보여줍니다.

**외부 데이터셋에서 불러오기**

외부 데이터셋에서 불러오려면, [외부 데이터셋에서 불러오기](datasets/loading_other_datasets#external-datasets)를 참고해주세요.

## 학습과 예측

## 컨벤션
