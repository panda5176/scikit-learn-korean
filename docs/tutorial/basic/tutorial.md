원문: [An introduction to machine learning with scikit-learn](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

# 사이킷런과 함께 하는 기계 학습 입문

**섹션 내용**

이 섹션에서는, 사이킷런에서 전반적으로 사용하는 [기계 학습(machine learning)](https://en.wikipedia.org/wiki/Machine_learning) 어휘를 소개하고 간단한 학습 예시를 제공합니다.

## 기계 학습: 문제 설정

일반적으로, 학습 문제는 n개 [표본(samples)](https://en.wikipedia.org/wiki/Sample_(statistics))의 데이터 집합을 고려하여 알려지지 않은 데이터의 속성을 예측하고자 합니다. 예를 들어, 만약 각 표본이 하나보단 많고 다차원 항목(일명 [다변량(multivariate)](https://en.wikipedia.org/wiki/Multivariate_random_variable) 데이터)일 때, 이를 여러 속성(attributes) 또는 **특성(features)**이 있다고 말합니다.

학습 문제는 몇 가지 범주로 나뉩니다:

- [지도 학습(supervised learning)](https://en.wikipedia.org/wiki/Supervised_learning)에서는, 우리가 예측하고자 하는 추가적인 속성이 데이터에 함께 제공됩니다(사이킷런 지도 학습 페이지로 가시려면 [여기](../../supervised_learning)를 클릭하세요). 이 문제는 다음 중 하나일 수 있습니다:

  - [분류(classification)](https://en.wikipedia.org/wiki/Classification_in_machine_learning): 표본은 둘 또는 그 이상의 클래스에 속하며 우리는 이미 레이블된(labeled) 데이터에서 어떻게 레이블되지 않은(unlabeled) 데이터의 클래스를 예측하는지 배우고 싶어합니다. 분류 문제의 예시로는 손글씨 숫자 인식이 있습니다, 이 때 목표는 각 입력 벡터(vector)를 유한한 숫자의 이산형 범주(discrete categories) 중 하나에 할당하는 것입니다. 분류를 생각해볼 수 있는 또다른 방법으로는 제한된 숫자의 범주가 있고 각각 n개의 표본이 제공되며, 표본들에 올바른 범주나 클래스를 레이블하고자 하는 이산형(연속형(continuous)의 반대) 형태의 지도 학습입니다.
  - [회귀(regression)](https://en.wikipedia.org/wiki/Regression_analysis): 만약 원하는 출력이 하나 이상의 연속형 변수라면, 이 작업은 *회귀*라고 부릅니다. 회긔 문제의 예시로는 연어의 길이를 나이와 무게의 함수로 예측하는 것이 있습니다.

- [비지도 학습](https://en.wikipedia.org/wiki/Unsupervised_learning)은, 훈련 데이터를 구성하는 입력 벡터 x가 대응하는 목표값이 전혀 없는 경우입니다. 이런 문제의 목표는 데이터 안에서 비슷한 예시들의 그룹을 찾는 것, 즉 [군집화(clustering)](https://en.wikipedia.org/wiki/Cluster_analysis)거나, 입력 공간(space) 속에서 데이터의 분포를 결정, 즉 [밀도 추정(density estimation)](https://en.wikipedia.org/wiki/Density_estimation)이거나, 또는 *시각화(visualization)* 목적으로 데이터를 고차원(high-dimensional) 공간에서 이차원 또는 삼차원 공간으로 투영(project)하는 것입니다(사이킷런 비지도 학습 페이지로 가시려면 [여기](../../unsupervised_learning)를 클릭하세요).

**훈련 세트(training set)와 테스트 세트(testing set)**

기계 학습은 데이터 세트의 일부 속성을 학습한 다음 이 속성들을 다른 데이터 세트에 대해 테스트하는 것입니다. 기계 학습의 일반적인 관행은 데이터 세트를 둘로 분할해 알고리즘을 평가하는 것입니다. 우리는 이 세트 중 하나를 **훈련 세트**라 부르는데, 이는 일부 속성을 학습하는 곳입니다; 다른 세트는 **테스트 세트**라 부르며, 학습한 속성을 테스트하는 곳입니다.

## 예시 데이터셋 불러오기

`Scikit-learn`에는 몇 가지 표준적인 데이터셋(datasets)이 함께 들어있는데, 예시로 분류를 위한 [붓꽃(iris)](https://en.wikipedia.org/wiki/Iris_flower_data_set)과 [숫자(digits)](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) 데이터셋과 회귀를 위한 [당뇨병(diabetes)](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) 등이 있습니다.

다음에서는, 우리의 셸(shell)에서 파이썬 인터프리터(Python interpreter)를 시작해 `iris`과 `digits` 데이터셋을 불러옵니다. 우리의 표기 규칙에서는 `$`이 셸 프롬프트(prompt)를 나타내고 `>>>`이 파이썬 인터프리터 프롬프트를 나타냅니다:

```python
$ python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> digits = datasets.load_digits()
```

데이터셋은 모든 데이터와 데이터에 대한 일부 메타데이터(metadata)를 담고 있는 사전과 같은(dictionary-like) 객체입니다. 이 데이터는 `.data` 멤버(member)에 저장되는데, 이는 `n_samples, n_features` 배열입니다. 지도 문제의 경우, 하나 이상의 반응 변수가 `.target` 멤버에 저장됩니다. 다양한 데이터셋에 대한 자세한 내용은 [전용 섹션](../../datasets)에서 찾아볼 수 있습니다.

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

그리고 `digits.target`은 숫자 데이터셋에 대한 실제 정답(ground truth)을 제공하는데, 이는 우리가 배우고자 하는 각 숫자 이미지에 대응하는 숫자입니다:

```python
>>> digits.target
array([0, 1, 2, ..., 8, 9, 8])
```

**데이터 배열의 형태**

데이터는 원본 데이터가 다른 형태를 가졌더라도 항상 2차원(2D) 배열, `(n_samples, n_features)` 형태입니다. 숫자의 경우, 각 원본 표본은 `(8, 8)` 형태의 이미지이며 다음처럼 접근할 수 있습니다:

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

[이 데이터셋의 간단한 예제](../../auto_examples/classification/plot_digits_classification)는 사이킷런에서 어떻게 원래 문제에서 시작해서 사용할만한 데이터의 형태를 만드는지 보여줍니다.

**외부 데이터셋에서 불러오기**

외부 데이터셋에서 불러오려면, [외부 데이터셋에서 불러오기](../../datasets/loading_other_datasets#external-datasets)를 참고해주세요.

## 학습과 예측

숫자 데이터셋의 경우, 작업은 이미지가 주어지고 어떤 숫자가 그 이미지를 나타내는지 예측하는 것입니다. 우리는 10개의 가능한 클래스(숫자 0에서 9까지)가 있는 표본들을 받아서, 본 적 없는 표본들이 속하는 클래스를 *예측(predict)*할 수 있게 [추정기](https://en.wikipedia.org/wiki/Estimator)를 *적합(fit)*하는 것입니다.

사이킷런에서는, 분류를 위한 추정기는 `fit(X, y)`와 `predict(T)`를 구현하는 파이썬 객체입니다.

추정기의 예시는 `sklearn.svm.SVC` 클래스인데, 이는 [서포트 벡터 분류(support vector classification)](https://en.wikipedia.org/wiki/Support_vector_machine)를 구현합니다. 추정기의 생성자(constructor)는 모델의 매개변수를 인자(arguments)로 받습니다.

지금부터, 추정기를 블랙 박스(black box)로 간주하겠습니다:

```python
>>> from sklearn import svm
>>> clf = svm.SVC(gamma=0.001, C=100.)
```

**모델의 매개변수 고르기**

이 예시에서는, `gamma` 값을 수동으로 설정했습니다. 이러한 매개변수의 좋은 값을 찾기 위해서, 우리는 [격자 탐색(grid search)](../../modules/격자_탐색(grid_search))이나 [교차 검증(cross validation)](../../modules/교차_검증(cross_validation))과 같은 좋은 도구를 사용할 수 있습니다.

(분류를 위한) `clf` 추정기 인스턴스(instance)는 먼저 모델에 적합됩니다; 즉, 모델에서 *학습(learn)*해야만 합니다. 이는 우리의 훈련 세트를 `fit` 메서드(method)에 전달함으로써 이뤄집니다. 훈련 세트로, 우리는 우리가 예측하기 위해 남겨둔 마지막 이미지만 제외하고 데이터셋의 모든 이미지를 사용할 것입니다. 우리는 훈련 세트를 `[:-1]` 파이썬 구문으로 선택했는데, 이는 `digits.data`에서 마지막 항목 외에 모든 항목을 포함하는 새로운 배열을 생성합니다:

```python
>>> clf.fit(digits.data[:-1], digits.target[:-1])
SVC(C=100.0, gamma=0.001)
```

이제 여러분은 새 값을 *예측(predict)*할 수 있습니다. 이 경우 `digits.data`의 마지막 이미지를 사용해 예측할 것입니다. 예측을 통해, 여러분은 훈련 세트에서 마지막 이미지와 가장 일치하는 이미지를 결정할 것입니다.

```python
>>> clf.predict(digits.data[-1:])
array([8])
```

해당 이미지는 다음과 같습니다:

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_last_image_001.png)

보시다시피, 어려운 작업입니다: 무엇보다도, 이미지는 해상도가 나쁩니다. 분류기에 동의하시나요?

이 분류 문제의 완전한 예시는 여러분이 실행하고 학습할 수 있는 예제로 제공됩니다: [손글씨 숫자 인식하기](../../auto_examples/classification/plot_digits_classification)

## 컨벤션

사이킷런 추정기는 그들의 행동을 더욱 예측 가능하게 만들기 위해 특정한 규칙을 따릅니다. [공통 용어 및 API 요소 용어 사전](../../glossary)에 더 자세한 내용이 설명되어 있습니다.

### 형 변환(type casting)

별도로 지정되지 않는다면, 입력은 `float64`로 변환됩니다:

```python
>>> import numpy as np
>>> from sklearn import kernel_approximation

>>> rng = np.random.RandomState(0)
>>> X = rng.rand(10, 2000)
>>> X = np.array(X, dtype='float32')
>>> X.dtype
dtype('float32')

>>> transformer = kernel_approximation.RBFSampler()
>>> X_new = transformer.fit_transform(X)
>>> X_new.dtype
dtype('float64')
```

이 예시에서, `X`는 `float32`인데, 이는 `fit_transform(X)`에 의해 `float64`로 변환됩니다.

회귀 목표값은 `float64`로 변환되고 분류 목표값은 유지됩니다:

```python
>>> from sklearn import datasets
>>> from sklearn.svm import SVC
>>> iris = datasets.load_iris()
>>> clf = SVC()
>>> clf.fit(iris.data, iris.target)
SVC()

>>> list(clf.predict(iris.data[:3]))
[0, 0, 0]

>>> clf.fit(iris.data, iris.target_names[iris.target])
SVC()

>>> list(clf.predict(iris.data[:3]))
['setosa', 'setosa', 'setosa']
```

여기서, `iris.target`(정수 배열)이 `fit`에 사용되었기 때문에, 첫 번째 `predict()`는 정수(integer) 배열을 반환합니다. `iris.target_names`가 적합에 사용되었기 때문에, 두 번째 `predict()`는 문자열(string) 배열을 반환합니다.

### 매개변수를 다시 적합하고 갱신하기

추정기의 초매개변수(hyper-parameters)는 생성된 다음에도 [set_params()](../../glossary#set_params) 메서드로 갱신할 수 있습니다. `fit()`을 두 번 이상 호출하면 이전에 모든 `fit()`에서 배운 것들을 덮어씁니다.

```python
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.svm import SVC
>>> X, y = load_iris(return_X_y=True)

>>> clf = SVC()
>>> clf.set_params(kernel='linear').fit(X, y)
SVC(kernel='linear')
>>> clf.predict(X[:5])
array([0, 0, 0, 0, 0])

>>> clf.set_params(kernel='rbf').fit(X, y)
SVC()
>>> clf.predict(X[:5])
array([0, 0, 0, 0, 0])
```

여기에서, 기본 커널(kernel) `rbf`는 추정기가 생성된 다음에 우선 [`SVC.set_params()`](../../modules/generated/sklearn.svm.SVC#sklearn.svm.SVC.set_params)를 통해 `linear`로 바뀌고, 추정기를 다시 적합하고(refit) 두 번째 예측을 하기 위해 `rbf`로 돌아갑니다.

### 다중클래스(multiclass) 대 다중레이블(multilabel) 적합

[`multiclass classifiers`](../../modules/classes#module-sklearn.multiclass)를 사용할 때, 수행되는 학습과 예측 작업은 표적 데이터의 형식에 의존합니다:

```python
>>> from sklearn.svm import SVC
>>> from sklearn.multiclass import OneVsRestClassifier
>>> from sklearn.preprocessing import LabelBinarizer

>>> X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
>>> y = [0, 0, 1, 1, 2]

>>> classif = OneVsRestClassifier(estimator=SVC(random_state=0))
>>> classif.fit(X, y).predict(X)
array([0, 0, 1, 1, 2])
```

위의 경우, 분류기는 다중클래스 레이블의 1차원 배열에 적합하므로 `predict()` 메서드는 이에 대응하는 다중클래스 예측을 제공합니다. 또한 이진(binary) 레이블 지표(indicators)에 적합하는 것도 가능합니다:

```python
>>> y = LabelBinarizer().fit_transform(y)
>>> classif.fit(X, y).predict(X)
array([[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0],
       [0, 0, 0],
       [0, 0, 0]])
```

여기서는, 분류기는 [`LabelBinarizer`](../../modules/generated/sklearn.preprocessing.LabelBinarizer)를 사용해 `y`의 2차원 이진 레이블 표현에 `fit()`합니다. 이 경우 `predict()`는 이에 대응하는 다중레이블 예측을 나타내는 2차원 배열을 반환합니다.

네 번째와 다섯 번째 인스턴스는 모두 0을 반환했음을 주의하세요, 이는 그들이 세 레이블 중 아무것도 `fit`하지 않았음을 나타냅니다. 다중레이블 출력과 함께라면, 비슷하게 인스턴스에게 여러 레이블을 할당하게 할 수 있습니다:

```python
>>> from sklearn.preprocessing import MultiLabelBinarizer
>>> y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
>>> y = MultiLabelBinarizer().fit_transform(y)
>>> classif.fit(X, y).predict(X)
array([[1, 1, 0, 0, 0],
       [1, 0, 1, 0, 0],
       [0, 1, 0, 1, 0],
       [1, 0, 1, 0, 0],
       [1, 0, 1, 0, 0]])
```

이 경우, 분류기는 각각 여러 레이블이 할당된 인스턴스에 적합합니다. [`MultiLabelBinarizer`](../../modules/generated/sklearn.preprocessing.MultiLabelBinarizer)는 `fit`하기 위한 다중레이블의 2차원 배열을 이진화하는데 사용됩니다. 결과적으로, `predict()`는 인스턴스마다 여러 예측 레이블이 있는 2차원 배열을 반환합니다.
