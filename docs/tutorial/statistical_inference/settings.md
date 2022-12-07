원문: [Statistical learning: the setting and the estimator object in scikit-learn](https://scikit-learn.org/stable/tutorial/statistical_inference/settings.html)

# 통계적 학습: 사이킷런에서의 설정과 추정기 객체

## 데이터셋

사이킷런은 하나 이상의 데이터셋에서 온 2차원 배열(2D array)로 표현되는 학습 정보를 다룹니다. 이 정보들은 다차원(multi-dimensional) 관측 목록으로 이해될 수 있습니다. 이 배열들의 첫 번째 축(axis)을 **표본(samples)** 축이라 부르고, 두 번째 축은 **특성(features)** 축이라고 부릅니다.

**사이킷런에 실린 간단한 예시: 붓꽃(iris) 데이터셋**

```python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> data = iris.data
>>> data.shape
(150, 4)
```

이건 붓꽃에 대한 150개의 관측결과로 만들어져있으며, 각각 4개의 특성으로 설명됩니다: 꽃받침(sepal)과 꽃잎(petal)의 길이(length)와 너비(width)이며, `iris.DESCR`에 자세한 내용이 있습니다.

데이터가 처음에 `(n_samples, n_features)` 형태가 아니라면, 사이킷런을 이용할 수 있도록 전처리되어야(preprocessed) 합니다.

**데이터 재구조화(reshaping)의 예시는 숫자(digits) 데이터셋입니다**

숫자 데이터셋은 1797개의 8x8 손글씨 숫자 이미지로 만들어져있습니다

```python
>>> digits = datasets.load_digits()
>>> digits.images.shape
(1797, 8, 8)
>>> import matplotlib.pyplot as plt
>>> plt.imshow(digits.images[-1],
...            cmap=plt.cm.gray_r)
<...>
```

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_last_image_001.png)

이 데이터셋을 사이킷런과 함께 사용하기 위해, 각 8x8 이미지를 64 길이의 특성 벡터(feature vector)로 변환(transform)합니다

```python
>>> data = digits.images.reshape(
...     (digits.images.shape[0], -1)
... )
```

## 추정기 객체

**데이터 적합**: 사이킷런에서 구현된 주요 API는 `estimator(추정기)` API입니다. 추정기는 데이터로 학습하는 어떤 객체입니다; 분류(classification), 회귀(regression) 또는 군집화(clustering) 알고리즘 또는 원시 데이터(raw data)에서 유용한 특성을 추출/필터하는 *변환기(transformer)*일 수도 있습니다.

모든 추정기 객체는 (보통 2차원 배열인) 데이터셋을 받는 `fit` 메서드(method)를 노출합니다:

```python
>>> estimator.fit(data)
```

**추정기 매개변수**: 추정기의 모든 매개변수(parameters)는 인스턴스화(instantiated)할 때나 해당 속성(attribute)을 수정함으로써 설정할 수 있습니다:

```python
>>> estimator = Estimator(param1=1, param2=2)
>>> estimator.param1
1
```

**추정된 매개변수**: 데이터가 추정기에 적합되면, 매개변수는 들고 있는 데이터로부터 추정됩니다. 모든 추정된 매개변수는 이름이 밑줄(underscore)로 끝나는 추정기 객체의 속성입니다:

```python
>>> estimator.estimated_param_ 
```
