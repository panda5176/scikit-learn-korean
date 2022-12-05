원문: [Getting Started](https://scikit-learn.org/stable/getting_started.html)

# 시작하기

이 가이드의 목적은 `scikit-learn`이 제공하는 몇 가지 주요 기능을 설명하는 것입니다. 이 설명은 기계 학습(machine learning)에 대한 매우 기초적인 지식(모델 적합, 예측, 교차검증 등)이 있다고 가정하고 있습니다. `scikit-learn`을 설치하시려면 우리의 [설치 설명서](install)를 참고해주세요.

## 적합과 예측: 추정기 기초

`Scikit-learn`은 [추정기(estimators)](glossary#추정기(estimators))라고 하는, 수십 개의 내장 기계 학습 알고리즘과 모델을 제공합니다. 각 추정기는 [`fit`](glossary#fit) 메서드를 이용해 어떤 데이터에 적합(fit)될 수 있습니다.

다음은 어떤 매우 기초적인 데이터에 [`RandomForestClassifier`](modules/generated/sklearn.ensemble.RandomForestClassifier#sklearn.ensemble.RandomForestClassifier)를 적합하는 간단한 예시입니다:

```python
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier(random_state=0)
>>> X = [[ 1,  2,  3],  # 2개 표본, 3개 특성
...      [11, 12, 13]]
>>> y = [0, 1]  # 각 표본의 클래스
>>> clf.fit(X, y)
RandomForestClassifier(random_state=0)
```

`fit` 메서드는 보통 2개의 입력을 받습니다:

- 표본 행렬(samples matrix)(또는 설계 행렬) [`X`](glossary#X). `X`의 크기는 보통 `(표본 수, 특성 수)`입니다, 즉 표본(samples)은 행을 나타내고 특성(features)은 열을 나타냅니다.
- 목표값(target values) [`y`](glossary#y)는 회귀(regression) 작업을 위한 실수값들(real numbers)이나 분류(classification)를 위한 정수값(integers)들(또는 어떤 불연속(discrete) 값들의 집합)입니다. 비지도 학습(unsupervised learning) 작업을 위해서는, `y`를 지정할 필요가 없습니다. `y`는 보통 1차원 배열(1d array)이며 `i`번째 항목은 `X`의 `i`번째 표본(행)의 목표값에 대응합니다.

보통 `X`와 `y` 둘다 numpy 배열이나 동등한 [유사-배열](glossary#유사-배열) 데이터 타입일 것으로 예상되며, 어떤 추정기는 희소 행렬(sparse matrix)과 같은 다른 형식들과 함께 작동할 수도 있습니다.

한 번 추정기가 적합되면, 새로운 데이터의 목표값들을 예측하는데에도 사용될 수 있습니다. 여러분은 다시 추정기를 훈련시킬 필요가 없습니다:

```python
>>> clf.predict(X)  # 훈련 데이터의 클래스를 예측합니다
array([0, 1])
>>> clf.predict([[4, 5, 6], [14, 15, 16]])  # 새로운 데이터의 클래스를 예측합니다
array([0, 1])
```

## 변환기와 전처리기

기계 학습 워크플로(workflows)는 종종 서로 다른 여러 부분들로 구성됩니다. 일반적인 파이프라인은 데이터를 변환하거나 대치하는 전처리 단계와, 목표값을 예측하는 최종 예측기로 구성됩니다.

`Scikit-learn`에서는, 전처리기(pre-processors)와 변환기(transformers)는 추정기 객체와 동일한 API를 따릅니다(실제로 모두 같은 `BaseEstimator` 클래스를 상속합니다). 변환기 객체는 [`predict`](glossary#predict) 메서드가 아니라 새롭게 변환된 표본 행렬 `X`를 출력하는 [`transform`](glossary#transform) 메서드를 가집니다:

```python
>>> from sklearn.preprocessing import StandardScaler
>>> X = [[0, 15],
...      [1, -10]]
>>> # 데이터를 스케일 계산된 값에 따라 스케일링합니다
>>> StandardScaler().fit(X).transform(X)
array([[-1.,  1.],
       [ 1., -1.]])
```

종종, 다른 특성에는 다른 변환을 적용하고 싶을 때가 있습니다: [ColumnTransformer](modules/compose#이종-데이터를-위한-ColumnTransformer)가 이런 사용 예시를 위해 설계되었습니다.

## 파이프라인: 전처리기와 추정기를 연결하기

변환기와 추정기(예측기)는 단일한 통합 객체로 결합할 수 있습니다: 이를 [파이프라인(pipeline)](modules/generated/sklearn.pipeline.Pipeline#sklearn.pipeline.Pipeline)이라 합니다. 파이프라인은 일반 추정기와 동일한 API를 제공합니다: 즉 `fit`과 `predict`로 적합되거나 예측에 사용될 수 있습니다. 나중에 살펴보겠지만, 파이프라인을 사용하는 것은 데이터 유출로부터 여러분을 지켜줄 것입니다, 다시 말해 훈련(training) 데이터에서 일부 테스트(test) 데이터를 공개하는 것입니다.

다음 예시에서, 우리는 붓꽃 데이터셋을 로드(load)하고, 훈련과 테스트 세트로 분할한 다음, 테스트 데이터에서 파이프라인의 정확도(accuracy) 점수를 계산합니다:

```python
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
...
>>> # 파이프라인 객체를 생성합니다
>>> pipe = make_pipeline(
...     StandardScaler(),
...     LogisticRegression()
... )
...
>>> # 붓꽃 데이터셋을 로드하고 훈련과 테스트 세트로 분할합니다
>>> X, y = load_iris(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
...
>>> # 전체 파이프라인에 적합합니다
>>> pipe.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression', LogisticRegression())])
>>> # 이제 다른 추정기처럼 사용할 수 있습니다
>>> accuracy_score(pipe.predict(X_test), y_test)
0.97...
```

## 모델 평가

모델을 어떤 데이터에 적합하는 것이 본 적 없는 데이터도 잘 예측하게끔 하는 것은 아닙니다. 이건 직접 평가되어야 합니다. 방금 [`train_test_split`](modules/generated/sklearn.model_selection.train_test_split#sklearn.model_selection.train_test_split) 도우미(helper)가 데이터셋을 훈련과 테스트 세트로 분할하는 것을 보았지만, `scikit-learn`은 모델 평가를 위한, 특히 [교차검증(cross-validation)](modules/cross_validation#교차검증(cross-validation))을 위한 많은 다른 도구들을 제공합니다.

여기 [`cross_validate`](modules/generated/sklearn.model_selection.cross_validate#sklearn.model_selection.cross_validate) 도우미가 어떻게 5겹(5-fold) 교차검증 절차를 수행하는지 간략하게 보여드립니다. 수동으로 겹을 반복하고, 다른 데이터 분할 전략을 사용하고, 사용자가 직접 정의한 점수를 매기는 함수(custom scoring function)를 사용할 수도 있습니다. 자세한 내용은 우리의 [사용자 안내서(User Guide)](modules/cross_validation#교차검증(cross-validation))를 참고하세요:

```python
>>> from sklearn.datasets import make_regression
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.model_selection import cross_validate
...
>>> X, y = make_regression(n_samples=1000, random_state=0)
>>> lr = LinearRegression()
...
>>> result = cross_validate(lr, X, y)  # 기본 설정은 5겹 교차검증입니다
>>> result['test_score']  # 데이터셋이 쉽기 때문에 결정계수가 높습니다
array([1., 1., 1., 1., 1.])
```

## 자동 매개변수 탐색

모든 추정기는 조정할 수 있는 매개변수(parameter)를 가집니다(문헌에서 종종 초매개변수(hyper-parameter)라고도 합니다). 추정기의 일반화 능력(generalization power)은 종종 적은 매개변수에 의해 치명적으로 좌우됩니다. 예를 들어 [`RandomForestRegressor`](modules/generated/sklearn.ensemble.RandomForestRegressor#sklearn.ensemble.RandomForestRegressor)는 숲(forest)에서 나무(trees)의 수를 결정하는 `n_estimators` 매개변수를 가지고, 각 나무의 최대 깊이를 결정하는 `max_depth` 매개변수도 가집니다. 꽤나 자주, 이런 매개변수의 정확한 값은 당면한 데이터에 의존하기 때문에 명확하지가 않습니다.

`Scikit-learn`은 자동으로 (교차검증을 통해) 자동으로 최고의 매개변수 조합을 찾는 도구를 제공합니다. 다음 예제에서는, [`RandomizedSearchCV`](modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) 객체와 함께 랜덤 포레스트(random forest) 매개변수 공간을 무작위로 탐색합니다. 탐색이 끝나면, [`RandomizedSearchCV`](modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)는 최고의 매개변수 세트로 적합된 [`RandomForestRegressor`](modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)처럼 작동합니다. [사용자 안내서](modules/grid_search.html#격자-탐색(grid-search))를 더 읽어보세요:

```python
>>> from sklearn.datasets import fetch_california_housing
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.model_selection import RandomizedSearchCV
>>> from sklearn.model_selection import train_test_split
>>> from scipy.stats import randint
...
>>> X, y = fetch_california_housing(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
...
>>> # 탐색될 매개변수 공간을 정의합니다
>>> param_distributions = {'n_estimators': randint(1, 5),
...                        'max_depth': randint(5, 10)}
...
>>> # 이제 새로운 searchCV 객체를 만들고 데이터에 적합합니다
>>> search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
...                             n_iter=5,
...                             param_distributions=param_distributions,
...                             random_state=0)
>>> search.fit(X_train, y_train)
RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,
                   param_distributions={'max_depth': ...,
                                        'n_estimators': ...},
                   random_state=0)
>>> search.best_params_
{'max_depth': 9, 'n_estimators': 4}

>>> # 탐색 객체는 이제 일반적인 랜덤 포레스트 추정기처럼 작동합니다
>>> # max_depth=9이고 n_estimators=4인 추정기처럼요
>>> search.score(X_test, y_test)
0.73...
```

> **메모:** 실제로는, 단일 추정기 대신, 거의 항상 [파이프라인 전부를 탐색](modules/grid_search.html#복합-격자-탐색(composite-grid-search))하고싶어 합니다. 주된 이유 중 하나로 만약에 여러분이 파이프라인을 사용하지 않고 전처리 단계를 전체 데이터셋에 적용한 다음 어떤 종류든 교차검증을 수행한다면, 훈련과 테스트 데이터 사이의 기본 독립성 가정(assumption of independence)이 깨지기 때문입니다. 실제로, 여러분이 전체 데이터셋을 이용해 전처리했다면, 테스트 세트에 대한 일부 정보는 훈련 세트에서도 사용 가능합니다. 이건 추정기의 일반화 능력을 과대평가하게 됩니다([Kaggle 게시물](https://www.kaggle.com/alexisbcook/data-leakage)에서 더 자세한 내용을 읽을 수 있습니다). 교차검증과 탐색에 파이프라인을 사용하는 것은 여러분을 이런 흔한 함정에 빠지지 않도록 해줄 것입니다.

## 다음 단계

추정기 적합과 예측, 전처리 단계, 파이프라인, 교차검증 도구와 자동 초매개변수 탐색을 간략하게 다루어보았습니다. 이 안내서는 라이브러리의 일부 주요 기능들에 대한 개요를 제공하긴 하지만, `scikit-learn`에는 더 많은 것들이 있습니다!

우리가 제공하는 모든 도구에 대한 자세한 설명은 [사용자 안내서](user_guide)를 참고하세요. [API 레퍼런스](modules/classes#API-레퍼런스)에서는 공개 API의 완전한 목록을 찾아볼 수 있습니다.

또 다양한 상황에서 `scikit-learn`의 사용 방식을 묘사한 방대한 [예시들](auto_examples/index#일반적인-예시)도 있습니다.

[튜토리얼](tutorial/index)도 추가적인 공부할 거리를 담고 있습니다.
