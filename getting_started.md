# 시작하기

이 가이드의 목적은 `scikit-learn`이 제공하는 몇 가지 주요 기능을 설명하는 것입니다. 설명은 기계 학습(machine learning)에 대한 매우 기초적인 지식(모델 적합, 예측, 교차검증 등)이 있다고 가정하고 있습니다. `scikit-learn`을 설치하시려면 우리의 [설치 설명서](install)를 참조해주세요.

## 적합과 예측: 추정기 기초

`Scikit-learn`은 [추정기(estimators)](glossary#추정기(estimators))라고 하는, 수십 개의 내장 기계 학습 알고리즘과 모델을 제공합니다. 각 추정기는 [`fit`](glossary#fit) 메서드를 이용해 어떤 데이터에 적합(fit)될 수 있습니다.

다음은 어떤 매우 기초적인 데이터에 [`RandomForestClassifier`](modules/generated/sklearn.ensemble.RandomForestClassifier#sklearn.ensemble.RandomForestClassifier)를 적합하는 간단한 예시입니다:

```python
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier(random_state=0)
>>> X = [[ 1,  2,  3],  # 2 샘플, 3 피쳐
...      [11, 12, 13]]
>>> y = [0, 1]  # 각 샘플의 클래스
>>> clf.fit(X, y)
RandomForestClassifier(random_state=0)
```

`fit` 메서드는 보통 2개의 입력을 받습니다:

- 샘플 행렬(samples matrix)(또는 설계 행렬) [`X`](glossary#X). `X`의 크기는 보통 `(샘플 크기, 피쳐 크기)`입니다, 즉 샘플(samples)은 행을 나타내고 피쳐(features)는 열을 나타냅니다.
- 목표값(target values) [`y`](glossary#y)는 회귀(regression) 작업을 위한 실수값들이나 분류(classification)를 위한 정수값들(또는 어떤 불연속 값들의 집합)입니다. 비지도 학습(unsupervised learning) 작업을 위해서는, `y`를 지정할 필요가 없습니다. `y`는 보통 1차원 배열이며 `i`번째 항목은 `X`의 `i`번째 샘플(행)의 목표값에 대응합니다.

보통 `X`와 `y` 둘다 numpy 배열이나 동등한 [유사-배열](glossary#유사-배열) 데이터 타입일 것으로 예상되며, 어떤 추정기는 희소 행렬과 같은 다른 형식들과 함께 작동할 수도 있습니다.

한 번 추정기가 적합되면, 새로운 데이터의 목표값들을 예측하는데에도 사용될 수 있습니다. 여러분은 다시 추정기를 훈련시킬 필요가 없습니다:

```python
>>> clf.predict(X)  # 훈련 데이터의 클래스를 예측합니다
array([0, 1])
>>> clf.predict([[4, 5, 6], [14, 15, 16]])  # 새로운 데이터의 클래스를 예측합니다
array([0, 1])
```

## 변환기와 전처리기

기계 학습 워크플로(workflows)는 종종 서로 다른 부분들로 구성됩니다. 일반적인 파이프라인은 데이터를 변환하거나 대치하는 전처리 단계와, 목표값을 예측하는 최종 예측기로 구성됩니다.

`Scikit-learn`에서는, 전처리기(pre-processors)와 변환기(transformers)는 추정기 객체와 동일한 API를 따릅니다(실제로 모두 같은 `BaseEstimator` 클래스를 상속합니다). 변환기 객체는 [`predict`](glossary#predict) 메서드가 아니라 새롭게 변환된 샘플 행렬 `X`를 출력하는 [`transform`](glossary#transform) 메서드를 가집니다:

```python
>>> from sklearn.preprocessing import StandardScaler
>>> X = [[0, 15],
...      [1, -10]]
>>> # 데이터를 스케일 계산된 값에 따라 스케일링합니다
>>> StandardScaler().fit(X).transform(X)
array([[-1.,  1.],
       [ 1., -1.]])
```

종종, 다른 피쳐에는 다른 변환을 적용하고 싶을 때가 있습니다: [ColumnTransformer](modules/compose#이종-데이터를-위한-ColumnTransformer)가 이런 사용례를 위해 설계되었습니다.

## 파이프라인: 전처리기와 추정기를 연결하기

변환기와 추정기(예측기)는 단일한 통합 객체로 함께 결합될 수 있습니다: 이를 [`파이프라인(pipeline)`](modules/generated/sklearn.pipeline.Pipeline#sklearn.pipeline.Pipeline)이라 합니다. 파이프라인은 일반 추정기와 동일한 API를 제공합니다: 즉 `fit`과 `predict`로 적합되거나 예측에 사용될 수 있습니다. 나중에 살펴보겠지만, 파이프라인을 사용하는 것은 데이터 유출로부터 여러분을 지켜줄 것입니다, 다시 말해 훈련(training) 데이터에서 일부 테스트(test) 데이터를 공개하는 것입니다.

다음 예시에서, 우리는 붓꽃 데이터 세트를 로드(load)하고, 훈련과 테스트 세트로 분할한 다음, 테스트 데이터에서 파이프라인의 정확도(accuracy) 점수를 계산합니다.

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
>>> # 붓꽃 데이터 세트를 로드하고 훈련과 테스트 세트로 분할합니다
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

