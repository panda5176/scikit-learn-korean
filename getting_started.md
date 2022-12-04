# 시작하기
이 가이드의 목적은 `scikit-learn`이 제공하는 몇 가지 주요 기능을 설명하는 것입니다. 설명은 기계 학습에 대한 매우 기초적인 지식(모델 적합, 예측, 교차검증 등)이 있다고 가정하고 있습니다. `scikit-learn`을 설치하시려면 우리의 [설치 설명서](install)를 참조해주세요.

## 적합과 예측: 추정기 기초
`Scikit-learn`은 [추정기(estimators)](glossary#추정기(estimators))라고 하는, 수십 개의 내장 기계 학습 알고리즘과 모델을 제공합니다. 각 추정기는 [`fit`](glossary#fit) 메서드를 이용해 어떤 데이터에 적합될 수 있습니다.

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

- 샘플 행렬(또는 설계 행렬) [`X`](glossary#X). `X`의 크기는 보통 `(샘플 크기, 피쳐 크기)`입니다, 즉 샘플은 행을 나타내고 피쳐는 열을 나타냅니다.
- 목표값 [`y`](glossary#y)는 회귀 작업을 위한 실수값들이나 분류를 위한 정수값들(또는 어떤 불연속 값들의 집합)입니다. 비지도 학습 작업을 위해서는, `y`를 지정할 필요가 없습니다. `y`는 보통 1차원 배열이며 `i`번째 항목은 `X`의 `i`번째 샘플(행)의 목표값에 대응합니다.

보통 `X`와 `y` 둘다 numpy 배열이나 동등한 [유사-배열](glossary#유사-배열) 데이터 타입일 것으로 예상되며, 어떤 추정기는 희소 행렬과 같은 다른 형식들과 작동할 수도 있습니다.

한 번 추정기가 적합되면, 새로운 데이터의 목표값들을 예측하는데에도 사용될 수 있습니다. 여러분은 다시 추정기를 훈련시킬 필요가 없습니다:

```python
>>> clf.predict(X)  # 훈련 데이터의 클래스를 예측한다
array([0, 1])
>>> clf.predict([[4, 5, 6], [14, 15, 16]])  # 새로운 데이터의 클래스를 예측한다
array([0, 1])
```

