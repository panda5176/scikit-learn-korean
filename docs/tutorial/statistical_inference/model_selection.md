원문: [Model selection: choosing estimators and their parameters](https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html)

# 모델 선택: 추정기와 매개변수 선택하기

## 점수, 그리고 교차 검증된 점수

우리가 보아왔듯이, 모든 추정기는 새로운 데이터에 대한 적합(또는 예측)의 품질을 판단할 수 있는 `score` 메서드(method)를 노출합니다. **클수록 좋습니다.**

```python
>>> from sklearn import datasets, svm
>>> X_digits, y_digits = datasets.load_digits(return_X_y=True)
>>> svc = svm.SVC(C=1, kernel='linear')
>>> svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
0.98
```

(모델의 적합도에 대한 프록시(proxy)로 사용할 수 있는) 더 좋은 예측 정확도(accuracy)를 얻기 위해서는, 훈련과 테스트를 위한 데이터를 *겹(folds)*으로 훌륭하게 분할할 수 있습니다.

```python
>>> import numpy as np
>>> X_folds = np.array_split(X_digits, 3)
>>> y_folds = np.array_split(y_digits, 3)
>>> scores = list()
>>> for k in range(3):
...     # 나중에 `pop`하기 위해서 `list`를 사용해 복사합니다
...     X_train = list(X_folds)
...     X_test = X_train.pop(k)
...     X_train = np.concatenate(X_train)
...     y_train = list(y_folds)
...     y_test = y_train.pop(k)
...     y_train = np.concatenate(y_train)
...     scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
>>> print(scores)
[0.934..., 0.956..., 0.939...]
```

이를 [`KFold`(K겹)](../../modules/generated/sklearn.model_selection.KFold) 교차 검증(cross-validation)이라 합니다.

## 교차 검증 생성기

사이킷런에는 유명한 교차 검증 전략을 위한 훈련/테스트 인덱스(indices) 목록을 만드는 데 사용할 수 있는 클래스 모음이 있습니다.

그것들은 분할하기 위한 입력 데이터셋을 받아서, 선택한 교차 검증 전략의 각 반복(iteration)에 대한 훈련/테스트 세트 인덱스를 생성하는 `split` 메서드를 노출합니다.

이 예시는 `split` 메서드의 사용례를 보여줍니다.

```python
>>> from sklearn.model_selection import KFold, cross_val_score
>>> X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
>>> k_fold = KFold(n_splits=5)
>>> for train_indices, test_indices in k_fold.split(X):
...      print('Train: %s | test: %s' % (train_indices, test_indices))
Train: [2 3 4 5 6 7 8 9] | test: [0 1]
Train: [0 1 4 5 6 7 8 9] | test: [2 3]
Train: [0 1 2 3 6 7 8 9] | test: [4 5]
Train: [0 1 2 3 4 5 8 9] | test: [6 7]
Train: [0 1 2 3 4 5 6 7] | test: [8 9]
```

그 다음 교차 검증은 더 쉽게 수행할 수 있습니다.

```python
>>> [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
...  for train, test in k_fold.split(X_digits)]
[0.963..., 0.922..., 0.963..., 0.963..., 0.930...]
```

교차 검증 점수는 [`cross_val_score`](../../modules/generated/sklearn.model_selection.cross_val_score) 도우미(helper)로 바로 계산할 수 있습니다. 추정기와 교차 검증 객체, 그리고 입력 데이터셋이 주어지면, [`cross_val_score`](../../modules/generated/sklearn.model_selection.cross_val_score)는 데이터를 반복적으로 훈련과 테스트 세트로 분할하고, 훈련 세트로 추정기를 훈련시키며, 교차 검증 각 반복에서의 테스트 세트에 기반한 점수를 계산합니다.

기본적으로 추정기의 `score` 메서드는 개별 점수를 계산하기 위해 사용합니다.

사용 가능한 채점 메서드들을 학습하려면 [측정 모듈(metrics module)](../modules/metrics)을 참고하세요.

```python
>>> cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
array([0.96388889, 0.92222222, 0.9637883 , 0.9637883 , 0.93036212])
```

`n_jobs=-1`은 계산이 컴퓨터의 모든 CPU에 디스패치(dispatch)됨을 의미합니다.

대안으로는, 대체 채점 메서드를 지정하기 위해 `scoring` 인자(argument)를 제공할 수 있습니다.

```python
>>> cross_val_score(svc, X_digits, y_digits, cv=k_fold,
...                 scoring='precision_macro')
array([0.96578289, 0.92708922, 0.96681476, 0.96362897, 0.93192644])
```

**교차 검증 생성기(generators)**

|[`KFold`](../../modules/generated/sklearn.model_selection.KFold) (n_splits, shuffle, random_state)|[`StratifiedKFold`](../../modules/generated/sklearn.model_selection.StratifiedKFold) (n_splits, shuffle, random_state)|[`GroupKFold`](../../modules/generated/sklearn.model_selection.GroupKFold) (n_splits)|
|---|---|---|
|K겹으로 분할하고, K-1개로 훈련하여 제외한 하나로 테스트합니다.|K겹과 같지만 각 겹 안에서 클래스 분포를 보존합니다.|같은 그룹이 테스트와 훈련 모두에 있지 않도록 보장합니다.|

|[`ShuffleSplit`](../../modules/generated/sklearn.model_selection.ShuffleSplit) (n_splits, test_size, train_size, random_state)|[`StratifiedShuffleSplit`](../../modules/generated/sklearn.model_selection.StratifiedShuffleSplit)|[`GroupShuffleSplit`](../../modules/generated/sklearn.model_selection.GroupShuffleSplit)|
|---|---|---|
|훈련/테스트 인덱스를 무작위 순열에 기반해 생성합니다.|셔플 분할(shuffle split)과 같지만 각 반복 안에서 클래스 분포를 보존합니다.|같은 그룹이 테스트와 훈련 모두에 있지 않도록 보장합니다.|

|[`LeaveOneGroupOut`](../../modules/generated/sklearn.model_selection.LeaveOneGroupOut) ()|[`LeavePGroupsOut`](../../modules/generated/sklearn.model_selection.LeavePGroupsOut) (n_groups)|[`LeaveOneOut`](../../modules/generated/sklearn.model_selection.LeaveOneOut) ()|
|---|---|---|
|그룹 관측을 위해 그룹 배열을 사용합니다.|P개 그룹을 제외합니다.|한 관측을 제외합니다.|

|[`LeaveOut`](../../modules/generated/sklearn.model_selection.LeavePOut) (p)|[`PredefinedSplit`](../../modules/generated/sklearn.model_selection.PredefinedSplit)|
|---|---|
|P개 관측을 제외합니다.|훈련/테스트 인덱스를 미리 정의된 분할을 기반으로 생성합니다.|

**연습**

숫자(digits) 데이터셋에서, [`SVC`](../../modules/generated/sklearn.svm.SVC) 추정기가 매개변수 `C`의 함수로 된 선형 커널(linear kernel)이 있을 때 교차-검증 점수 플롯(plot)을 그리세요(1부터 10까지의 로그 격자점(logarithmic grid of points)를 사용하세요).

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

X, y = datasets.load_digits(return_X_y=True)

svc = svm.SVC(kernel="linear")
C_s = np.logspace(-10, 0, 10)

scores = list()
scores_std = list()
```

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_digits_001.png)

**정답**: [숫자 데이터셋 교차 검증 연습](../../auto_examples/exercises/plot_cv_digits)

## 격자 탐색과 교차 검증된 추정기

### 격자 탐색(grid-search)

사이킷런은 데이터가 주어지면, 매개변수 격자에서 추정기가 적합하는 동안 점수를 계산하고 교차 검증 점수를 최대화하는 매개변수를 선택하게끔 하는 객체를 제공합니다. 이 객체는 생성될 때 추정기를 받아서 추정기 API를 노출합니다:

```python
>>> from sklearn.model_selection import GridSearchCV, cross_val_score
>>> Cs = np.logspace(-6, -1, 10)
>>> clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
...                    n_jobs=-1)
>>> clf.fit(X_digits[:1000], y_digits[:1000])
GridSearchCV(cv=None,...
>>> clf.best_score_
0.925...
>>> clf.best_estimator_.C
0.0077...

>>> # 테스트 세트에서 예측 성능은 훈련 세트에서의 성능만큼 좋지 않습니다
>>> clf.score(X_digits[1000:], y_digits[1000:])
0.943...
```

기본적으로, [`GridSearchCV`](../../modules/generated/sklearn.model_selection.GridSearchCV)는 5겹 교차 검증을 사용합니다. 하지만, 회귀자(regressor)가 아니라 분류기가 전달되었음을 인식하면, 계층적(stratified) 5겹을 사용합니다.

**중첩 교차 검증(nested cross-validation)**

```python
>>> cross_val_score(clf, X_digits, y_digits) 
array([0.938..., 0.963..., 0.944...])
```

두 교차 검증 반복은 병렬(parallel)로 수행됩니다: 하나는 [`GridSearchCV`](../../modules/generated/sklearn.model_selection.GridSearchCV) 추정기를 `gamma`를 설정하기 위해 수행되고 다른 하나는 `cross_val_score`를 추정기의 예측 성능을 측정하게 하기 위해 수행됩니다. 결과 점수는 새로운 데이터에 대한 불편추정량(unbiased estimates)입니다.

> **경고:** 여러분은 (`n_jobs`가 1과 다른) 병렬 컴퓨팅으로 객체를 중첩시킬 수 없습니다.

### 교차 검증 추정기

매개변수를 설정하기 위한 교차 검증은 알고리즘에 따라서 보다 효율적으로 수행할 수 있습니다. 왜냐하면, 어떤 추정기의 경우, 사이킷런은 매개변수를 교차 검증을 통해 자동적으로 설정할 수 있도록 [교차 검증: 추정기 성능 평가](../../modules/cross_validation) 추정기를 노출하기 때문입니다.

```python
>>> from sklearn import linear_model, datasets
>>> lasso = linear_model.LassoCV()
>>> X_diabetes, y_diabetes = datasets.load_diabetes(return_X_y=True)
>>> lasso.fit(X_diabetes, y_diabetes)
LassoCV()
>>> # 추정기가 자동으로 람다(lambda)를 골랐습니다:
>>> lasso.alpha_
0.00375...
```

이러한 추정기들은 그들의 대조본(counterparts)과 비슷하게 불리되, 'CV'가 이름에 더해집니다.

**연습**

당뇨병 데이터셋에서, 최적의 정규화 매개변서 알파(alpha)를 찾으세요.

**보너스**: 알파 선택을 얼마나 신뢰할 수 있습니까?

```python
import numpy as np

from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

X, y = datasets.load_diabetes(return_X_y=True)
X = X[:150]
```

**정답:** [당뇨병 데이터셋 교차 검증 연습](../../auto_examples/exercises/plot_cv_diabetes)
