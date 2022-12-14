원문: [Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

# 텍스트 데이터 작업

이 안내서의 목표는 하나의 실용적인 작업: 스무 가지 다른 주제(topics)로 텍스트 문서(뉴스그룹 게시물(newsgroups posts)) 모음집을 분석하는 것을 통해 몇 가지 주요 `scikit-learn`(사이킷런) 도구를 탐색하는 것입니다.

이 섹션에서 다음 방법들을 살펴볼 것입니다:

- 파일 내용과 범주(categories)를 불러옵니다
- 기계 학습(machine learning)에 적합한 특성 벡터(feature vectors)를 추출합니다
- 범주화(categorization)를 수행하기 위한 선형 모델(linear model)을 훈련합니다
- 특성 추출 구성요소(feature extraction components)와 분류기(classifier) 모두에게 좋은 구성(configuration)을 찾기 위해 격자 탐색(grid search) 전략을 사용합니다

## 튜토리얼 설정

이 튜토리얼을 시작하기 위해, 여러분은 먼저 *사이킷런*과 그의 모든 의존 항목들(dependencies)을 설치하셔야 합니다.

더 많은 정보와 시스템별 설명은 [설치 설명서](../../install) 페이지를 참조하세요.

이 튜토리얼의 소스는 여러분의 사이킷런 폴더에서 찾아보실 수 있습니다.

```sh
scikit-learn/doc/tutorial/text_analytics/
```

소스는 [Github에서](https://github.com/scikit-learn/scikit-learn/tree/main/doc/tutorial/text_analytics)도 찾아보실 수 있습니다.

튜토리얼 폴더는 다음의 하위 폴더들을 포함해야 합니다:

- `*.rst 파일들` - 스핑크스(sphinx)로 쓰여진 튜토리얼 문서 소스
- `data` - 튜토리얼 동안 데이터셋을 넣을 폴더
- `skeletons` - 연습 문제를 위한 견본 미완성 스크립트
- `solutions` - 연습 문제 정답지

여러분들이 하드 드라이브 어딘가 `sklearn_tut_workspace`라는 새로운 폴더에 뼈대구조(skeletons)를 복사해서 연습 문제를 위한 파일을 직접 수정하고, 원본 뼈대구조는 그대로 유지할 수도 있습니다.

```sh
$ cp -r skeletons work_directory/sklearn_tut_workspace
```

기계 학습 알고리즘은 데이터가 필요합니다. 각 `$TUTORIAL_HOME/data` 하위 폴더에 가서 거기서 `fetch_data.py` 스크립트를 실행하세요(일단 읽은 다음).

예를 들어:

```sh
cd $TUTORIAL_HOME/data/languages
less fetch_data.py
python fetch_data.py
```

## 20 가지 뉴스그룹 데이터셋 불러오기

데이터셋은 "스무 가지 뉴스그룹(Twenty Newsgroups)"이라 불립니다. 여기 [웹사이트](http://people.csail.mit.edu/jrennie/20Newsgroups/)에서 인용한, 공식 설명입니다:

20 가지 뉴스그룹 데이터 세트는 20 가지 다른 뉴스그룹으로 (거의) 균등하게 분할한 약 20,000 뉴스그룹 문서의 모음집입니다. 우리가 아는 한, 이는 원래 Ken Lang이 아마도 그의 논문 "Newsweeder: Learning to filter netnews"을 위해 수집했습니다만, 그는 명확하게 이 모음집을 언급하지는 않았습니다. 20 가지 뉴스그룹 모음집은 텍스트 분류나 텍스트 군집화(clustering) 등, 기계 학습 기술의 텍스트 응용 분야 실험을 위한 인기 있는 데이터 세트가 되었습니다.

다음에서 사이킷런의 20 가지 뉴스그룹을 위해 내장 데이터셋 로더(dataset loader)를 사용할 것입니다. 다른 방법으로는, 데이터셋을 직접 웹사이트에서 다운로드하고 [`sklearn.datasets.load_files`](../../modules/generated/sklearn.datasets.load_files) 함수를 사용해 압축되지 않은 보관(archive) 폴더의 `20news-bydate-train` 하위 폴더를 가리키는 것도 가능합니다.

이 첫 예제의 더 빠른 실행 시간을 위해, 우리는 데이터셋에서 사용 가능한 20 가지 중 단 4 가지 범주만으로 된 부분적인 데이터셋 위에서 작업할 것입니다.

```python
>>> categories = ['alt.atheism', 'soc.religion.christian',
...               'comp.graphics', 'sci.med']
```

이제 다음과 같이 이 범주들과 일치하는 파일 목록을 불러올 수 있습니다:

```python
>>> from sklearn.datasets import fetch_20newsgroups
>>> twenty_train = fetch_20newsgroups(subset='train',
...     categories=categories, shuffle=True, random_state=42)
```

반환된 데이터셋은 `scikit-learn` "묶음(bunch)"입니다: 파이썬(python) `dict` 키(keys)나 `object` 속성(attributes)으로 접근하는 필드(fields)가 있는 간단한 홀더 객체(holder object)로, 예를 들어 `target_names`는 요청된 범주 이름의 목록을 가지고 있습니다:

```python
>>> twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
```

파일 자체는 `data` 속성에서 메모리에 불러와집니다. 참고로 파일 이름도 사용할 수 있습니다:

```python
>>> len(twenty_train.data)
2257
>>> len(twenty_train.filenames)
2257
```

첫 번째로 불러온 파일의 처음 줄들을 출력해봅시다:

```python
>>> print("\n".join(twenty_train.data[0].split("\n")[:3]))
From: sd345@city.ac.uk (Michael Collier)
Subject: Converting images to HP LaserJet III?
Nntp-Posting-Host: hampton

>>> print(twenty_train.target_names[twenty_train.target[0]])
comp.graphics
```

지도 학습(supervised learning) 알고리즘은 훈련 세트의 각 문서를 위한 범주 레이블(label)을 필요로 할 것입니다. 이 경우 범주는 각 문서를 담고 있는 폴더의 이름이기도 한, 뉴스그룹의 이름입니다.

속도와 공간 효율성을 위해 `scikit-learn`은 목표 속성을 `target_names` 목록의 범주 이름 인덱스(index)에 해당하는 정수(integers) 배열로 불러옵니다. 각 표본의 범주 정수 ID는 `target` 속성에 저장됩니다:

```python
>>> twenty_train.target[:10]
array([1, 1, 3, 3, 3, 3, 3, 2, 2, 2])
```

다음과 같이 범주 이름을 가져오는 것도 가능합니다:

```python
>>> for t in twenty_train.target[:10]:
...     print(twenty_train.target_names[t])
...
comp.graphics
comp.graphics
soc.religion.christian
soc.religion.christian
soc.religion.christian
soc.religion.christian
soc.religion.christian
sci.med
sci.med
sci.med
```

여러분은 우리가 `fetch_20newsgroups(..., shuffle=True, random_state=42)`를 호출했을 때 표본들이 무작위로 섞였음을 알아차리셨을 겁니다: 이는 여러분이 나중에 완성된 데이터셋으로 재훈련(re-training)하기 전에, 표본의 일부만 선택하여 빠르게 모델을 훈련시키고 결과로부터 첫 아이디어를 얻고자 할 때 유용합니다.

## 텍스트 파일에서 특성 추출

텍스트 문서에서 기계 학습을 수행하려면, 우선 텍스트 내용을 수치 특성 벡터(numerical feature vectors)로 바꿔야 합니다.

### 단어 가방

그러기 위한 가장 직관적인 방법은 단어 가방(a bags of words) 표현을 사용하는 것입니다:

1. 훈련 세트 모든 문서에서 나타나는 단어에 각각 고정된 정수 ID를 할당합니다(예를 들어 단어에서 정수 인덱스로 딕셔너리(dictionary)를 구축하는 것입니다).
2. 각 문서 `#i`에 대해, 각 단어 `w`가 나타난 횟수를 세어서 `X[i, j]`에 저장하는데, 여기서 특성 `#j`는 딕셔너리에서 `w`의 인덱스입니다.

단어 가방 표현은 `n_features`(특성 수)가 말뭉치(corpus)에서 개별 단어의 숫자임을 암시합니다: 이 숫자는 일반적으로 100,000개보다 많습니다.

`n_samples == 10000`이면, `X`를 float32 타입(type)의 넘파이(NumPy) 배열로 저장하려면 오늘날 컴퓨터에서 거의 관리할 수 없는 양인 10000 x 100000 x 4 바이트(bytes) = **4GB의 램(RAM)**이 필요합니다.

다행히도, 주어진 문서에서는 수천 개 이하의 개별 단어만 사용될 것이므로 **X의 대부분의 값은 0일 것입니다**. 이러한 이유로 우리는 단어 가방이 일반적으로 **고차원 희소 데이터셋(high-dimensional sparse datasets)**이라고 말합니다. 우린 오직 특성 벡터의 0이 아닌 부분만 메모리에 저장함으로써 많은 메모리를 아낄 수 있습니다.

`scipy.sparse` 행렬(matrices)은 정확히 이걸 해주는 데이터 구조이며, `scikit-learn`은 이러한 구조를 내장으로 지원합니다.

### `scikit-learn`으로 텍스트 토큰화

텍스트 전처리(preprocessing), 토큰화(tokenizing)과 불용어(stopwords)의 필터링(filtering)이 모두 [`CountVectorizer`](../../modules/generated/sklearn.feature_extraction.text.CountVectorizer)에 포함되어있으며, 이는 특성의 딕셔너리를 구축하고 문서를 특성 벡터로 변환합니다:

```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> count_vect = CountVectorizer()
>>> X_train_counts = count_vect.fit_transform(twenty_train.data)
>>> X_train_counts.shape
(2257, 35788)
```

[`CountVectorizer`](../../modules/generated/sklearn.feature_extraction.text.CountVectorizer)는 N-그램(N-grams)의 단어나 연속된 문자(consecutive characters)의 숫자 세기를 지원합니다. 한 번 적합되면(fitted), 벡터라이저(vectorizer)는 특성 인덱스의 딕셔너리를 구축한 상태입니다:

```python
>>> count_vect.vocabulary_.get(u'algorithm')
4690
```

어휘(vocabulary)에 있는 단어의 인덱스 값은 모든 훈련 말뭉치에서 그 빈도에 연결됩니다.

### 등장에서 빈도로

등장 수(occurrence count)는 좋은 시작이지만 문제가 있습니다: 같은 주제에 대해 이야기하더라도, 긴 문서는 짧은 문서보다 높은 평균 숫자를 가질 것입니다.

이러한 잠재적 모순을 피하려면 문서에서 각 단어의 등장 수를 문서 내 전체 단어 수로 나누어주기만 해도 충분합니다: 이 새로운 특성을 Term Frequencies(단어 빈도)에서 따와 `tf`이라 합니다.

`tf`에서 다른 개선점은 말뭉치의 많은 문서에서 등장하는 단어의 가중치(weights)를 낮춤으로써, 말뭉치의 작은 부분에서만 등장하는 단어에 비해 적은 정보를 주게끔 하는 것입니다.

이 규모 축소(downscaling)는 "Term Frequency times Inverse Document Frequency(단어 빈도 역 문서 빈도)"에서 따와 [`tf-idf`](https://en.wikipedia.org/wiki/Tf-idf)라 부릅니다.

**tf**와 **tf-idf**는 모두 다음과 같이 [`TfidfTransformer`](../../modules/generated/sklearn.feature_extraction.text.TfidfTransformer)를 사용해 계산할 수 있습니다:

```python
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
>>> X_train_tf = tf_transformer.transform(X_train_counts)
>>> X_train_tf.shape
(2257, 35788)
```

위 예시 코드에서, 첫번째로 데이터에 추정기를 적합시키기 위해 `fit(..)` 메서드(method)를 사용하고, 두번째로 우리의 숫자 행렬(count-matrix)을 tf-idf 표현으로 변형하기 위해 `transform(..)` 메서드를 사용했습니다. 이 두 단계는 중복된 처리를 생략하고 같은 최종 결과를 빠르게 얻기 위해 결합될 수 있습니다. 이는 아래에서 볼 수 있듯이, 그리고 이전 섹션의 참고에서 언급되었듯이, `fit_transform(..)` 메서드로 이뤄집니다:

```python
>>> tfidf_transformer = TfidfTransformer()
>>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
>>> X_train_tfidf.shape
(2257, 35788)
```

## 분류기 훈련

이제 우리는 특성을 가지고 있으니, 게시물의 범주를 예측해보도록 분류기를 훈련시킬 수 있습니다. 이 작업에 훌륭한 기초를 제공해줄 [나이브 베이즈(naïve Bayes)](../../modules/naive_bayes.) 분류기부터 시작해봅시다. `scikit-learn`은 이 분류기의 여러 변형(variants)을 포함합니다; 단어 세기를 위해 가장 적합한 것은 다항(multinomial)의 변형입니다:

```python
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
```

새 문서에서 결과를 예측하기 위해서는 이전과 거의 동일한 특성 추출 체인(feature extracting chain)을 사용해 특성을 추출해야합니다. 차이점이 있다면 이미 훈련 세트에 적합을 시켰기 때문에, 변환기(transformers)에서 `fit_transform` 대신 `transform`을 호출한다는 것입니다.

```python
>>> docs_new = ['God is love', 'OpenGL on the GPU is fast']
>>> X_new_counts = count_vect.transform(docs_new)
>>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)

>>> predicted = clf.predict(X_new_tfidf)

>>> for doc, category in zip(docs_new, predicted):
...     print('%r => %s' % (doc, twenty_train.target_names[category]))
...
'God is love' => soc.religion.christian
'OpenGL on the GPU is fast' => comp.graphics
```

## 파이프라인 구축

벡터라이저(vectorizer) => 변환기(transformer) => 분류기(classifier)와 일하기 쉽게 하기 위해서, `scikit-learn`은 복합적인 분류기처럼 작동하는 [`Pipeline`](../../modules/generated/sklearn.pipeline.Pipeline) 클래스를 제공합니다:

```python
>>> from sklearn.pipeline import Pipeline
>>> text_clf = Pipeline([
...     ('vect', CountVectorizer()),
...     ('tfidf', TfidfTransformer()),
...     ('clf', MultinomialNB()),
... ])
```

`vect`, `tfidf` 그리고 `clf`(분류기)라는 이름은 임의로 붙였습니다. 아래에서 적절한 초매개변수(hyperparameters)를 찾기 위한 격자 탐색(grid search)을 수행하기 위해 이들을 이용할 것입니다. 이제 하나의 명령어(command)로 모델을 훈련시킬 수 있습니다:

```python
>>> text_clf.fit(twenty_train.data, twenty_train.target)
Pipeline(...)
```

## 테스트 세트에 대한 성능 평가

모델의 예측 정확도(predictive accuracy)를 평가하는 것은 똑같이 쉽습니다:

```python
>>> import numpy as np
>>> twenty_test = fetch_20newsgroups(subset='test',
...     categories=categories, shuffle=True, random_state=42)
>>> docs_test = twenty_test.data
>>> predicted = text_clf.predict(docs_test)
>>> np.mean(predicted == twenty_test.target)
0.8348...
```

우리는 83.5% 정확도를 달성했습니다. (나이브 베이즈보다 약간 느리긴 하지만) 최고의 텍스트 분류 알고리즘 중 하나로 널리 여겨지는 선형 [서포트 벡터 머신(support vector machine, SVM)](../../modules/svm)으로 더 좋아질 수 있는지 봅시다. 간단하게 다른 분류기 객체를 우리 파이프라인에 연결하여 학습기를 바꿀 수 있습니다:

```python
>>> from sklearn.linear_model import SGDClassifier
>>> text_clf = Pipeline([
...     ('vect', CountVectorizer()),
...     ('tfidf', TfidfTransformer()),
...     ('clf', SGDClassifier(loss='hinge', penalty='l2',
...                           alpha=1e-3, random_state=42,
...                           max_iter=5, tol=None)),
... ])

>>> text_clf.fit(twenty_train.data, twenty_train.target)
Pipeline(...)
>>> predicted = text_clf.predict(docs_test)
>>> np.mean(predicted == twenty_test.target)
0.9101...
```

SVM을 사용하면서 91.3% 정확도를 달성했습니다. `scikit-learn`은 결과의 더 자세한 성능 분석을 위한 추가적인 유틸리티(utilities)를 제공합니다.

```python
>>> from sklearn import metrics
>>> print(metrics.classification_report(twenty_test.target, predicted,
...     target_names=twenty_test.target_names))
                        precision    recall  f1-score   support

           alt.atheism       0.95      0.80      0.87       319
         comp.graphics       0.87      0.98      0.92       389
               sci.med       0.94      0.89      0.91       396
soc.religion.christian       0.90      0.95      0.93       398

              accuracy                           0.91      1502
             macro avg       0.91      0.91      0.91      1502
          weighted avg       0.91      0.91      0.91      1502


>>> metrics.confusion_matrix(twenty_test.target, predicted)
array([[256,  11,  16,  36],
       [  4, 380,   3,   2],
       [  5,  35, 353,   3],
       [  5,  11,   4, 378]])
```

예상했듯이 혼동 행렬(confusion matrix)은 무신론(atheism)과 기독교(Christianity)에 관한 뉴스그룹에서 온 게시물들이 컴퓨터 그래픽(computer graphics)에서 온 것들보다 더 자주 혼동됨(confused)을 보여줍니다.

## 격자 탐색으로 매개변수 튜닝

우리는 이미 `TfidfTransformer`에서의 `use_idf`와 같은 몇 가지 매개변수를 마주쳤습니다. 분류기는 뿐만 아니라 많은 매개변수를 갖는 경향이 있습니다; 예를 들어, `MultinomialNB`는 평활화(smoothing) 매개변수 `alpha`를 포함하고 `SGDClassifier`는 벌칙(penalty) 매개변수 `alpha`와 설정 가능한 손실(loss) 및 벌칙 조건(terms)을 목적 함수(objective function)에서 갖고 있습니다(이들에 대한 자세한 설명을 원하시면 모듈 문서를 보시거나, 파이썬 `help` 함수를 사용하세요).

체인의 다양한 구성요소에서 매개변수를 조금씩 건드려보는 대신, 가능한 값들의 격자로 최고의 매개변수를 완전 탐색(exhaustive search) 해볼 수 있습니다. 단어 또는 바이그램(bigrams), idf가 있는지 없는지, 선형 SVM의 벌칙 매개변수가 0.01이거나 0.001이거나를 모든 분류기에 시도해봅니다.

```python
>>> from sklearn.model_selection import GridSearchCV
>>> parameters = {
...     'vect__ngram_range': [(1, 1), (1, 2)],
...     'tfidf__use_idf': (True, False),
...     'clf__alpha': (1e-2, 1e-3),
... }
```

분명히, 이런 완전 탐색은 비용이 많이 들 수 있습니다. 마음대로 사용할 수 있는 다중 CPU 코어(cores)가 있다면, `n_jobs` 매개변수로 격자 탐색기에게 병렬로(in parallel) 여덟 개 조합을 시도해보게 할 수 있습니다. 이 매개변수의 값을 `-1`로 주면, 격자 탐색은 얼마나 많은 코어가 설치되었는지 감지해서 모두 사용할 것입니다:

```python
>>> gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
```

격자 탐색 인스턴스(instance)는 일반적인 `scikit-learn` 모델처럼 작동합니다. 계산 속도를 높히기 위해 훈련 데이터의 더 작은 부분집합에서 탐색을 수행해봅시다:

```python
>>> gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
```

`GridSearchCV` 객체에서 `fit`을 호출한 결과물은 `predict`에서 사용할 수 있는 분류기입니다:

```python
>>> twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
'soc.religion.christian'
```

객체의 `best_score_`와 `best_params_` 속성은 최고의 평균 점수와 그 점수에 해당하는 매개변수를 저장합니다:

```python
>>> gs_clf.best_score_
0.9...
>>> for param_name in sorted(parameters.keys()):
...     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
...
clf__alpha: 0.001
tfidf__use_idf: True
vect__ngram_range: (1, 1)
```

탐색의 더 자세한 요약은 `gs_clf.cv_results_`에서 사용 가능합니다.

`cv_results_` 매개변수는 추가적인 검사를 위해 `DataFrame`으로 쉽게 판다스(pandas)에게로 가져올(imported) 수 있습니다.

### 연습

연습으로, 'skeletons'(뼈대구조) 폴더의 내용물을 'workspace'(작업공간)이라는 새 폴더로 복사하세요:

```sh
$ cp -r skeletons workspace
```

그러면 원본 연습 설명서를 잃어버린다는 두려움 없이도 작업공간의 내용물을 편집할 수 있습니다.

그 다음 아이파이썬 셸(ipython shell)을 실행하고 진행 중인 스크립트를 다음과 같이 실행하세요:

```python
[1] %run workspace/exercise_XX_script.py arg1 arg2 arg3
```

예외(exception)가 발생하면(triggered), `%debug`을 써서 사후(post mortem) ipdb 세션을 시작하세요.

연습이 해결될 때까지 구현(implementation)을 정제(refine)하고 반복하세요.

**각 연습에 대해, 뼈대구조 파일은 모든 필요한 가져오기(import) 구문과 데이터를 불러오기 위한 재사용 코드(boiler-plate code), 그리고 모델의 예측 정확도를 평가하기 위한 견본 코드를 제공합니다.**

## 연습 1: 언어 식별

- 직접 만든 전처리기 및 훈련 세트로 위키피디아(Wikipedia) 기사 데이터를 사용하는 `CharNGramAnalyzer`를 사용해 텍스트 분류 파이프라인을 작성하세요.
- 보류된(held out) 일부 테스트 세트에 대해 성능을 평가하세요.

아이파이썬 명령줄:

```python
%run workspace/exercise_01_language_train_model.py data/languages/paragraphs/
```

## 연습 2: 영화 후기 감성 분석

- 영화 후기가 긍정적인지 부정적인지 분류하는 텍스트 분류 파이프라인을 작성하세요.
- 격자 탐색으로 좋은 매개변수 집합을 찾으세요.
- 보류된 테스트 세트에 대해 성능을 평가하세요.

아이파이썬 명령줄:

```python
%run workspace/exercise_02_sentiment.py data/movie_reviews/txt_sentoken/
```

## 연습 3: CLI 텍스트 분류 유틸리티

이전 연습들의 결과와 표준 라이브러리(standard library)의 `cPickle` 모듈을 사용하여, `stdin`에서 제공하는 몇몇 텍스트의 언어를 감지하고 만약 텍스트가 영어로 쓰였는지 극성(polarity, 양성(positive) 또는 음성(negative))을 추정하는 명령줄 유틸리티를 작성하세요.

유틸리티가 예측에 대해 신뢰 수준(confidence level)을 제공할 수 있다면 보너스 점수입니다.

## 여기서부터 어디로

여기 이 튜토리얼 완료 후 여러분의 사이킷런 직관력(intuition)을 향상시키는데 도움을 주는 몇 가지 제안이 있습니다:

- [`CountVectorizer`](../../modules/generated/sklearn.feature_extraction.text.CountVectorizer)의 `analyzer`와 `token normalisation`를 가지고 놀아보세요.
- 레이블이 없다면, 여러분의 문제에 [군집화(Clustering)](../../auto_examples/text/plot_document_clustering)을 써보세요.
- 범주처럼 한 문서 당 여러 레이블이 있다면, [다중클래스와 다중레이블 섹션](../../modules/multiclass)을 살펴보세요.
- [잠재 의미 분석](https://en.wikipedia.org/wiki/Latent_semantic_analysis)을 위해 [절단 SVD(truncated SVD)](../../modules/decomposition#lsa)를 사용해보세요.
- 컴퓨터 메인 메모리(main memory)에 들어가지 않는 데이터에서 학습하기 위해서는 [아웃 오브 코어(out-of-core) 분류](../../auto_examples/applications/plot_out_of_core_classification) 사용을 살펴보세요.
- [`CountVectorizer`](../../modules/generated/sklearn.feature_extraction.text.CountVectorizer)보다 메모리 효율적인 대안으로 [해싱 벡터라이저(hashing vectorizer)](../../modules/feature_extraction#hashing-vectorizer)를 살펴보세요.
