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

반환된 데이터셋은 `scikit-learn` "묶음(bunch)"입니다: 파이썬(python) `dict` 키(keys)나 `object` 속성(attributes)으로 접근하는 필드(fields)가 있는 간단한 홀더(holder) 객체로, 예를 들어 `target_names`는 요청된 범주 이름의 목록을 가지고 있습니다:

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

이러한 잠재적 모순을 피하려면 문서의 각 단어의 등장 수를 문서 내 전체 단어 수로 나누어주기만 해도 충분합니다: 이 새로운 특성을 Term Frequencies(단어 빈도)에서 따와 `tf`이라 합니다.

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


## 파이프라인 구축


## 테스트 세트에 대한 성능 평가


## 격자 탐색으로 매개변수 튜닝


## 연습 1: 언어 식별


## 연습 2: 영화 후기 감성 분석


## 연습 3: CLI 텍스트 분류 유틸리티


## 여기서부터 어디로