원문: [Frequently Asked Questions](https://scikit-learn.org/stable/faq.html)

# 자주 묻는 질문

여기에는 메일링 리스트(mailing list)에 꾸준히 올라오는 질문들에 대해 몇 가지 답변을 드리고자 합니다.

**목차**

- [프로젝트에 대해](#프로젝트에-대해)
  - [(많은 사람들이 잘못 알고 있는) 프로젝트의 이름이 무엇인가요?](#(많은-사람들이-잘못-알고-있는)-프로젝트의-이름이-무엇인가요)
  - [프로젝트 이름을 어떻게 발음하나요?](#프로젝트-이름을-어떻게-발음하나요)
  - [왜 사이킷인가요?](#왜-사이킷인가요)
  - [파이파이를 지원하나요?](#파이파이를-지원하나요)
- [구현 결정](#구현-결정)
  - [왜 심층 또는 강화 학습에 대한 지원이 없는지 / 사이킷런에 심층 또는 강화 학습 지원이 있을까요?](#왜-심층-또는-강화-학습에-대한-지원이-없는지-/-사이킷런에-심층-또는-강화-학습-지원이-있을까요)
  - [사이킷런에 그래프 모델이나 시퀀스 예측을 추가하실 건가요?](#사이킷런에-그래프-모델이나-시퀀스-예측을-추가하실-건가요)
  - [왜 사이킷런에서 HMMs를 제거하셨나요?](#왜-사이킷런에서-HMMs를-제거하셨나요)
  - [GPU 지원을 추가하실건가요?](#GPU-지원을-추가하실건가요)
  - [왜 다른 도구들과 다르게, 사이킷런에서는 범주형 변수의 전처리가 필요한가요?](#왜-다른-도구들과-다르게,-사이킷런에서는-범주형-변수의-전처리가-필요한가요)
  - [왜 사이킷런은, 예를 들어 pandas.DataFrame과, 직접적으로 같이 작동하지 않나요?](#왜-사이킷런은,-예를-들어-pandas.DataFrame과,-직접적으로-같이-작동하지-않나요)
  - [파이프라인에서 목표 y를 위한 변환을 구현할 계획이신가요?](#파이프라인에서-목표-y를-위한-변환을-구현할-계획이신가요)
  - [선형 모델에는 왜 그렇게 다양한 추정기가 있나요?](#선형-모델에는-왜-그렇게-다양한-추정기가-있나요)
- [기여하기](#기여하기)
  - [어떻게 사이킷런에 기여할 수 있을까요?](#어떻게-사이킷런에-기여할-수-있을까요)
  - [왜 저의 풀 리퀘스트는 어떤 관심도 받지 못할까요?](#왜-저의-풀-리퀘스트는-어떤-관심도-받지-못할까요)
  - [새 알고리즘을 포함하게 되는 기준은 무엇인가요?](#새-알고리즘을-포함하게-되는-기준은-무엇인가요)
  - [사이킷런에서 포함할 알고리즘을 고를 때 왜 그렇게 선택적인가요?](#사이킷런에서-포함할-알고리즘을-고를-때-왜-그렇게-선택적인가요)
- [사이킷런 사용하기](#사이킷런-사용하기)
  - [사이킷런 사용법의 도움을 얻기 위한 최고의 방법은 무엇인가요?](#사이킷런-사용법의-도움을-얻기-위한-최고의-방법은-무엇인가요)
  - [운영을 하려면 추정기를 어떻게 저장하고 내보내거나 배포해야 하나요?](#운영을-하려면-추정기를-어떻게-저장하고-내보내거나-배포해야-하나요)
  - [묶음 객체를 어떻게 만들 수 있나요?](#묶음-객체를-어떻게-만들-수-있나요)
  - [나만의 데이터셋을 어떻게 사이킷런에서 쓸만한 형식으로 불러올 수 있을까요?](#나만의-데이터셋을-어떻게-사이킷런에서-쓸만한-형식으로-불러올-수-있을까요)
  - [문자열 데이터(또는 트리, 그래프...)를 어떻게 다룰까요?](#문자열-데이터(또는-트리,-그래프...)를-어떻게-다룰까요)
  - [왜 OSX나 리눅스에서 가끔 n_jobs > 1과 함께 충돌/멈춤이 일어날까요?](#왜-OSX나-리눅스에서-가끔-n_jobs->-1과-함께-충돌/멈춤이-일어날까요)
  - [내 작업이 왜 지정된 n_jobs보다 많은 코어를 사용할까요?](#내-작업이-왜-지정된-n_jobs보다-많은-코어를-사용할까요)
  - [어떻게 `random_state`를 전체 실행에 대해 설정하나요?](#어떻게-`random_state`를-전체-실행에-대해-설정하나요)

## 프로젝트에 대해

### (많은 사람들이 잘못 알고 있는) 프로젝트의 이름이 무엇인가요?

scikit-learn(사이킷런), 하지만 scikit이나 SciKit이나 sci-kit learn은 아닙니다. 예전에 그렇게 쓰이긴 했지만 scikits.learn이나 scikits-learn도 아닙니다.

### 프로젝트 이름을 어떻게 발음하나요?

sy-kit learn(사이-킷 런). sci는 science(과학)을 의미합니다!

### 왜 사이킷인가요?

사이파이(SciPy)를 중심으로 구축된 과학적 도구모음인 다양한 사이킷들이 있습니다. 사이킷런 외에도, 다른 유명한 것은 [사이킷이미지(scikit-image)](https://scikit-image.org/)가 있습니다.

### 파이파이를 지원하나요?

사이킷런은 (적시 컴파일러(just-in-time compiler)가 내장된 대체 파이썬(Python) 구현체인) [파이파이(PyPy)](https://pypy.org/)와 함께 작동하기 위해 규칙적으로 테스트와 유지보수를 합니다.

그러나 이 지원사항은 여전히 실험적인 것으로 간주되며 특정 구성요소는 약간 다르게 작동할 수 있습니다. 더 자세한 사항은 관심 있는 특정 모듈(module)의 테스트 수트(test suite)를 참고하세요.

## 구현 결정

### 왜 심층 또는 강화 학습에 대한 지원이 없나요 / 사이킷런에 심층 또는 강화 학습 지원이 있을까요?

심층 학습(deep learning)과 강화 학습(reinforcement learning)은 둘다 아키텍처(architecture)를 정의하기 위한 풍부한 어휘가 필요하며, 심층 학습은 추가적으로 효율적인 계산을 위해 GPU가 필요합니다. 하지만, 이 중 어느 것도 사이킷런의 설계 제약조건에 맞지 않습니다; 결과적으로, 심층 학습과 강화 학습은 현재 사이킷런이 이룩하고자 하는 범위를 벗어납니다.

GPU 지원을 추가하는 것에 대한 더 많은 정보를 [GPU 지원을 추가하실건가요?](#GPU-지원을-추가하실건가요)에서 찾아보실 수 있습니다.

### 사이킷런에 그래프 모델이나 시퀀스 예측을 추가하실 건가요?

예측가능한 미래에는 아닙니다. 사이킷런은 파이프라인 및 격자 탐색(grid search) 같은 메타 알고리즘(meta-algorithms)과 함께, 기계 학습의 기초 작업을 위한 통합된 API를 제공해 모든 것을 함께 묶고자 합니다. 구조화된 학습(structured learning)을 위해 필요한 개념, APIs, 알고리즘과 전문가는 사이킷런이 제공하는 것들과는 다릅니다. 우리가 임의적으로 구조화된 학습으로 시작했다면, 전체 패키지를 다시 설계하고 프로젝트는 스스로의 무게로 인해 무너질 가능성이 높습니다.

사이킷런과 유사한 API의 구조화된 예측을 하는 두 프로젝트가 있습니다:

- [파이스트럭트(pystruct)](https://pystruct.github.io/)는 일반적인 구조화된 학습을 처리합니다(대략적인 추론으로 임의적인 그래프 구조(graph structures)에 대한 SSVMs에 초점을 둡니다; 그래프 구조의 인스턴스(instance)로 표본의 개념을 정의합니다)
- [시크런(seqlearn)](https://larsmans.github.io/seqlearn/)은 시퀀스(sequences)만 처리합니다(정확한 추론에 초점을 둡니다; HMMs가 있지만, 대부분 완전성을 위해서입니다; 특성 벡터(feature vector)를 표본으로 다루고 특성 벡터 간 의존성을 위해 오프셋 인코딩(offset encoding)을 사용합니다)

### 왜 사이킷런에서 HMMs를 제거하셨나요?

[사이킷런에 그래프 모델이나 시퀀스 예측을 추가하실 건가요?](#사이킷런에-그래프-모델이나-시퀀스-예측을-추가하실-건가요)를 보세요.

### GPU 지원을 추가하실건가요?

아니요, 혹은 가까운 미래에는 적어도 아닙니다. 주된 이유는 GPU 지원이 많은 소프트웨어 의존성을 발생시키고 플랫폼(platform) 특이적인 문제도 발생시키기 때문입니다. 사이킷런은 넓은 다양성의 플랫폼에 설치하기 쉽게 설계되었습니다. 신경망(neural networks)을 제외하면, GPUs는 오늘날의 기계 학습에서 큰 역할을 하지 않으며, 종종 알고리즘을 신중하게 선택하는 것이 속도에서 더 큰 이득을 볼 수 있습니다.

### 왜 다른 도구들과 다르게, 사이킷런에서는 범주형 변수의 전처리가 필요한가요?

대부분의 사이킷런은 데이터가 넘파이(NumPy) 배열이나 사이파이(SciPy) 희소 행렬(sparse matrices)에 단일한 수치 데이터 타입(numeric dtype)으로 있다고 가정합니다. 현재 이는 범주형 변수(categorical variables)를 명시적으로 표현하고 있지 않습니다. 따라서, R의 data.frames나 pandas.DataFrame과 달리, [범주형 특성 부호화](modules/preprocessing#범주형-특성-부호화)에서 논의한대로, 범주형 특성을 수치 값으로 명시적으로 변환해야 합니다. 이종(heterogeneous) (예를 들면 범주형과 수치형) 데이터로 작업하는 예제로 [혼합 타입과 열 변환기](auto_examples/compose/plot_column_transformer_mixed_types)도 보세요.

### 왜 사이킷런은, 예를 들어 pandas.DataFrame과, 직접적으로 같이 작동하지 않나요?

동종(homogeneous)의 넘파이(NumPy)와 사이파이(SciPy) 데이터 객체가 현재 대부분 작업을 처리하기에 가장 효율적일 것이라 기대됩니다. 판다스(Pandas) 범주형 타입을 지원하려면 아마 광범위한 작업도 필요할 것입니다. 입력을 동종 타입으로 제한하는 것은 따라서 유지 비용을 절감하고 효율적인 자료 구조(data structures)의 사용을 장려합니다.

그러나 [`ColumnTransformer`](modules/generated/sklearn.compose.ColumnTransformer)가 이름이나 데이터 타입(dtype)으로 선택한 이종(heterogeneous) 데이터프레임(dataframe)의 열 일부분을 전용 사이킷런 변환기(transformers)에 대응(mapping)시켜 판다스 데이터프레임을 다루기 쉽게 해준다는 것을 참고하세요.

따라서 [`ColumnTransformer`](modules/generated/sklearn.compose.ColumnTransformer)는 이종 데이터프레임을 처리할 때 사이킷런 파이프라인(pipelines)의 첫 단계에서 자주 사용됩니다(더 자세한 내용은 [파이프라인: 추정기 연결하기](modules/compose#파이프라인-추정기-연결하기)를 보세요).

이종(heterogeneous) (예를 들면 범주형과 수치형) 데이터로 작업하는 예제로 [혼합 타입과 열 변환기](auto_examples/compose/plot_column_transformer_mixed_types)도 보세요.

### 파이프라인에서 목표 y를 위한 변환을 구현할 계획이신가요?

현재 변환은 파이프라인에서 특성 X에만 작동합니다. 파이프라인에서 y를 변환할 수 없음에 대한 오랜 논의가 있습니다. 깃헙 이슈(github issue) [#4143](https://github.com/scikit-learn/scikit-learn/issues/4143)에서 따라오세요. 한편 [`TransformedTargetRegressor`](modules/generated/sklearn.compose.TransformedTargetRegressor), [파이프그래프(pipegraph)](https://github.com/mcasl/PipeGraph), [임밸런스드런(imbalanced-learn)](https://github.com/scikit-learn-contrib/imbalanced-learn)을 확인해보세요. 사이킷런이 훈련 전에 y에 가역 변환(invertable transformation)이 가해지고 예측 후에 역변환(inverted)되었을 때의 경우를 해결했음을 참고하세요. 사이킷런은 `imbalanced-learn`에서처럼 재표집(resampling)과 비슷한 사례에서, y가 테스트 시간이 아닌 훈련 시간에 변환되어야 할 때의 사용 사례(use cases)를 해결하고자 합니다. 일반적으로, 이러한 사용 사례는 파이프라인보다는 사용자가 직접 정의한 메타 추정기(meta estimator)로 해결할 수 있습니다.

### 선형 모델에는 왜 그렇게 다양한 추정기가 있나요?

보통, 예를 들어 [`GradientBoostingClassifier`](modules/generated/sklearn.ensemble.GradientBoostingClassifier)와 [`GradientBoostingRegressor`](modules/generated/sklearn.ensemble.GradientBoostingRegressor)처럼, 모델(model) 유형마다 하나의 분류기(classifier)와 하나의 회귀자(regressor)가 있습니다. 둘다 비슷한 옵션이 있으며, 조건부 분위수(conditional quantiles)뿐 아니라 조건부 평균(conditional mean)의 추정도 가능케 하여 회귀 사례에서 특히 유용한 매개변수 `loss`도 둘다 가집니다.

선형 모델을 위해, 서로 아주 가까운 많은 추정기 클래스가 있습니다. 한번 살펴보자면

- [`LinearRegression`](modules/generated/sklearn.linear_model.LinearRegression), 벌칙(penalty) 없음
- [`Ridge`](modules/generated/sklearn.linear_model.Ridge), L2 벌칙
- [`Lasso`](modules/generated/sklearn.linear_model.Lasso), L1 벌칙(희소 모델(sparse models))
- [`ElasticNet`](modules/generated/sklearn.linear_model.ElasticNet), L1 + L2 벌칙(덜 희소한 모델)
- [`SGDRegressor`](modules/generated/sklearn.linear_model.SGDRegressor)를 `loss='squared_loss'`와 함께

**관리자 관점:** 그들은 모두 원리는 동일하며 부과하는 벌칙에 따라서만 다릅니다. 그러나, 이에 깔려있는 최적화 문제를 해결하는 방식에 큰 영향을 줍니다. 결국에, 선형 대수학(linear algebra)에서 온 다양한 방법과 요령을 사용하는 결과에 이릅니다. 특별한 사례로는 모든 4개의 이전 모델들로 구성되어있으며 최적화 과정에 따라 달라지는 `SGDRegressor`이 있습니다. 추가적인 부작용은 다른 추정기가 다른 데이터 레이아웃(data layouts)을 선호하는 것입니다(`X` c-연속(contiguous)이나 f-연속, 희소 csr이나 csc). 겉으로 보기에는 단순한 선형 모델의 이러한 복잡성은 다양한 벌칙에 따라 다른 추정기 클래스를 갖는 이유입니다.

**사용자 관점:** 첫째로, 현재 설계는 다양한 정규화(regularization)/벌칙과 함께 하는 선형 회귀 모델들이, 예를 들어 *릿지 회귀(ridge regression)*처럼 각각 다른 이름을 받았다는 과학 문헌에서 영감을 받았습니다. 이름에 따라 다른 모델 클래스를 갖는 것이 사용자들에게 이 회귀 모델을 찾는 것을 쉽게 만들어줍니다. 둘째로, 언급된 위 5개의 모든 선형 모델이 단일 클래스로 통합되면, `solver` 매개변수처럼 정말 많은 옵션을 갖고 있는 매개변수들이 있게 될 것입니다. 무엇보다도, 여러 매개변수 간에 배타적인 상호작용(exclusive interactions)이 정말 많이 있을 것입니다. 예를 들어, 매개변수 `solver`, `precompute`, 그리고 `selection`의 가능한 옵션들은 벌칙 매개변수 `alpha`와 `l1_ratio`의 선택된 값에 의존하게 될 것입니다.

## 기여하기

### 어떻게 사이킷런에 기여할 수 있을까요?

[기여하기](https://scikit-learn.org/dev/developers/contributing.html#contributing)를 보세요. 중요하고 시간이 오래 걸리는 새 알고리즘 추가를 원하기보단, 그 전에 [알려진 문제들](https://scikit-learn.org/dev/developers/contributing.html#new-contributors)부터 시작하는 것을 추천드립니다. 사이킷런에 기여하기에 대해 사이킷런 기여자들에게 직접 연락하지 마시길 부탁드립니다.

### 왜 저의 풀 리퀘스트는 어떤 관심도 받지 못할까요?

사이킷런 리뷰(review) 과정은 꽤 많은 시간이 걸리며, 기여자분들께서는 풀 리퀘스트(pull request)에 대한 활동이나 리뷰가 부족하다고 낙심하지 마세요. 유지보수와 사후 변경은 비용이 많이 들기 때문에, 저희는 처음부터 올바르게 처리되는 것에 관심을 많이 기울입니다. 어떠한 "실험용" 코드를 릴리즈하지 않으므로, 모든 컨트리뷰션(contribution)은 즉시 많은 사용이 있어야 하며 즉시 가능한 한 최고의 품질이어야 합니다.

그 외에도, 사이킷런은 리뷰의 대역폭(bandwidth)이 제한되어 있습니다; 많은 리뷰어들(reviewers)과 핵심 개발자들은 그들만의 시간에 사이킷런을 작업하고 있습니다. 여러분의 풀 리퀘스트에 대한 리뷰가 느리게 오면, 그건 리뷰어가 바쁘기 때문일 것입니다. 이러한 이유 하나만으로 여러분의 풀 리퀘스트를 닫거나 작업을 중단하시지 않기를 양해와 요청 부탁드립니다.

### 새 알고리즘을 포함하게 되는 기준은 무엇인가요?

저희는 오직 잘 확립된 알고리즘만 포함을 고려합니다. 경험적인 규칙으로는 출판 후 적어도 3년이 지났으며, 200+ 인용이 되었고, 넓은 사용법과 유용성이 있습니다. 널리 사용된느 방법에 대한 명확한(clear-cut) 개선을 제공하는 기법(예를 들어 향상된 자료 구조나 더 효율적인 추정 기법)도 또한 포함을 위해 고려할 것입니다.

위 기준을 충족하는 알고리즘이나 기법 중에서, 사이킷런의 현재 API에 잘 들어맞는 것만, 말하자면 `fit`, `predict/transform` 인터페이스(interface)와 보통 넘파이(numpy) 배열이나 희소 행렬(sparse matrix)를 입/출력으로 가질 때, 승인합니다.

기여자는 연구 논문에서 제안한 추가사항, 그리고/또는 다른 비슷한 패키지들에서의 구현의 중요성을 지원해야 하고, 일반적인 사용 사례/응용을 통해 유용성을 입증하며, 만약 성능 개선이 있다면 벤치마크(benchmarks) 그리고/또는 도표(plots)와 함께 이를 확증해야 합니다. 제안한 알고리즘은 최소한 어떤 영역에서는 사이킷런에 이미 구현된 기법의 성능을 뛰어넘어야 하는 것으로 기대합니다.

이미 존재하는 모델의 속도를 높이는 새로운 알고리즘은 만약 이렇다면 쉽게 포함될 것입니다:

- 새로운 초매개변수(hyper-parameters)를 도입하지 않거나(라이브러리가 훗날을 더 잘 대비하게 합니다),
- 예를 들어 "n_features >> n_samples일 때"처럼, 컨트리뷰션(contribution)이 속도를 향상시킬 때와 그렇지 않을 때를 깔끔하게 문서화하기 쉽거나,
- 벤치마크가 명확하게 속도 향상을 보여주거나 입니다.

또한, 여러분의 구현을 사이킷런 도구들과 함께 사용하기 위해 사이킷런에 있어야할 필요는 없다는 것을 참고하세요. 여러분은 가장 좋아하는 알고리즘을 사이킷런에 호환(compatible)되는 방식으로 구현하시고, 깃헙(GitHub)에 업로드한 다음 저희가 알도록 하실 수 있습니다. 저희가 기꺼이 [관련 프로젝트](related_projects) 아래에 나열하겠습니다. 이미 여러분이 사이킷런 API를 따르는 패키지를 깃헙에 가지고 계시다면, [사이킷런컨트립(scikit-learn-contrib)](https://github.com/scikit-learn-contrib/scikit-learn-contrib/blob/master/README.md)을 살펴보시는데 흥미가 있으실겁니다.

### 사이킷런에서 포함할 알고리즘을 고를 때 왜 그렇게 선택적인가요?

코드에는 유지보수 비용이 따라오며, 저희는 가지고 있는 코드의 양과 팀의 크기에서 균형을 맞춰야 합니다(그리고 복잡도(complexity)는 기능의 수에 따라 비선형적으로(non linearly) 커진다는 사실도 추가해야 합니다). 패키지는 버그를 고치고 코드를 유지보수하며 컨트리뷰션(contribution)을 리뷰(review)하는 것에 여유 시간을 사용하는 핵심 개발자들에게 기대고 있습니다. 모든 추가된 알고리즘은 개발자들이 향후 관심을 주어야 하지만, 그 시점에 원저자가 한참동안 관심을 잃었을 수도 있습니다. [새 알고리즘을 포함하게 되는 기준은 무엇인가요?](#새-알고리즘을-포함하게-되는-기준은-무엇인가요)도 보세요. 오픈소스 소프트웨어(open-source software)의 장기 유지보수 문제에 대한 좋은 독서로는, [도로 및 교량 핵심 보고(the Executive Summary of Roads and Bridges)](https://www.fordfoundation.org/media/2976/roads-and-bridges-the-unseen-labor-behind-our-digital-infrastructure.pdf#page=8)를 살펴보세요.

## 사이킷런 사용하기

### 사이킷런 사용법의 도움을 얻기 위한 최고의 방법은 무엇인가요?

**일반적인 기계 학습 질문은**, [크로스 밸리데이티드(Cross Validated)](https://stackoverflow.com/questions/tagged/scikit-learn)의 `[machine-learning]` 태그(tag)를 사용해주세요.

**사이킷런 사용법 질문은**, [스택 오버플로우(Stack Overflow)](https://stackoverflow.com/questions/tagged/scikit-learn)의 `[scikit-learn]`과 `[python]` 태그를 사용해주세요. 아니면 [메일링 리스트(mailing list)](https://mail.python.org/mailman/listinfo/scikit-learn)를 사용하실 수도 있습니다.

(예를 들어 `sklearn.datasets`에서나 고정된 무작위 시드(seed)의 `numpy.random` 함수로 무작위로 만들어진) 토이 데이터셋(toy dataset)에서 여러분의 문제를 강조하는 (이상적으로는 10줄 미만의) 최소한의 재생산용 코드 스니펫(code snippet)을 포함해야함을 명심하세요. 여러분의 문제를 재생산하는데 필요하지 않은 어떠한 코드 줄도 삭제해주세요.

문제는 사이킷럿이 설치된 파이썬 셸(Python shell)에 여러분의 코드 스니펫을 쉽게 복사 붙여넣기하여 재생산할 수 있어야 합니다. 들여오기(import) 구문을 꼭 포함하도록 하세요.

좋은 재생산 코드 스니펫을 작성하는 더 많은 지침들은 다음에서 찾을 수 있습니다:

[https://stackoverflow.com/help/mcve](https://stackoverflow.com/help/mcve)

여러분의 문제가 (구글링(googling)을 해봐도) 이해할 수 없는 예외(exception)를 일으킨다면, 재생산 스크립트(script)를 실행할 때 얻은 전체 역추적(traceback)을 포함하도록 꼭 부탁드립니다.

버그 제보나 기능 요구는, [GitHub(깃헙)의 이슈 트랙커(issue tracker)](https://github.com/scikit-learn/scikit-learn/issues)를 사용해주세요.

몇몇 사용자와 개발자들을 찾을 수 있는 [사이킷런 기터(Gitter) 채널](https://gitter.im/scikit-learn/scikit-learn)도 있습니다.

**도움을 요청하거나, 버그를 제보하거나, 또는 사이킷런에 관련된 어떤 문제들에 대해서도, 어떤 저자들에게도 직접 메일을 보내지 마세요.**

### 운영을 하려면 추정기를 어떻게 저장하고 내보내거나 배포해야 하나요?

[모델 영속성](model_persistence)을 보세요.

### 어떻게 묶음 객체를 만들 수 있나요?

묶음 객체(bunch objects)는 가끔 함수(functions)와 메서드(methods)의 출력으로 사용됩니다. 그들은 `bunch["value_key"]` 키(key)나 `bunch.value_key` 속성(attribute)으로 값에 접근할 수 있게끔 딕셔너리(dictionaries)를 확장합니다.

그들을 입력으로 사용하면 안됩니다; 게다가 여러분은 사이킷런 API를 확장하지 않는다면, 거의 `Bunch` 객체를 만들 필요가 없을 것입니다.

### 어떻게 나만의 데이터셋을 사이킷런에서 쓸만한 형식으로 불러올 수 있을까요?

일반적으로, 사이킷런은 넘파이(numpy) 배열이나 사이파이(scipy) 희소 행렬(sparse matrices)로 저장된 아무 수치 데이터에나 작동합니다. 다른 판다스(pandas) 데이터프레임(DataFrame)같은 수치 배열로 변환 가능한 유형들도 받아들일 수 있습니다.

여러분의 데이터 파일을 이러한 사용가능한 데이터 구조로 불러오는 것에 대한 더 많은 정보로는, [외부 데이터셋 불러오기](datasets/loading_other_datasets#외부-데이터셋에서-불러오기)를 참고해주세요.

### 문자열 데이터(또는 트리, 그래프...)를 어떻게 다룰까요?

사이킷런 추정기(estimators)는 여러분이 데이터를 실수값의 특성 벡터(feature vectors)로 제공할 것이라 가정합니다. 이 가정은 거의 모든 라이브러리에서 하드 코딩(hard-coded)되어 있습니다. 하지만, 여러분은 여러 방법으로 수치형이 아닌 입력을 추정기에 제공할 수 있습니다.

여러분이 텍스트 문서(text documents)가 있다면, 단어 빈도(term frequency) 특성을 사용할 수 있습니다. 내장 *텍스트 벡터라이저(text vectorizers)*를 위해, [텍스트 특성 추출](modules/feature_extraction#텍스트-특성-추출)을 보세요. 어떤 종류의 데이터에서든지 더 일반적인 특성 추출을 원하시면, [딕셔너리에서 특성 불러오기](modules/feature_extraction#딕셔너리에서-특성-불러오기)와 [특성 해싱](modules/feature_extraction#특성-해싱)을 보세요.

또다른 일반적인 사례는 여러분이 수치형이 아닌 데이터와 이에 대한 자체적인 거리(distance, 또는 유사도(similarity)) 측정(metric)이 있을 때입니다. 예제는 편집 거리(다른 말로는 레벤슈타인(Levenshtein) 거리; 예를 들면 DNA나 RNA 서열(sequences))와 함께 하는 문자열(strings)를 포함합니다. 이는 숫자로 부호화(encoded)할 수 있지만, 그건 너무 고통스럽고 오류가 발생하기 쉽습니다(error-prone). 임의의 데이터에서 거리 측정 작업은 두 가지 방식으로 이루어질 수 있습니다.

첫째로, 많은 추정기는 미리 계산된 거리/유사도 행렬을 받아서, 데이터셋이 너무 크지 않으면, 여러분은 입력의 모든 쌍에 대한 거리를 계산할 수 있습니다. 데이터셋이 너무 크면, 여러분은 분리된 자료 구조(data structure)에 대한 인덱스(index)로 된 한 "특성"만 갖는 특성 벡터를 사용할 수 있으며, 이 자료 구조에서 실제 데이터를 바라보는 자체적인 측정 함수를 적용할 수 있습니다. 예를 들어, 레벤슈타인 거리와 함께 DBSCAN을 사용하려면:

```python
>>> from leven import levenshtein
>>> import numpy as np
>>> from sklearn.cluster import dbscan
>>> data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
>>> def lev_metric(x, y):
...     i, j = int(x[0]), int(y[0])     # 인덱스를 추출합니다
...     return levenshtein(data[i], data[j])
...
>>> X = np.arange(len(data)).reshape(-1, 1)
>>> X
array([[0],
       [1],
       [2]])
>>> # 연속 특성 공간(continuous feature space)을 기본 가정으로 하려면
>>> # algorithm='brute'로 특정해주어야 합니다.
>>> dbscan(X, metric=lev_metric, eps=5, min_samples=2, algorithm='brute')
... 
([0, 1], array([ 0,  0, -1]))
```

(여기서는 서드파티(third-party) 편집 거리 패키지 `leven`을 사용합니다.)

조금만 주의한다면, 트리 커널(tree kernels)과 그래프 커널(graph kernels) 등에도 비슷한 요령을 사용할 수 있습니다.

### 왜 OSX나 리눅스에서 가끔 n_jobs > 1과 함께 충돌/멈춤이 일어날까요?

`GridSearchCV`와 `cross_val_score` 같은 다양한 사이킷런 도구들은 `n_jobs > 1`을 인자(argument)로 전달하면서, 실행을 여러 파이썬 프로세스(processes)에서 병렬화(parallelize)하기 위해, 내부적으로 파이썬의 `multiprocessing` 모듈(module)에 의존합니다.

문제는 성능 상의 이유로 파이썬 `multiprocessing` 모듈이 `exec` 시스템 호출(system call)을 따르는 대신 `fork` 시스템 호출을 한다는 것입니다. (일부 버전의) OSX의 Accelerate/vecLib, (일부 버전의) MKL, GCC의 OpenMP 런타임(runtime), 엔비디아(nvidia)의 쿠다(Cuda)(그리고 아마 더 다양한) 많은 라이브러리들은 그들의 내부적인 스레드 풀(thread pool)을 관리합니다. `fork`를 호출하면, 자식 프로세스(child process)의 스레드 풀 상태(state)가 손상됩니다: 스레드 풀은 주 스레드(main thread) 상태가 포크되었을(forked)뿐인데도 많은 스레드가 있다고 믿습니다. 그런 사례에서 라이브러리를 바꿔서 포크가 일어났는지 인식하고 스레드 풀을 다시 시작할 수 있습니다: 저희가 (메인(main)에 0.2.10부터 병합된) OpenBLAS에 그렇게 해두었고 (아직 리뷰되지 않았지만) GCC의 OpenMP 런타임에 [패치(patch)](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60035)를 기여했습니다.

하지만 결국 진짜 범인은, 병렬 계산을 위한 새로운 파이썬 프로세스를 시작하고 사용하는 오버헤드(overhead)를 줄이기 위해 `exec` 없이 `fork`를 하는 파이썬의 `multiprocessing`입니다. 불행하게도 이는 POSIX 표준에 대한 위반이므로, 애플(Apple)과 같은 일부 소프트웨어 편집자(editors)는 Accelerate/vecLib에서 포크-안전성(fork-safety)의 부족을 버그로 간주하기를 거부합니다.

파이썬 3.4+에서 이제 프로세스 풀 관리를 위한 시작 방식으로 ('fork' 대신) 'forkserver'나 'spawn'을 사용하는 `multiprocessing`을 설정할 수 있게 되었습니다. 사이킷런을 사용할 때 이 문제를 해결하려면, `JOBLIB_START_METHOD` 환경 변수(environment variable)를 'forkserver'로 설정하시면 됩니다. 하지만 사용자는 'forkserver' 방식을 사용하는 것이 joblib.Parallel이 셸 세션(shell session)에서 대화형으로(interactively) 정의된 함수를 부르는 것을 막는단 걸 알아두어야 합니다. 

`multiprocessing`을 joblib을 통하지 않고 직접적으로 사용하는 자체적인 코드를 가지고 계시다면, 'forkserver' 모드를 여러분 프로그램에 전역으로(globally) 활성화하실 수 있습니다: 아래의 지침을 여러분 메인 스크립트(main script)에 삽입하세요:

```python
import multiprocessing

# 다른 imports(들여오기), 자체 코드, 데이터 불러오기, 모델 정의...

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    # 여기에 n_jobs > 1과 함께 사이킷런 유틸(utils)을 호출하세요
```

[멀티프로세싱(multiprocessing) 문서](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)에서 새로운 시작 방법에 대한 기본값을 찾으실 수 있습니다.

### 내 작업이 왜 지정된 n_jobs보다 많은 코어를 사용할까요?

왜냐하면 `n_jobs`가 `joblib`으로 병렬화된 루틴(routines)에 대한 작업(jobs)의 숫자만 제어하기 때문입니다만, 병렬 코드를 다른 소스(sources)에서 가져올 수 있습니다:

- (C나 사이썬(Cython)으로 쓰인 코드를 위해) 일부 루틴은 OpenMP와 병렬화될 수 있습니다.
- 사이킷런은 넘파이(numpy)에 많이 의존하며, MKL나 OpenBLAS, BLIS처럼 병렬 구현을 제공하는 수치 라이브러리에도 의존할 수 있습니다.

더 자세한 내용은, 저희 [병렬성 노트](computing/parallelism)를 참고해주세요.

### 어떻게 `random_state`를 전체 실행에 대해 설정하나요?

[무작위성 제어하기](common_pitfalls#무작위성-제어하기)를 참고해주세요.
