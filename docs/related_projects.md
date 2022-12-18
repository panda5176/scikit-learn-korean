원문: [Related Projects](https://scikit-learn.org/stable/related_projects.html)

# 관련 프로젝트

사이킷런 추정기 API를 구현하는 프로젝트는, 추정기 테스트와 문서화를 위한 모범 사례를 용이하도록 하는 [사이킷런컨트립(scikit-learn-contrib) 템플릿(template)](https://github.com/scikit-learn-contrib/project-template)을 사용하게끔 권장합니다. 또한 사이킷럽컨트립 깃헙(GitHub) 조직은 이 템플릿을 준수하는 고품질의 저장소 컨트리뷰션(contributions of repositories)을 허용합니다.

아래는 자매 프로젝트(sister-projects), 확장 그리고 도메인 특이적인 패키지들의 목록입니다.

## 상호 운용성과 프레임워크 개선

이 도구들은 다른 기술과의 사용을 위해 사이킷런을 조정하거나 사이킷런 추정기의 기능성을 개선합니다.

**데이터 형식**

- [Fast svmlight / libsvm file loader](https://github.com/mblondel/svmlight-loader) 파이썬(Python)을 위한 빠르고 메모리 효율적인 svmlight / libsvm 파일 로더(loader)입니다.
- [sklearn_pandas](https://github.com/paulgb/sklearn-pandas/)는 사이킷런 파이프라인(pipelines)과 판다스(pandas) 데이터 프레임(data frame)을 전용 번환기로 연결합니다.
- [sklearn_xarray](https://github.com/phausamann/sklearn-xarray/)는 사이킷런 추정기와 xarray 자료 구조(data structures) 간 호환성을 제공합니다.

**오토 ML(Auto-ML)**

- [auto-sklearn](https://github.com/automl/auto-sklearn/) 자동화된 기계 학습(machine learning) 도구모음(toolkit)과 사이킷런 추정기의 간편한(drop-in) 대체품입니다.
- [autoviml](https://github.com/AutoViML/Auto_ViML/) 한 줄의 코드로 자동으로 다중 기계 학습 모델을 구축합니다. 데이터 전처리 없이 사이킷런 모델을 사용하기 위한 더 빠른 방법으로 설계되었습니다.
- [TPOT](https://github.com/rhiever/tpot) 추정기뿐 아니라 데이터와 특성 전처리기를 포함하는 기계 학습 파이프라인을 설계하기 위해, 연속된 사이킷런 연산자를 최적화하는 자동화된 기계 학습 도구모음입니다. 사이킷런 추정기의 간편한 대체품처럼 작동합니다.
- [Featuretools](https://github.com/alteryx/featuretools) 자동화된 특성 공학(feature engineering)을 수행하는 프레임워크(framework)입니다. 시간(temporal) 및 관계형(relational) 데이터셋을 기계 학습을 위한 특성 행렬(feature matrices)로 변환하는데 사용할 수 있습니다.
- [Neuraxle](https://github.com/Neuraxio/Neuraxle) 정돈된 파이프라인을 구축하기 위한 라이브러리로, 연구, 개발, 그리고 기계 학습 응용 프로그램의 배포를 용이하게 하는 올바른 추상화(abstractions)를 제공합니다. 심층 학습(deep learning) 프레임워크 및 사이킷런 API와 호환되며, 미니배치(minibatches)를 스트리밍(stream)하고, 데이터 체크포인트(data checkpoints)를 사용하며, 펑키한(funky) 파이프라인을 구축하며, 맞춤형 단계별 저장기(savers)로 모델을 직렬화(serialize)할 수 있습니다.
- [EvalML](https://github.com/alteryx/evalml)은 도메인 특이적인(domain-specific) 목적 함수(objective functions)를 사용하여 기계 학습 파이프라인을 구축, 최적화, 그리고 평가하는 오토 ML 라이브러리입니다. 하나의 API 아래에 다양한 모델링 라이브러리를 통합하고, 만드는 객체는 사이킷런 호환 API를 사용합니다.

**실험 프레임워크**

- [Neptune](https://neptune.ai/) 실험을 많이 하는 팀을 위해 구축된, ML옵스(MLOps)를 위한 메타데이터(metadata) 저장소입니다. 여러분의 모든 모델 구축 메타데이터를 기록, 저장, 표시, 조직, 비교, 쿼리(query)할 단일한 위치를 제공합니다.
- [Sacred](https://github.com/IDSIA/Sacred) 실험을 구성, 조직, 기록, 재현하는데 도움을 줄 도구입니다.
- [REP](https://github.com/yandex/REP) 일관적이고 재현 가능한 방식으로 데이터 주도적 연구를 수행하기 위한 환경(environment)입니다.
- [Scikit-Learn Laboratory](https://skll.readthedocs.io/en/latest/index.html) 기계 학습 실험을 다중 학습자(learners)와 거대 특성 세트(feature sets)로 쉽게 실행할 수 있게 해주는 사이킷런 주변의 명령줄 래퍼(command-line wrapper)입니다.

**모델 검사와 시각화**

- [dtreeviz](https://github.com/parrt/dtreeviz/) 의사결정나무(decision tree) 시각화와 모델 해석을 위한 파이썬 라이브러리입니다.
- [eli5](https://github.com/TeamHG-Memex/eli5/) 기계 학습 모델의 디버깅/검사(debugging/inspecting)와 예측 설명을 위한 라이브러리입니다.
- [mlxtend](https://github.com/rasbt/mlxtend) 모델 시각화 유틸리티(utilities)를 포함합니다.
- [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) 사이킷런 추정기의 시각적인 특성 분석, 모델 선택, 평가, 진단을 위한 맞춤형 맷플롯립(matplotlib) 시각화기(visualizers) 모음입니다.

**모델 선택**

- [scikit-optimize](https://scikit-optimize.github.io/) (매우) 비싸고 잡음이 많은(noisy) 블랙박스(black-box) 함수들(functions)을 최소화하는 라이브러리입니다. 연속적인 모델 기반 최적화를 위한 다양한 메서드(methods)를 구현하고, `GridSearchCV`나 `RandomizedSearchCV` 전략을 사용하는 교차 검증된(cross-validated) 매개변수 탐색(parameter search)를 위하여 이들의 대체제를 포함합니다.
- [sklearn-deap](https://github.com/rsteca/sklearn-deap) 사이킷런의 격자 탐색(gridsearch) 대신 진화 알고리즘(evolutionary algorithms)을 사용합니다.

**운영(production)을 위한 모델 내보내기**

- [sklearn-onnx](https://github.com/onnx/sklearn-onnx) 교환(interchange)과 예측을 위해 많은 사이킷런 파이프라인을 [ONNX](https://onnx.ai/)로 직렬화합니다.
- [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) 라이브러리의 도움으로 다양한 사이킷런 추정기와 변환기를 PMML로 직렬화합니다.
- [sklearn-porter](https://github.com/nok/sklearn-porter) 훈련된 사이킷런 모델을 C, Java(자바), Javascript(자바스크립트) 등으로 트랜스파일(transpile)합니다.
- [m2cgen](https://github.com/BayesWitnesses/m2cgen) 많은 사이킷런 추정기를 포함하는 훈련된 기계 학습 모델들을 C, Java, Go(고), R, PHP, Dart(다트), Haskell(하스켈), Rust(러스트)와 많은 다른 프로그래밍 언어로 트랜스파일하게 해주는 경량 라이브러리입니다.
- [treelite](https://treelite.readthedocs.io/) 예측 지연(latency)를 최소화하기 위해 트리-기반(tree-based) 앙상블(ensemble) 모델을 C 코드로 컴파일(compiles)합니다.

## 다른 추정기들과 작업들

모든 것들이 중앙 사이킷런 프로젝트를 위해 속하거나 충분히 성숙한 것은 아닙니다. 다음은 추가적인 학습 알고리즘, 인프라 구조(infrastructures)와 작업들을 위해 사이킷런과 유사한 인터페이스(interfaces)를 제공하는 프로젝트들입니다.

**구조화된 학습(structured learning)**

- [tslearn](https://github.com/tslearn-team/tslearn) 시계열(time series)을 위한 기계 학습 라이브러리로, 군집화(clustering), 분류, 회귀를 위한 전용 모델뿐 아니라 전처리와 특성 추출을 위한 도구도 제공합니다.
- [sktime](https://github.com/alan-turing-institute/sktime) 시계열 분류/회귀와 (지도/패널(panel)) 예측(forecasting)을 포함하는 시계열 기계 학습을 위한 사이킷런 호환 도구상자입니다.
- [HMMLearn](https://github.com/hmmlearn/hmmlearn) 이전에는 사이킷런의 일부였던 은닉 마르코프 모델(hidden markov models)의 구현체입니다.
- [PyStruct](https://pystruct.github.io/) 일반적인 조건부 임의장(conditional random field)과 구조화된 예측입니다.
- [pomegranate](https://github.com/jmschrei/pomegranate) 은닉 마르코프 코델에 중점을 둔 파이썬을 위한 확률적 모델링(probabilistic modelling)입니다.
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) (사이킷런과 비슷한 API의 [CRFsuite](http://www.chokkan.org/software/crfsuite/) 래퍼) 선형 연쇄(linear-chain) 조건부 임의장입니다.

**심층 신경망(deep neural networks) 등**

- [nolearn](https://github.com/dnouri/nolearn) 기존 신경망 라이브러리에 대한 수많은 래퍼와 추상화입니다.
- [Keras](https://www.tensorflow.org/api_docs/python/tf/keras) 사이킷런에서 영감을 받은 API와 함께 하는 텐서플로우(TensorFlow) 고수준 API입니다.
- [lasagne](https://github.com/Lasagne/Lasagne) 테아노(Theano) 신경망을 구축하고 훈련하기 위한 경량 라이브러리입니다.
- [skorch](https://github.com/dnouri/skorch) 파이토치(PyTorch)를 감싸는 사이킷런 호환 가능한 신경망 라이브러리입니다.
- [scikeras](https://github.com/adriangb/scikeras) 케라스(Keras)가 사이킷런과 인터페이스(interface)하기 위한 래퍼를 제공합니다. SciKeras는 `tk.Keras.wrappers.scikit_learn`의 계승자입니다.

**연합 학습(federated learning)**

- [Flower](https://flower.dev/) 어떠한 워크로드(workload), ML 프레임워크, 프로그래밍 언어도 연합할 수 있는 통합 접근법의 친근한 연합 학습 프레임워크입니다.

**넓은 범위**

- [mlxtend](https://github.com/rasbt/mlxtend) 모델 시각화 유틸리티뿐 아니라 수많은 추가적인 추정기를 포함합니다.
- [scikit-lego](https://github.com/koaning/scikit-lego) 실제 산업계 작업을 해결하는데 초점을 맞춘, 수많은 사이킷런 호환 가능한 자체 변환기, 모델, 측정법입니다.

**다른 회귀와 분류**

- [xgboost](https://github.com/dmlc/xgboost) 최적화된 경사 부스트(gradient boosted) 의사결정나무 라이브러리입니다.
- [ML-Ensemble](https://mlens.readthedocs.io/) 일반화된 앙상블 학습(스태킹(stacking), 블렌딩(blending), 서브셈블(subsemble), 딥 앙상블(deep ensemble) 등)입니다.
- [lightning](https://github.com/scikit-learn-contrib/lightning) 빠른 최신 선형 모델 솔버(solvers)(SDCA, AdaGrad, SVRG, SAG 등)입니다.
- [py-earth](https://github.com/scikit-learn-contrib/py-earth) 다변량(multivariate) 적응형 회귀 스플라인(splines)입니다.
- [Kernel Regression](https://github.com/jmetzen/kernel_regression) 자동 대역폭(bandwidth) 선택을 통한 나다라야-왓슨 커널 회귀(Nadaraya-Watson kernel regression) 구현체입니다.
- [gplearn]https://github.com/trevorstephens/gplearn) 기호 회귀(symbolic regression) 작업을 위한 유전 프로그래밍(genetic programming)입니다.
- [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) 레이블 공간 조작(label space manipulation)에 초점을 둔 다중 레이블 분류입니다.
- [seglearn](https://github.com/dmbee/seglearn) 슬라이딩 윈도우 분할(sliding window segmentation)을 사용한 시계열과 시퀀스(sequence) 학습입니다.
- [libOPF](https://github.com/jppbsi/LibOPF) 포레스트(forest) 분류기 최적 경로입니다.
- [fastFM](https://github.com/ibayer/fastFM) 사이킷런과 호환 가능한 빠른 인수분해 기계(factorization machine) 구현체입니다.

**분해(decomposition)와 군집화(clustering)**

- [Ida](https://github.com/lda-project/lda/) 실제 사후 분포(posterior distribution)에서 표집(sample)하기 위해 [깁스 샘플링(Gibbs sampling)](https://en.wikipedia.org/wiki/Gibbs_sampling)을 사용하는 잠재 디리클레 할당(latent Dirichlet allocation)의 빠른 사이썬(Cython) 구현체입니다. (사이킷런 [`LatentDirichletAllocation`](modules/generated/sklearn.decomposition.LatentDirichletAllocation) 구현은 주제(topic) 모델의 사후 분포로부터 다루기 쉬운 근사치를 표집하기 위해서 [변분 추론(variational inference)](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)를 사용합니다.)
- [kmodes](https://github.com/nicodv/kmodes) 범주형 데이터를 위한 k 모드(k-modes) 군집화 알고리즘이나 그 변형들입니다.
- [hdbscan](https://github.com/scikit-learn-contrib/hdbscan) 로버스트(robust)한 가변 밀도 군집화(variable density clustering)을 위한 HDBSCAN과 로버스트 단일 연결(single linkage) 군집화 알고리즘입니다.
- [spherecluster](https://github.com/clara-labs/spherecluster) 단위 초구(hypersphere) 위 데이터를 위한 K 평균(spherical K-means)와 본 마이즈 피셔 군집화 루틴(von Mises Fisher clustering routines)의 혼합입니다.

**전처리**

- [categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding) 사이킷런 호환 가능한 범주형 변수 부호화기(encoders) 라이브러리입니다.
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) 과소 및 과대 표집(under- and over-sample) 데이터셋을 위한 다양한 메서드들입니다.
- [Feature-engine](https://github.com/solegalli/feature_engine) 결측값 대치(missing data imputation), 범주형 부호화, 변수 변환, 이산화(discretization), 이상값(outlier) 처리 등을 위한 사이킷런 호환 가능한 변환기 라이브러리입니다. Feature-engine은 선택된 변수 집단에 전처리 단계를 적용할 수 있도록 하고 사이킷런 파이프라인와 완벽하게 호환 가능합니다.

**위상(topological) 데이터 분석**

- [giotto-tda](https://github.com/giotto-ai/giotto-tda) 사이킷런 호환 가능한 API 제공을 목표로 하는 [위상 데이터 분석(topological data analysis)](https://en.wikipedia.org/wiki/Topological_data_analysis)을 위한 라이브러리입니다. 데이터 입력(점구름(point clouds), 그래프, 시계열, 이미지)을 위상 요약(topological summaries) 계산에 적합한 형태로 변환하는 도구를 제공하며, 사이킷런의 다른 특성 추출 메서드와 함께 사용할 수 있는, 위상 원점(topological origin)의 스칼라(scalar) 특성 집합을 추출하는데 사용할 구성 요소들도 제공합니다.

## 파이썬 통계적 학습

데이터 분석과 기계 학습에 유용한 다른 패키지들입니다.

- [Pandas](https://pandas.pydata.org/) 이종(heterogeneous) 및 열(columnar) 데이터, 관계형 쿼리, 시계열과 기초 통계 작업을 위한 도구입니다.
- [statsmodels](https://www.statsmodels.org/) 통계 모델 추론과 분석입니다. 사이킷런에 비해 예측은 덜 집중하고 통계 검정(tests)에 초점을 더 맞춥니다.
- [PyMC](https://pymc-devs.github.io/pymc/) 베이지안(Bayesian) 통계 모델과 적합 알고리즘입니다.
- [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/) 맷플롯립(matplotlib)에 기반한 시각화 라이브러리입니다. 매력적인 통계 그래픽을 그리기 위한 고수준 인터페이스를 제공합니다.
- [scikit-survival](https://scikit-survival.readthedocs.io/) (생존 분석(survival analysis)이라고도 하는) 중도절단된(censored) 시간 이벤트(time-to-event) 데이터에서 학습하는 모델을 구현한 라이브러리입니다. 모델은 사이킷런과 완벽하게 호환됩니다.

### 추천 엔진(recommendation engine) 패키지

- [implicit](https://github.com/benfred/implicit) 암시적 피드백(implicit feedback) 데이터셋을 위한 라이브러리입니다.
- [lightfm](https://github.com/lyst/lightfm) 복합 추천 시스템의 파이썬/사이썬 구현체입니다.
- [OpenRec](https://github.com/ylongqi/openrec) 텐서플로우(TensorFlow) 기반 신경망에서 영감을 받은 추천 알고리즘입니다.
- [Spotlight](https://github.com/maciejkula/spotlight) 심층 추천 모델(deep recommender model)의 파이토치(Pytorch) 기반 구현체입니다.
- [Surprise Lib](http://surpriselib.com/) 명시적 피드백(explicit feedback) 데이터셋을 위한 라이브러리입니다.

### 도메인 특이적(domain specific) 패키지

- [scikit-network](https://scikit-network.readthedocs.io/) 그래프 기계 학습입니다.
- [scikit-image](https://scikit-image.org/) 파이썬 이미지 처리와 컴퓨터 비전(vision)입니다.
- [Natural language toolkit (nltk)](https://www.nltk.org/) 자연어(natural language) 처리와 약간의 기계 학습입니다.
- [gensim](https://radimrehurek.com/gensim/) 주제 모델링(topic modelling), 문서 인덱싱(document indexing) 그리고 유사도 검색(similarity retrieval)을 위한 라이브러리입니다.
- [NiLearn](https://nilearn.github.io/) 신경 이미징(neuro-imaging)을 위한 기계 학습입니다.
- [AstroML](https://www.astroml.org/) 천문학(astronomy)을 위한 기계 학습입니다.
- [MSMBuilder](http://msmbuilder.org/) 단백질 구조 역학(protein conformational dynamics) 시계열을 위한 기계 학습입니다.

## 사이킷런 문서의 번역

번역의 목적은 영어가 아닌 언어들로 쉽게 읽고 이해하는 것입니다. 이는 영어를 이해하지 못하거나 해석에 의문이 드는 사람들을 돕는 것을 목적으로 합니다. 또한, 어떤 사람들은 문서를 그들의 모국어로 읽기를 선호하지만, 공식 문서는 영어 문서뿐이라는 점을 명심해주세요 [[1]](#1).

그러한 번역의 노력은 공동체 주도적(community initiatives)이며 저희는 그에 대한 어떠한 통제권도 없습니다. 만약 여러분이 번역에 기여하거나 문제점을 보고하고 싶으시다면, 번역문의 저자들에게 연락해주세요. 배포를 개선하고 공동체적 노력을 촉진하기 위해 일부 사용 가능한 번역문들을 여기에 링크해두었습니다.

- [중국어 번역](https://sklearn.apachecn.org/)([소스](https://github.com/apachecn/sklearn-doc-zh))
- [페르시아어 번역](https://sklearn.ir/)([소스](https://github.com/mehrdad-dev/scikit-learn))
- [스페인어 번역](https://qu4nt.github.io/sklearn-doc-es/)([소스](https://github.com/qu4nt/sklearn-doc-es))

###### [1]  
[리눅스 문서 면책 조항(linux documentation Disclaimer)](https://www.kernel.org/doc/html/latest/translations/index.html#disclaimer)을 따름
