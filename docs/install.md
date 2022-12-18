원문: [Installing scikit-learn](https://scikit-learn.org/stable/install.html)

# 설치

사이킷런을 설치하려면 여러 방법이 있습니다:

- [최신 공식 릴리즈(official release)](최신-릴리즈-설치)을 설치하세요. 대부분 사용자들에게 최고의 접근법입니다. 안정된 버전과 대부분의 플랫폼(platforms)에서 사용 가능한 미리 빌드된 패키지들(pre-built packages)을 제공합니다.
- 여러분의 [운영 체제(operating system)나 파이썬 배포판(Python distribution)](사이킷런-서드-파티-배포판)에 맞게 제공되는 사이킷런 버전을 설치하세요. 사이킷런을 배포하는 운영 체제나 파이썬 배포판을 가지고 계신 분들을 위한 빠른 선택지입니다. 최신 릴리즈 버전은 제공하지 않을 수도 있습니다.
- [소스(source)에서 패키지를 빌드(build)합니다](https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge). 가장 최신의 기능들을 원하고 신제품 코드를 실행하는데 두려움이 없는 사용자들을 위해서는 최고입니다. 프로젝트에 기여하고 싶으신 사용자들에게도 필요합니다.

## 최신 릴리즈 설치

> \*역주: 번역문에는 개발 여건의 한계로 원문에 포함된 링크가 없습니다. 각 운영 체제와 패키저(packager) 조건에 따라 별도의 섹션을 새로 만들었으니 양해 부탁드립니다.

### **운영 체제** 윈도우(Windows), **패키저(packager)** pip

파이썬 3(Python 3)의 64비트(64bit) 버전을, 예를 들어 [https://www.python.org](https://www.python.org/)에서 설치하세요.  
그리고 실행하세요:

```sh
$ pip install -U scikit-learn
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ python -m pip show scikit-learn  # 어떤 버전의 사이킷런이 어디 설치되었는지 보기
$ python -m pip freeze  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 윈도우(Windows), **패키저(packager)** pip, pip virtualenv 사용

파이썬 3(Python 3)의 64비트(64bit) 버전을, 예를 들어 [https://www.python.org](https://www.python.org/)에서 설치하세요.  
그리고 실행하세요:

```sh
$ python -m venv sklearn-venv
$ sklearn-venv\Scripts\activate
$ pip install -U scikit-learn
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ python -m pip show scikit-learn  # 어떤 버전의 사이킷런이 어디 설치되었는지 보기
$ python -m pip freeze  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 윈도우(Windows), **패키저(packager)** 콘다(conda)

[아나콘다(anaconda)나 미니콘다(miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 인스톨러(installers) 또는 [미니포지(miniforge)](https://https//github.com/conda-forge/miniforge#miniforge) 인스톨러로 콘다(conda)를 설치하세요(이 중 무엇도 관리자 권한(asministrator permission)이 필요하지 않습니다).  
그리고 실행하세요:

```sh
$ conda create -n sklearn-env -c conda-forge scikit-learn
$ conda activate sklearn-env
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ conda list scikit-learn  # 어떤 버전의 사이킷런이 설치되었는지 보기
$ conda list  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 맥OS(MacOS), **패키저(packager)** pip

[홈브루(homebrew)](https://brew.sh/)로 파이썬 3(Python 3)을 설치하거나 [https://www.python.org](https://www.python.org/)에서 직접 패지키지를 설치하세요. 
그리고 실행하세요:

```sh
$ pip install -U scikit-learn
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ python -m pip show scikit-learn  # 어떤 버전의 사이킷런이 어디 설치되었는지 보기
$ python -m pip freeze  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 맥OS(MacOS), **패키저(packager)** pip, pip virtualenv 사용

[홈브루(homebrew)](https://brew.sh/)로 파이썬 3(Python 3)을 설치하거나 [https://www.python.org](https://www.python.org/)에서 직접 패지키지를 설치하세요. 
그리고 실행하세요:

```sh
$ python -m venv sklearn-venv
$ source sklearn-venv/bin/activate
$ pip install -U scikit-learn
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ python -m pip show scikit-learn  # 어떤 버전의 사이킷런이 어디 설치되었는지 보기
$ python -m pip freeze  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 맥OS(MacOS), **패키저(packager)** 콘다(conda)

[아나콘다(anaconda)나 미니콘다(miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 인스톨러(installers) 또는 [미니포지(miniforge)](https://https//github.com/conda-forge/miniforge#miniforge) 인스톨러로 콘다(conda)를 설치하세요(이 중 무엇도 관리자 권한(asministrator permission)이 필요하지 않습니다).  
그리고 실행하세요:

```sh
$ conda create -n sklearn-env -c conda-forge scikit-learn
$ conda activate sklearn-env
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ conda list scikit-learn  # 어떤 버전의 사이킷런이 설치되었는지 보기
$ conda list  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 리눅스(Linux), **패키저(packager)** pip

리눅스 배포판의 패키지 관리자(package manager)로 python3와 python3-pip를 설치하세요.  
그리고 실행하세요:

```sh
$ pip3 install -U scikit-learn
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ python -m pip show scikit-learn  # 어떤 버전의 사이킷런이 어디 설치되었는지 보기
$ python -m pip freeze  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 리눅스(Linux), **패키저(packager)** pip, pip virtualenv 사용

리눅스 배포판의 패키지 관리자(package manager)로 python3와 python3-pip를 설치하세요.  
그리고 실행하세요:

```sh
$ python -m venv sklearn-venv
$ sklearn-venv\Scripts\activate
$ pip install -U scikit-learn
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ python -m pip show scikit-learn  # 어떤 버전의 사이킷런이 어디 설치되었는지 보기
$ python -m pip freeze  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

### **운영 체제** 리눅스(Linux), **패키저(packager)** 콘다(conda)

[아나콘다(anaconda)나 미니콘다(miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 인스톨러(installers) 또는 [미니포지(miniforge)](https://https//github.com/conda-forge/miniforge#miniforge) 인스톨러로 콘다(conda)를 설치하세요(이 중 무엇도 관리자 권한(asministrator permission)이 필요하지 않습니다).  
그리고 실행하세요:

```sh
$ conda create -n sklearn-env -c conda-forge scikit-learn
$ conda activate sklearn-env
```

설치를 확인하시려면 다음을 사용하실 수 있습니다

```sh
$ conda list scikit-learn  # 어떤 버전의 사이킷런이 설치되었는지 보기
$ conda list  # 활성화된 가상환경에 모든 패키지가 설치되었는지 보기
$ python -c "import sklearn; sklearn.show_versions()"
```

다른 패키지와의 잠재적인 충돌(conflicts)을 피하기 위해 [가상 환경(virtual environment, venv)](https://docs.python.org/3/tutorial/venv.html)이나 [콘다 환경(conda environment)](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)을 사용하는 것을 강력하게 추천한다는 것을 참고하세요.

이러한 격리된 환경은 전에 설치된 어떠한 파이썬 패키지와 상관 없이, pip나 콘다 및 그 의존 항목들(dependencies)로 사이킷런의 특정 버전을 설치할 수 있게 합니다. 특히 리눅스에서는 배포판의 패키지 관리자(apt, dnf, pacman...)가 관리하는 패키지들과 함께 pip 패키지를 설치하지 않는 것이 좋습니다.

넘파이(NumPy)나 사이파이(SciPy)를 아직 설치하지 않았다면, 콘다나 pip로 설치할 수 있습니다. pip를 쓴다면, (라즈베리 파이(Raspberry Pi)에서의 리눅스처럼) 특정 구성의 운영 체제와 하드웨어(hardware)를 사용할 때, *바이너리 휠(binary wheels)*을 사용하는지와 넘파이와 사이파이가 소스에서 다시 컴파일(recompile)되지 않는지 확인하세요.

사이킷런 플로팅(plotting) 기능(예시로 "plot_"으로 시작하는 함수나 "Display"로 끝나는 클래스 등)은 맷플롯립(Matplotlib)이 필요합니다. 예제는 맷플롯립이 필요하고 몇몇 예제는 사이킷이미지(scikit-image), 판다스(pandas), 또는 씨본(seaborn)이 필요합니다. 사이킷런 의존 항목들의 최소 버전이 그 목적과 함께 아래에 나열되어 있습니다.

|의존 항목|최소 버전|목적|
|---|---|---|
|numpy|1.17.3|빌드, 설치|
|scipy|1.3.2|빌드, 설치|
|joblib|1.1.1|설치|
|threadpoolctl|2.0.0|설치|
|cython|0.29.24|빌드|
|matplotlib|3.1.3|벤치마크(benchmark), 문서, 예제, 테스트|
|scikit-image|0.16.2|문서, 예제, 테스트|
|pandas|1.0.5|벤치마크, 문서, 예제, 테스트|
|seaborn|0.9.0|문서, 예제|
|memory_profiler|0.57.0|벤치마크, 문서|
|pytest|5.3.1|테스트|
|pytest-cov|2.9.0|테스트|
|flake8|3.8.2|테스트|
|black|22.3.0|테스트|
|mypy|0.961|테스트|
|pyamg|4.0.0|테스트|
|sphinx|4.0.1|문서|
|sphinx-gallery|0.7.0|문서|
|numpydoc|1.2.0|문서, 테스트|
|Pillow|7.1.2|문서|
|pooch|1.6.0|문서, 예제, 테스트|
|sphinx-prompt|1.3.0|문서|
|sphinxext-opengraph|0.4.2|문서|
|plotly|5.10.0|문서, 예제|
|conda-lock|1.2.1|유지보수|

> **경고:** 사이킷런 0.20은 파이썬 2.7과 파이썬 3.4를 지원하는 마지막 버전이었습니다. 사이킷런 0.21은 파이썬 3.5-3.7을 지원했습니다. 사이킷런 0.22는 파이썬 3.5-3.8을 지원했습니다. 사이킷런 0.23 - 0.24는 파이썬 3.6 이상이 필요합니다. 사이킷런 1.0은 파이썬 3.7-3.10을 지원했습니다. 사이킷런 1.1 이상은 파이썬 3.8 이상이 필요합니다.

> **참고:** 파이파이(PyPy)를 설치하려면, PyPy3-v5.10+, 넘파이 1.14.0+, 그리고 사이파이 1.1.0+이 필요합니다.

## 애플 실리콘 M1(Apple Silicon M1) 하드웨어에 설치

최근 소개된 `macos/arm64` 플랫폼(가끔 `macos/aarch64`라고 하는)은 빌드 구성과 자동화를 업그레이드(upgrade)하고 적절히 지원하기 위해서 오픈소스 커뮤니티(open source community)가 필요합니다.

작성 시점에는(2021년 1월), 이 하드웨어에서 사이킷런의 작업 설치를 하기 위한 유일한 방법은 콘다 포지(conda-forge) 배포판에서 사이킷런을 설치하는 것인데, 예를 들어 미니포지(miniforge) 인스톨러를 사용한다면:

[https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge)

다음의 이슈(issue)는 PyPI에서 pip로 사이킷런을 설치할 수 있도록 하는 과정을 추적하고 있습니다:

[https://github.com/scikit-learn/scikit-learn/issues/19137](https://github.com/scikit-learn/scikit-learn/issues/19137)

## 사이킷런 서드 파티 배포판

어떤 서드 파티(third-party) 배포판은 그들의 패키지 관리 시스템(package management systems)에 통합된 사이킷런 버전을 제공합니다.

이는 통합판이 사이킷런이 필요로 하는 의존 항목들(넘파이, 사이파이)을 자동적으로 설치하는 기능을 포함하므로, 사용자들이 더 쉽게 설치와 업그레이드를 할 수 있게끔 해줍니다.

다음은 자체적인 사이킷런 버전을 제공하는 OS(운영 체제)와 파이썬 배포판의 완전하지는 않은 목록입니다.

### 알파인 리눅스(Alpine Linux)

알파인 리눅스 패키지는 [공식 저장소(repositories)](https://pkgs.alpinelinux.org/packages?name=py3-scikit-learn)를 통해 파이썬을 위한 `py3-scikit-learn`으로 제공됩니다. 다음의 명령을 입력해 설치할 수 있습니다:

```sh
sudo apk add py3-scikit-learn
```

### 아크 리눅스(Arch Linux)

아크 리눅스 패키지는 [공식 저장소(repositories)](https://www.archlinux.org/packages/?q=scikit-learn)를 통해 파이썬을 위한 `python-scikit-learn`으로 제공됩니다. 다음의 명령을 입력해 설치할 수 있습니다:

```sh
sudo pacman -S python-scikit-learn
```

### 데비안/우분투(Debian/Ubuntu)

데비안/우분투 패키지는 `python3-sklearn`(파이썬 모듈들), `python3-sklearn-lib`(저수준 구현과 바인딩(bindings)), `python3-sklearn-doc`(문서)의 세 개의 다른 패키지에 분할되어 있습니다. 오직 파이썬 3 버전만 데비안 버스터(Debian Buster, 더 최신의 데비안 배포판)에서 사용 가능합니다. 패키지는 `apt-get`으로 설치할 수 있습니다:

```sh
sudo apt-get install python3-sklearn python3-sklearn-lib python3-sklearn-doc
```

### 페도라(Fedora)

페도라 패키지는 파이썬 3 버전을 위한 `python3-scikit-learn`이며, 페도라30(Fedora30)에서 유일하게 사용 가능한 패키지입니다. `dnf`로 설치할 수 있습니다:

```sh
sudo dnf install python3-scikit-learn
```

### 넷BSD(NetBSD)

사이킷런은 [pkgsrc-wip](http://pkgsrc-wip.sourceforge.net/)으로 사용 가능합니다:

[http://pkgsrc.se/math/py-scikit-learn](http://pkgsrc.se/math/py-scikit-learn)

### 맥 OSX를 위한 맥포트(MacPorts)

맥포트 패키지 이름은 `py<XY>-scikits-learn`이며, `XY`는 파이썬 버전을 나타냅니다. 다음 명령을 입력해 설치할 수 있습니다:

```sh
sudo port install py39-scikit-learn
```

### 모든 지원되는 플랫폼을 위한 아나콘다(Anaconda)와 엔쏘우트 배포 관리자(Enthought deployment manager)

[아나콘다](https://www.anaconda.com/download)와 [엔쏘우트 배포 관리자](https://assets.enthought.com/downloads/)는 사이킷런을 포함한 과학적인 파이썬 라이브러리의 거대한 세트를 윈도우, 맥 OSX, 그리고 리눅스를 위해 담고 있습니다.

아나콘다는 무료 배포판의 일부로 사이킷런을 제공합니다.

### 인텔 콘다 채널(Intel conda channel)

인텔은 사이킷런을 담은 전용 콘다 채널을 유지보수합니다.

```sh
conda install -c intel scikit-learn
```

이 버전의 사이킷런에는 일부 일반적인 추정기들에 대한 대체 솔버(solvers)가 있습니다. 이 솔버들은 DAAL C++ 라이브러리에서 왔고 다중 코어(multi-core) 인텔 CPU에 최적화되어 있습니다.

솔버들이 기본적으로 사용 가능하지는 않음을 인지하고, 더 자세한 설명은 [daal4py](https://intelpython.github.io/daal4py/sklearn.html) 문서를 참고해주세요.

표준 사이킷런 솔버와의 호환성(compatibility)은 [https://github.com/IntelPython/daal4py](https://github.com/IntelPython/daal4py)에 보고되어 있듯이 자동 연속 통합(continuous integration)을 통해 전체 사이킷런 테스트 수트(test suite)를 실행하며 확인되고 있습니다.

### 윈도우를 위한 윈파이썬(WinPython)

[윈파이썬](https://winpython.github.io/) 프로젝트는 사이킷런을 추가 플러그인(plugin)으로 배포합니다.

## 문제 해결

### 윈도우의 파일 경로 길이 제한으로 인한 오류

만약 파이썬이 사용자 홈 디렉토리(home directory)의 `AppData` 폴더처럼 중첩된 위치에 설치되어 있다면 윈도우의 기본 경로 크기 제한에 도달하여 pip가 패키지 설치에 실패할 수 있습니다, 예를 들어:

```sh
C:\Users\username>C:\Users\username\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install scikit-learn
Collecting scikit-learn
...
Installing collected packages: scikit-learn
ERROR: Could not install packages due to an EnvironmentError: [Errno 2] No such file or directory: 'C:\\Users\\username\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\sklearn\\datasets\\tests\\data\\openml\\292\\api-v1-json-data-list-data_name-australian-limit-2-data_version-1-status-deactivated.json.gz'
```

이 경우 `regedit` 도구로 윈도우 레지스트리(registry) 안의 제한을 풀 수 있습니다:

1. 윈도우 시작 메뉴(start menu)에서 "regedit"을 입력하고 `regedit`을 실행하세요.
2. `Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem` 키(key)로 가세요.
3. 그 키의 `LongPathsEnabled` 속성 값을 편집해 1로 설정하세요.
4. (이전에 중단된 설치는 무시하고) 사이킷런을 다시 설치하세요.

```sh
pip install --exists-action=i scikit-learn
```
