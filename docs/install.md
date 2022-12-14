원문: [Installing scikit-learn](https://scikit-learn.org/stable/install.html)

# 설치

사이킷런을 설치하려면 여러 방법이 있습니다:

- [최신 공식 릴리즈(official release)](최신-릴리즈-설치)을 설치하세요. 대부분 사용자들에게 최고의 접근법입니다. 안정된 버전과 대부분의 플랫폼(platforms)에서 사용 가능한 미리 빌드된 패키지들(pre-built packages)을 제공합니다.
- 여러분의 [운영 체제(operating system)나 파이썬 배포판(Python distribution)](사이킷런-서드-파티-배포판)에 맞게 제공되는 사이킷런 버전을 설치하세요. 사이킷런을 배포하는 운영 체제나 파이썬 배포판을 가지고 계신 분들을 위한 빠른 선택지입니다. 최신 릴리즈 버전은 제공하지 않을 수도 있습니다.
- [소스(source)에서 패키지를 빌드(build)합니다](../../developers/advanced_installation#install-bleeding-edge). 가장 최신의 기능들을 원하고 신제품 코드를 실행하는데 두려움이 없는 사용자들을 위해서는 최고입니다. 프로젝트에 기여하고 싶으신 사용자들에게도 필요합니다.

### 최신 릴리즈 설치

> \*역주: 번역문에는 개발 여건의 한계로 원문에 포함된 링크가 없습니다. 각 운영 체제와 패키저(packager) 조건에 따라 별도의 섹션을 새로 만들었으니 양해 부탁드립니다.

<details>
<summary>**운영 체제** 윈도우(Windows), **패키저(packager)** pip</summary>

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

</details>

<details>
<summary>**운영 체제** 윈도우(Windows), **패키저(packager)** pip, pip virtualenv 사용</summary>

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

</details>

<details>
<summary>**운영 체제** 윈도우(Windows), **패키저(packager)** 콘다(conda)</summary>

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

</details>

<details>
<summary>**운영 체제** 맥OS(MacOS), **패키저(packager)** pip</summary>

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

</details>

<details>
<summary>**운영 체제** 맥OS(MacOS), **패키저(packager)** pip, pip virtualenv 사용</summary>

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

</details>

<details>
<summary>**운영 체제** 맥OS(MacOS), **패키저(packager)** 콘다(conda)</summary>

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

</details>

<details>
<summary>**운영 체제** 리눅스(Linux), **패키저(packager)** pip</summary>

리눅스 배포판의 패키지 매니저(package manager)로 python3와 python3-pip를 설치하세요.  
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

</details>

<details>
<summary>**운영 체제** 리눅스(Linux), **패키저(packager)** pip, pip virtualenv 사용</summary>

리눅스 배포판의 패키지 매니저(package manager)로 python3와 python3-pip를 설치하세요.  
그리고 실행하세요:

```sh
$ python -m venv sklearn-venv
$ sklearn-venv\Scripts\activate
$ pip install -U scikit-learn
```

</details>

<details>
<summary>**운영 체제** 리눅스(Linux), **패키저(packager)** 콘다(conda)</summary>

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

</details>

