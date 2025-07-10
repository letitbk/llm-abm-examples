# 한글 폰트 설정 가이드 (Korean Font Setup Guide)

## 개요

`schelling_llm.py`와 `schelling_traditional.py` 파일에서 한글 문자가 제대로 표시되지 않는 문제를 해결하기 위한 가이드입니다.

## 해결된 문제

### 1. 터미널 출력 문제
- **문제**: 한글 문자가 터미널에서 깨져서 표시됨
- **해결**: UTF-8 인코딩 설정 및 시스템별 한글 폰트 자동 감지

### 2. matplotlib 그래프 문제
- **문제**: 그래프 제목, 축 라벨, 범례 등에서 한글이 네모 박스로 표시됨
- **해결**: 운영체제별 한글 폰트 자동 설정

## 지원되는 운영체제 및 폰트

### macOS
- **AppleGothic** (기본 선택)
- Apple SD Gothic Neo
- Nanum Gothic
- Malgun Gothic
- Arial Unicode MS

### Windows
- **Malgun Gothic** (기본 선택)
- Nanum Gothic
- Arial Unicode MS
- Gulim
- Dotum

### Linux
- **Nanum Gothic** (기본 선택)
- Noto Sans CJK KR
- UnDotum
- Liberation Sans
- DejaVu Sans

## 사용 방법

### 1. 자동 설정 (권장)
```python
# LLM 모델의 경우
from schelling_llm import setup_korean_font

# 전통적 모델의 경우
from schelling_traditional import setup_korean_font

# 한글 폰트 자동 설정
font_name = setup_korean_font()
print(f"설정된 폰트: {font_name}")
```

### 2. 수동 설정
```python
import matplotlib.pyplot as plt

# 특정 폰트 직접 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'Nanum Gothic'  # Linux

plt.rcParams['axes.unicode_minus'] = False
```

## 테스트 방법

### 1. 한글 폰트 테스트 스크립트 실행
```bash
python test_korean_font.py
```

### 2. 주요 모듈 테스트

**LLM 모델 테스트:**
```bash
python -c "
from schelling_llm import setup_korean_font, LLMAgent
font = setup_korean_font()
print('한글 테스트: 안녕하세요')
agent = LLMAgent(1, 5, 5)
print(f'그룹: {agent.group_characteristics[\"name\"]}')
"
```

**전통적 모델 테스트:**
```bash
python -c "
from schelling_traditional import setup_korean_font, initialize_grid, calculate_segregation_index
font = setup_korean_font()
print('한글 테스트: 안녕하세요')
grid = initialize_grid(10, 10, 0.3)
print(f'분리지수: {calculate_segregation_index(grid):.3f}')
"
```

### 3. 그래프 테스트
```python
import matplotlib.pyplot as plt
# from schelling_llm import setup_korean_font  # LLM 모델용
from schelling_traditional import setup_korean_font  # 전통적 모델용

setup_korean_font()
plt.figure(figsize=(8, 6))
plt.title('한글 제목 테스트')
plt.xlabel('X축 라벨')
plt.ylabel('Y축 라벨')
plt.plot([1, 2, 3], [1, 4, 2], label='한글 범례')
plt.legend()
plt.show()
```

## 문제 해결

### 1. 한글이 네모 박스로 표시되는 경우

**원인**: 시스템에 적절한 한글 폰트가 설치되지 않음

**해결책**:
- **macOS**: 시스템 기본 폰트 사용 (자동 해결)
- **Windows**:
  ```bash
  # 나눔고딕 폰트 설치
  # https://hangeul.naver.com/2017/nanum 에서 다운로드
  ```
- **Linux**:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install fonts-nanum

  # CentOS/RHEL
  sudo yum install naver-nanum-fonts

  # Arch Linux
  sudo pacman -S noto-fonts-cjk
  ```

### 2. 터미널에서 한글이 깨지는 경우

**원인**: 터미널 인코딩 설정 문제

**해결책**:
```bash
# 터미널 환경변수 설정
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8

# 또는 영어 환경에서 UTF-8 사용
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### 3. 특정 폰트 강제 사용

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 사용 가능한 폰트 목록 확인
fonts = [f.name for f in fm.fontManager.ttflist]
korean_fonts = [f for f in fonts if 'Gothic' in f or 'Nanum' in f]
print("사용 가능한 한글 폰트:", korean_fonts)

# 특정 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # 원하는 폰트명 입력
```

## 추가 정보

### 폰트 캐시 새로고침
```python
import matplotlib.font_manager as fm
fm._rebuild()  # 폰트 캐시 새로고침
```

### 현재 설정 확인
```python
import matplotlib.pyplot as plt
print(f"현재 폰트: {plt.rcParams['font.family']}")
print(f"유니코드 마이너스: {plt.rcParams['axes.unicode_minus']}")
```

## 참고사항

1. **성능**: 한글 폰트 설정은 프로그램 시작 시 한 번만 실행됩니다.
2. **호환성**: Python 3.7 이상에서 테스트되었습니다.
3. **의존성**: matplotlib, numpy가 필요합니다.

## 문의

한글 폰트 설정에 문제가 있거나 추가 도움이 필요한 경우, 다음 정보와 함께 문의하세요:

1. 운영체제 및 버전
2. Python 버전
3. matplotlib 버전
4. 오류 메시지
5. `test_korean_font.py` 실행 결과