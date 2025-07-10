# AI 사회 실험실: 생성형 에이전트 모델링 실습

## 📚 프로젝트 개요

이 저장소는 "**AI 사회 실험실 구축하기: 생성형 에이전트 모델링의 이론과 실제**" 책의 실습 코드를 제공합니다. 전통적인 행위자 기반 모형(Agent-Based Model, ABM)과 거대언어모델(Large Language Model, LLM)을 활용한 생성형 ABM을 비교하고 학습할 수 있습니다.

### 🎯 학습 목표

- 쉘링의 분리 모형을 통한 ABM 기본 개념 이해
- 전통적 규칙 기반 ABM과 LLM 기반 ABM의 차이점 학습
- 생성형 에이전트의 세 가지 유형 실습
- 시뮬레이션 결과 분석 및 시각화 방법 습득

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/letitbk/ai-social-laboratory.git
cd ai-social-laboratory

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정 (LLM 기능 사용 시)

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 또는 환경변수로 설정
export OPENAI_API_KEY="your_api_key_here"
```

### 3. 💰 비용 확인 (중요!)

**LLM 기반 시뮬레이션을 실행하기 전에 반드시 예상 비용을 확인하세요!**

```bash
# 기본 설정 비용 확인
python cost_calculator.py

# 모델별 비용 비교
python cost_calculator.py --compare

# 사용자 정의 설정 비용 확인
python cost_calculator.py --width 10 --height 10 --max-steps 3 --model gpt-4o-mini
```

#### 📊 주요 모델 비용 비교 (2025년 6월 기준)
| 모델 | 기본 설정 비용 | 상대 비용 | 권장 용도 |
|------|---------------|----------|----------|
| **gpt-4o-mini** | $0.0216 | 1.0x ✅ | 💡 최고 경제성 |
| **gpt-4.1-mini** | $0.0575 | 2.7x ✅ | 💡 새로운 경제형 |
| **o3** | $0.2875 | 13.3x ⭐ | 🔥 고급 추론 + 저비용 |
| gpt-4o | $0.3295 | 15.3x ⚠️ | 🔄 출력 비용 인하 |
| gpt-4 | $3.9534 | 183.3x ❌ | 레거시 모델 |

> ⚠️ **중요**: 가격은 변동될 수 있습니다. 최신 정보는 [OpenAI 공식 가격 페이지](https://openai.com/pricing)에서 확인하세요.


### 4. 🇰🇷 한글 폰트 설정 (Korean Font Setup)

프로그램에서 한글이 제대로 표시되지 않는 경우:

```bash
# 한글 폰트 테스트
python test_korean_font.py
```

#### 운영체제별 권장 폰트
- **macOS**: AppleGothic (자동 감지)
- **Windows**: Malgun Gothic (자동 감지)
- **Linux**: Nanum Gothic (설치 필요)

#### Linux 사용자 폰트 설치
```bash
# Ubuntu/Debian
sudo apt-get install fonts-nanum

# CentOS/RHEL
sudo yum install naver-nanum-fonts
```

### 5. 기본 실행

```bash
# 전통적 쉘링 모형 실행 (무료, 즉시 실행)
python schelling_traditional.py

# LLM 기반 쉘링 모형 실행 (비용 발생, 비용 확인 후 실행)
python schelling_llm.py

# 데모 모드 실행 (무료, API 키 불필요)
python demo_schelling_llm.py
```

> ⚠️ **중요**: `schelling_llm.py`는 실행 전에 비용 확인 창이 나타납니다.
> 비용이 $0.01 이상인 경우 사용자 확인을 요청합니다.

## 📁 파일 구조

```
ai-social-laboratory/
├── README.md                 # 프로젝트 설명서
├── requirements.txt          # 필요한 패키지 목록
├── schelling_traditional.py  # 전통적 쉘링 모형
├── schelling_llm.py         # LLM 기반 쉘링 모형 (비용 발생)
├── demo_schelling_llm.py    # 데모 모드 (무료)
├── cost_calculator.py       # 비용 계산기
├── utils.py                 # 공통 유틸리티 함수
├── examples/                # 실습 예제들
│   ├── basic_comparison.py  # 기본 비교 실습
│   ├── parameter_study.py   # 매개변수 연구
│   └── advanced_analysis.py # 고급 분석
├── data/                    # 시뮬레이션 결과 데이터
└── outputs/                 # 생성된 그래프 및 애니메이션
```

## 🔧 주요 기능

### 1. 전통적 쉘링 모형 (`schelling_traditional.py`)

- 규칙 기반 행위자 의사결정
- 명시적 만족도 계산 함수
- 빠른 시뮬레이션 실행
- 다양한 분석 메트릭 제공

```python
from schelling_traditional import *

# 시뮬레이션 실행
grid = initialize_grid()
frames = simulate_schelling(grid)

# 결과 분석
results = analyze_results(frames)
print(f"최종 분리지수: {results['final_segregation']:.3f}")
```

### 2. LLM 기반 쉘링 모형 (`schelling_llm.py`)

- 자연어 기반 상황 이해
- 세 가지 행위자 유형 지원
- OpenAI API 통합
- 응답 히스토리 분석

```python
from schelling_llm import *

# LLM 시뮬레이션 실행
sim = LLMSchellingSimulation()
frames = sim.run_simulation(agent_type="group")

# 행위자 응답 분석
responses = sim.get_agent_responses()
```

### 3. 유틸리티 함수 (`utils.py`)

- 다양한 분리 지수 계산
- 시뮬레이션 결과 비교
- 매개변수 연구 도구
- 고급 시각화 기능

```python
from utils import *

# 시뮬레이션 비교
comparison = compare_simulations(frames1, frames2)
create_comparison_plot(comparison)

# 나란히 애니메이션 생성
anim = create_side_by_side_animation(frames1, frames2,
                                   ["Traditional", "LLM"])
```

## 🎓 실습 가이드

전통적 방법과 LLM 방법의 기본적인 차이를 확인해보세요.

```python
# examples/basic_comparison.py 실행
python examples/basic_comparison.py
```

## 🎨 시각화 기능

### 애니메이션 생성
```python
# 단일 시뮬레이션 애니메이션
anim = visualize_simulation(frames, save_path="simulation.gif")

# 비교 애니메이션
anim = create_side_by_side_animation(frames1, frames2,
                                   ["Method A", "Method B"])
```

### 통계 차트
```python
# 비교 차트 생성
create_comparison_plot(comparison_results)

# 매개변수 연구 차트
create_parameter_study_plot(parameter_results)
```

## 🔍 LLM 행위자 유형

### 1. 단순 LLM 행위자
- 기본적인 규칙 기반 의사결정
- 명시적 임계값 활용
- 빠른 응답 시간

### 2. 집단 LLM 행위자
- 사회집단별 특성 반영
- 보수적 vs 진보적 성향
- 맥락적 의사결정

### 3. 개인 LLM 행위자 (고급)
- 개별 인간의 고유성 모사
- 복잡한 개인 특성 반영
- 높은 현실성

## ⚙️ 설정 옵션

## 💡 비용 절약 전략

### 1. 무료 대안 사용
- **데모 모드**: API 키 없이 실행 가능
- **전통적 방법**: 완전 무료, 즉시 실행

### 2. 설정 최적화로 비용 절약
| 변경 사항 | 비용 절약 | 영향 |
|----------|----------|------|
| 격자 크기: 15x15 → 10x10 | 56% 절약 | 에이전트 수 감소 |
| 단계 수: 5 → 3 | 40% 절약 | 시뮬레이션 기간 단축 |
| 모델: gpt-4 → gpt-4o-mini | 95% 절약 | 성능 거의 동일 |
| 빈 공간: 0.3 → 0.5 | 29% 절약 | 에이전트 수 감소 |

### 3. 단계적 접근
1. **1단계**: 데모 모드로 기능 확인
2. **2단계**: 작은 격자(10x10)로 테스트
3. **3단계**: 필요시 크기 확장

## 🛠️ 문제 해결 (Troubleshooting)

### 한글 표시 문제
```bash
# 한글 폰트 테스트
python test_korean_font.py

# 폰트 캐시 새로고침 (Python)
python -c "import matplotlib.font_manager as fm; fm._rebuild()"
```

### 비용 관련 문제
```bash
# 비용 계산기로 예상 비용 확인
python cost_calculator.py

# 무료 데모 모드 실행
python demo_schelling_llm.py
```

### API 키 관련 문제
```bash
# API 키 확인
echo $OPENAI_API_KEY

# .env 파일 확인
cat .env
```

## 📚 추가 가이드

- **COST_GUIDE.md**: 상세한 비용 분석 및 절약 전략
- **KOREAN_FONT_SETUP.md**: 한글 폰트 설정 완전 가이드

### 기본 매개변수
```python
# 격자 설정
WIDTH, HEIGHT = 15, 15
EMPTY_RATIO = 0.1

# 시뮬레이션 설정
THRESHOLD_VALUE = 0.6
MAX_STEPS = 10

# LLM 설정
API_DELAY = 0.1  # API 호출 간 대기시간
```

### 고급 설정
```python
# 시각화 설정
ANIMATION_INTERVAL = 200  # 애니메이션 속도
FIGURE_SIZE = (10, 10)    # 그림 크기

# 분석 설정
CONVERGENCE_THRESHOLD = 0.001  # 수렴 임계값
```

## 🚨 주의사항

### 💰 비용 관리 (중요!)

**OpenAI API는 유료 서비스입니다!** 다음 사항을 반드시 확인하세요:

#### 비용 확인 절차
1. **실행 전 비용 계산**: `python cost_calculator.py`
2. **예산 설정**: OpenAI 계정에서 월 한도 설정
3. **사용량 모니터링**: [OpenAI 사용량 대시보드](https://platform.openai.com/usage) 확인

#### 비용 절약 전략
- 🔸 **데모 모드 우선 사용**: `demo_schelling_llm.py` (무료)
- 🔸 **작은 격자로 시작**: 10x10 또는 15x15 권장
- 🔸 **단계 수 제한**: 3-5 단계로 시작
- 🔸 **저렴한 모델**: gpt-4o-mini 사용 (권장)
- 🔸 **API 지연 조정**: `API_DELAY` 값 증가

#### 예상 비용 가이드 (2025년 6월 업데이트)
| 격자 크기 | 단계 수 | 모델 | 예상 비용 | 특징 |
|----------|--------|------|----------|------|
| 10x10 | 3 | gpt-4o-mini | ~$0.0015 | 💡 최고 경제성 |
| 15x15 | 5 | gpt-4o-mini | ~$0.0041 | 💡 여전히 권장 |
| 10x10 | 3 | gpt-4.1-mini | ~$0.0040 | 💡 새로운 경제형 |
| 15x15 | 5 | gpt-4.1-mini | ~$0.0109 | 💡 gpt-4o-mini 대안 |
| 10x10 | 3 | o3 | ~$0.0200 | 🔥 고급 추론 + 저비용 |
| 15x15 | 5 | o3 | ~$0.0540 | 🔥 80% 인하 혜택 |
| 20x20 | 10 | gpt-4o-mini | ~$0.0196 | 💡 대용량 저비용 |
| 15x15 | 5 | gpt-4 | ~$0.0825 | 레거시 모델 |

### API 사용량 관리
- 작은 격자 크기로 시작하세요 (15x15 권장)
- API_DELAY를 조정하여 호출 빈도를 제어하세요
- 실행 중 중단하려면 Ctrl+C를 사용하세요

### 메모리 사용량
- 큰 격자는 많은 메모리를 사용합니다
- 장기간 시뮬레이션 시 중간 저장을 권장합니다

### 재현성
- 동일한 결과를 위해 random seed를 설정하세요
- LLM 응답은 확률적이므로 완전한 재현이 어려울 수 있습니다

## 🛠️ 문제 해결

### 일반적인 문제들

**Q: API 키 오류가 발생합니다**
```
A: OPENAI_API_KEY 환경변수가 올바르게 설정되었는지 확인하세요.
   API 키가 없어도 fallback 모드로 실행됩니다.
```

**Q: 시뮬레이션이 너무 느립니다**
```
A: 격자 크기를 줄이거나 API_DELAY를 조정하세요.
   전통적 방법은 훨씬 빠르게 실행됩니다.
```

**Q: 애니메이션이 저장되지 않습니다**
```
A: pillow 패키지가 설치되어 있는지 확인하세요.
   pip install pillow
```

### 디버깅 팁

```python
# 디버그 모드 실행
import logging
logging.basicConfig(level=logging.DEBUG)

# 중간 결과 저장
save_simulation_data(frames, metadata, "debug_data.json")

# 응답 히스토리 확인
responses = sim.get_agent_responses()
for resp in responses[:5]:
    print(resp['response'])
```

## 🤝 기여 방법

1. Fork 저장소
2. 새 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📚 참고 자료

### 학술 논문
- Schelling, T. C. (1971). Dynamic models of segregation. Journal of Mathematical Sociology, 1(2), 143-186.

### 관련 도서
- "AI 사회 실험실 구축 가이드: 생성형 에이전트 모델링의 이론과 실제"

### 온라인 자료
- [OpenAI API 문서](https://platform.openai.com/docs)

## 💬 문의 및 지원

- 📧 이메일: bklee@nyu.edu
- 💬 디스커션: [GitHub Discussions](https://github.com/letitbk/ai-social-laboratory/discussions)
- 🐛 버그 신고: [GitHub Issues](https://github.com/letitbk/ai-social-laboratory/issues)

---

**즐거운 시뮬레이션 되세요! 🎉**

*이 프로젝트는 사회과학과 인공지능의 융합 연구를 위한 교육 목적으로 제작되었습니다.*