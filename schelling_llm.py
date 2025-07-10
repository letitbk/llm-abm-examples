#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 기반 쉘링 분리 모형
LLM-based Schelling Segregation Model

이 모듈은 토마스 쉘링의 분리 모형을 거대언어모델(LLM) 행위자를 사용하여 구현합니다.
각 행위자는 LLM을 통해 자연어로 상황을 이해하고 의사결정을 내립니다.
"""

from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import random
import time
import os
import sys
import platform
from typing import List, Tuple, Dict, Optional

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트를 설정합니다."""
    # 터미널 출력 인코딩 설정
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    system = platform.system()

    if system == "Darwin":  # macOS
        font_candidates = [
            "AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic",
            "Malgun Gothic", "Arial Unicode MS"
        ]
    elif system == "Windows":
        font_candidates = [
            "Malgun Gothic", "Nanum Gothic", "Arial Unicode MS",
            "Gulim", "Dotum"
        ]
    else:  # Linux
        font_candidates = [
            "Nanum Gothic", "DejaVu Sans", "Liberation Sans",
            "Noto Sans CJK KR", "UnDotum"
        ]

    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 폰트 설정: {font}")
            return font

    # 한글 폰트를 찾지 못한 경우 기본 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("경고: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    print("한글이 제대로 표시되지 않을 수 있습니다.")
    return None

# 한글 폰트 설정 실행
setup_korean_font()

# OpenAI 클라이언트 초기화 (API 키가 있을 때만)
client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"OpenAI 클라이언트 초기화 오류: {e}")
        client = None

# 모형 매개변수 설정 (빠른 수렴을 위한 최적화 설정)
WIDTH, HEIGHT = 15, 15          # 더 큰 격자로 명확한 분리 패턴 생성
EMPTY_RATIO = 0.25              # 높은 밀도로 빠른 클러스터링 유도 (75% 점유)
THRESHOLD_VALUE = 0.2           # 낮은 임계값으로 쉬운 만족 조건 (20%)
MAX_STEPS = 20                  # 충분한 단계로 완전한 수렴 허용
API_DELAY = 0.02                # API 호출 간 대기 시간 (초) - 단축

# OpenAI API 키 설정 확인
if not os.getenv("OPENAI_API_KEY"):
    print("경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("실제 LLM 기능을 사용하려면 API 키를 설정하세요.")

class LLMAgent:
    """LLM 기반 행위자 클래스"""

    def __init__(self, agent_type: int, x: int, y: int):
        """
        LLM 행위자를 초기화합니다.

        Args:
            agent_type: 행위자 유형 (1 또는 2)
            x, y: 행위자의 위치
        """
        self.agent_type = agent_type
        self.x = x
        self.y = y
        self.type_name = "A" if agent_type == 1 else "B"
        self.satisfaction_history = []

        # 집단별 특성 정의 (쉘링 모형의 사회적 맥락 반영)
        if self.type_name == "A":
            self.group_characteristics = {
                "name": "그룹 A (사회적 집단)",
                "traits": [
                    "비슷한 사회경제적 배경을 가진 사람들과의 근접성을 선호",
                    "안전하고 예측 가능한 거주 환경을 중요시",
                    "자녀 교육과 지역 공동체 참여를 우선시",
                    "장기적 거주 안정성과 자산 가치 보존을 고려"
                ],
                "preferences": "비슷한 생활 패턴과 가치관을 가진 이웃들과 함께 살며 공동체 결속을 통한 안정감을 추구"
            }
        else:
            self.group_characteristics = {
                "name": "그룹 B (사회적 집단)",
                "traits": [
                    "다양한 문화적 배경을 가진 사람들과의 상호작용을 선호",
                    "창의적이고 역동적인 거주 환경을 추구",
                    "개인의 자율성과 생활 방식의 다양성을 중시",
                    "새로운 기회와 경험에 대한 개방적 태도"
                ],
                "preferences": "다양성이 있는 환경에서 새로운 아이디어와 경험을 통해 개인적 성장을 추구"
            }

    def get_neighborhood_info(self, grid: np.ndarray, observation_radius: int = 2) -> Dict:
        """
        주변 환경 정보를 수집합니다.

        Args:
            grid: 현재 격자 상태
            observation_radius: 관찰 반경 (기본값: 2, 즉 5x5 영역)

        Returns:
            Dict: 이웃 정보와 격자 시각화
        """
        neighbors = {"A": 0, "B": 0, "empty": 0}
        neighborhood_grid = []

        # 유연한 크기의 영역 관찰
        for dx in range(-observation_radius, observation_radius + 1):
            row = []
            for dy in range(-observation_radius, observation_radius + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    cell_value = grid[nx, ny]
                    if cell_value == 0:
                        row.append(".")
                        if abs(dx) <= 1 and abs(dy) <= 1 and not (dx == 0 and dy == 0):
                            neighbors["empty"] += 1
                    elif cell_value == 1:
                        row.append("A")
                        if abs(dx) <= 1 and abs(dy) <= 1 and not (dx == 0 and dy == 0):
                            neighbors["A"] += 1
                    else:  # cell_value == 2
                        row.append("B")
                        if abs(dx) <= 1 and abs(dy) <= 1 and not (dx == 0 and dy == 0):
                            neighbors["B"] += 1
                else:
                    row.append("X")  # 경계 밖
            neighborhood_grid.append(row)

        grid_size = observation_radius * 2 + 1
        return {
            "neighbors": neighbors,
            "grid": neighborhood_grid,
            "total_neighbors": neighbors["A"] + neighbors["B"],
            "grid_size": f"{grid_size}x{grid_size}",
            "observation_radius": observation_radius
        }

    def decide_satisfaction_simple(self, grid: np.ndarray, threshold: float, observation_radius: int = 2) -> bool:
        """
        단순 LLM 행위자의 만족도 판단

        Args:
            grid: 현재 격자 상태
            threshold: 만족 임계값
            observation_radius: 관찰 반경

        Returns:
            bool: 만족 여부
        """
        if not client:
            return self._fallback_satisfaction(grid, threshold)

        neighborhood_info = self.get_neighborhood_info(grid, observation_radius)
        grid_str = "\n".join(["".join(row) for row in neighborhood_info["grid"]])

        # 현재 비율 계산
        total_neighbors = neighborhood_info["total_neighbors"]
        same_type_count = neighborhood_info["neighbors"][self.type_name]
        current_ratio = same_type_count / max(1, total_neighbors) if total_neighbors > 0 else 0
        required_ratio = threshold

        prompt = f"""
        당신은 유형 {self.type_name}의 행위자입니다. 현재 위치는 격자의 중앙입니다.

        다음은 당신 주변의 {neighborhood_info["grid_size"]} 격자 상황입니다:
        {grid_str}

        기호 설명:
        - A: 유형 A 행위자
        - B: 유형 B 행위자
        - .: 빈 공간
        - X: 경계 밖

        당신의 바로 옆 이웃(8방향) 분석:
        - 같은 유형 ({self.type_name}): {same_type_count}명
        - 다른 유형: {neighborhood_info["neighbors"]["A" if self.type_name == "B" else "B"]}명
        - 빈 공간: {neighborhood_info["neighbors"]["empty"]}개
        - 총 이웃 수: {total_neighbors}명

        계산 과정:
        1. 현재 같은 유형 비율: {same_type_count}/{total_neighbors} = {current_ratio:.3f} ({current_ratio:.1%})
        2. 필요한 최소 비율: {required_ratio:.3f} ({required_ratio:.1%})
        3. 조건 충족 여부: {current_ratio:.3f} >= {required_ratio:.3f}? {"예" if current_ratio >= required_ratio else "아니오"}

        엄격한 기준: 당신은 이웃 중 정확히 {threshold:.1%} 이상이 같은 유형이어야만 만족합니다.
        현재 비율이 기준을 충족하지 못하면 반드시 불만족해야 합니다.

        위 계산을 바탕으로 "만족" 또는 "불만족"으로만 답하세요.
        """

        try:
            response = client.chat.completions.create(model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "당신은 수학적 계산에 기반하여 거주지 만족도를 엄격하게 판단하는 행위자입니다. 주어진 계산 결과를 정확히 따라야 하며, 감정이나 주관적 판단을 배제하고 오직 수치적 기준만을 사용해야 합니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1)

            result = response.choices[0].message.content.strip()
            is_satisfied = "만족" in result and "불만족" not in result

            # 히스토리 저장
            self.satisfaction_history.append({
                "satisfied": is_satisfied,
                "same_type_ratio": neighborhood_info["neighbors"][self.type_name] / max(1, neighborhood_info["total_neighbors"]),
                "response": result,
                "type": "simple"
            })

            return is_satisfied

        except Exception as e:
            print(f"API 호출 오류: {e}")
            return self._fallback_satisfaction(grid, threshold)

    def decide_satisfaction_group(self, grid: np.ndarray, threshold: float, observation_radius: int = 2) -> bool:
        """
        집단 LLM 행위자의 만족도 판단

        Args:
            grid: 현재 격자 상태
            threshold: 만족 임계값
            observation_radius: 관찰 반경

        Returns:
            bool: 만족 여부
        """
        if not client:
            return self._fallback_satisfaction(grid, threshold)

        neighborhood_info = self.get_neighborhood_info(grid, observation_radius)
        grid_str = "\n".join(["".join(row) for row in neighborhood_info["grid"]])

        # 집단 특성 설명
        traits_str = "\n".join([f"- {trait}" for trait in self.group_characteristics["traits"]])

        # 현재 비율 계산
        total_neighbors = neighborhood_info["total_neighbors"]
        same_type_count = neighborhood_info["neighbors"][self.type_name]
        current_ratio = same_type_count / max(1, total_neighbors) if total_neighbors > 0 else 0
        required_ratio = threshold

        prompt = f"""
        당신은 거주지 선택에 있어 특정한 사회적 선호를 가진 개인입니다.

        === 당신의 사회적 배경과 선호 ===
        집단: {self.group_characteristics["name"]}

        주요 특성:
        {traits_str}

        거주지 선호도: {self.group_characteristics["preferences"]}

        === 현재 거주지 상황 분석 ===

        주변 {neighborhood_info["grid_size"]} 지역 현황:
        {grid_str}

        기호 설명: A = 그룹A 거주자, B = 그룹B 거주자, . = 빈 집, X = 지역 경계

        직접 이웃(8방향) 구성:
        - 같은 그룹 ({self.type_name}): {same_type_count}명
        - 다른 그룹: {neighborhood_info["neighbors"]["A" if self.type_name == "B" else "B"]}명
        - 빈 집: {neighborhood_info["neighbors"]["empty"]}개
        - 총 이웃 수: {total_neighbors}명

        === 거주 만족도 판단 기준 ===

        - 현재 같은 그룹 비율: {same_type_count}/{total_neighbors} = {current_ratio:.1%}
        - 개인적 선호 임계값: {required_ratio:.1%}
        - 기준 충족 여부: {"충족" if current_ratio >= required_ratio else "미충족"}

        === 의사결정 요청 ===

        위의 상황을 종합하여, 당신의 사회적 선호와 현재 이웃 구성을 고려할 때:

        1. 현재 거주지에서 만족하며 계속 살고 싶은가?
        2. 아니면 더 적합한 지역으로 이주하고 싶은가?

        수치적 기준: 같은 그룹 비율이 {threshold:.1%} 이상이어야 만족
        현재 상황: {current_ratio:.1%} (기준 {"충족" if current_ratio >= required_ratio else "미충족"})

        "만족" 또는 "불만족"으로 답하고, 당신의 사회적 선호에 기반한 간단한 이유를 설명하세요.
        """

        try:
            response = client.chat.completions.create(model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "당신은 특정 사회집단의 구성원이지만, 거주지 만족도는 오직 수학적 계산에 기반하여 엄격하게 판단해야 합니다. 그룹 특성은 참고사항일 뿐이며, 최종 결정은 반드시 제시된 수치적 기준을 따라야 합니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3)

            result = response.choices[0].message.content.strip()
            is_satisfied = "만족" in result and "불만족" not in result

            # 히스토리 저장
            self.satisfaction_history.append({
                "satisfied": is_satisfied,
                "same_type_ratio": neighborhood_info["neighbors"][self.type_name] / max(1, neighborhood_info["total_neighbors"]),
                "response": result,
                "type": "group"
            })

            return is_satisfied

        except Exception as e:
            print(f"API 호출 오류: {e}")
            return self._fallback_satisfaction(grid, threshold)

    def _fallback_satisfaction(self, grid: np.ndarray, threshold: float) -> bool:
        """
        API 오류 시 사용하는 전통적 만족도 계산

        Args:
            grid: 현재 격자 상태
            threshold: 만족 임계값

        Returns:
            bool: 만족 여부
        """
        neighborhood_info = self.get_neighborhood_info(grid)
        if neighborhood_info["total_neighbors"] == 0:
            return True

        same_type_ratio = neighborhood_info["neighbors"][self.type_name] / neighborhood_info["total_neighbors"]
        return same_type_ratio >= threshold

class LLMSchellingSimulation:
    """LLM 기반 쉘링 시뮬레이션 관리 클래스"""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT,
                 empty_ratio: float = EMPTY_RATIO):
        """
        시뮬레이션을 초기화합니다.

        Args:
            width: 격자 너비
            height: 격자 높이
            empty_ratio: 빈 공간 비율
        """
        self.width = width
        self.height = height
        self.empty_ratio = empty_ratio
        self.grid = None
        self.agents = {}
        self.frames = []

    def initialize_grid(self):
        """격자와 행위자를 초기화합니다."""
        agent_ratio = (1 - self.empty_ratio) / 2
        self.grid = np.random.choice([1, 2, 0], size=(self.width, self.height),
                                    p=[agent_ratio, agent_ratio, self.empty_ratio])

        # 행위자 객체 생성
        self.agents = {}
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] != 0:
                    agent = LLMAgent(self.grid[x, y], x, y)
                    self.agents[(x, y)] = agent

    def move_unsatisfied_agents(self, threshold: float, agent_type: str = "simple", observation_radius: int = 2):
        """
        불만족한 행위자들을 이동시킵니다.

        Args:
            threshold: 만족 임계값
            agent_type: 행위자 유형 ("simple" 또는 "group")
            observation_radius: 관찰 반경
        """
        unsatisfied_agents = []
        total_agents = len(self.agents)
        processed_agents = 0

        print(f"총 {total_agents}개 행위자의 만족도 확인 중...")

        # 불만족한 행위자 찾기
        for (x, y), agent in self.agents.items():
            processed_agents += 1

            # 진행률 표시 (10% 단위)
            if processed_agents % max(1, total_agents // 10) == 0:
                progress = (processed_agents / total_agents) * 100
                print(f"  진행률: {progress:.0f}% ({processed_agents}/{total_agents})")

            if agent_type == "simple":
                is_satisfied = agent.decide_satisfaction_simple(self.grid, threshold, observation_radius)
            elif agent_type == "group":
                is_satisfied = agent.decide_satisfaction_group(self.grid, threshold, observation_radius)
            else:
                is_satisfied = agent.decide_satisfaction_simple(self.grid, threshold, observation_radius)

            if not is_satisfied:
                unsatisfied_agents.append((x, y))

            # API 호출 제한을 위한 대기 (API 키가 있을 때만)
            if client:
                time.sleep(API_DELAY)

        print(f"불만족한 행위자 수: {len(unsatisfied_agents)}/{total_agents}")

        # 불만족한 행위자들 이동
        if unsatisfied_agents:
            print(f"{len(unsatisfied_agents)}개 행위자 이동 중...")

            for i, (x, y) in enumerate(unsatisfied_agents):
                # 빈 공간 찾기
                empty_spaces = [(i, j) for i in range(self.width)
                               for j in range(self.height) if self.grid[i, j] == 0]

                if empty_spaces:
                    new_x, new_y = random.choice(empty_spaces)

                    # 이동 실행
                    agent = self.agents[(x, y)]
                    agent.x, agent.y = new_x, new_y
                    self.grid[new_x, new_y] = self.grid[x, y]
                    self.grid[x, y] = 0

                    # 행위자 딕셔너리 업데이트
                    self.agents[(new_x, new_y)] = agent
                    del self.agents[(x, y)]

                    # 이동 진행률 표시
                    if (i + 1) % max(1, len(unsatisfied_agents) // 5) == 0:
                        progress = ((i + 1) / len(unsatisfied_agents)) * 100
                        print(f"  이동 완료: {progress:.0f}% ({i + 1}/{len(unsatisfied_agents)})")
        else:
            print("모든 행위자가 만족함. 이동 없음.")

    def calculate_satisfaction_rate(self, threshold: float, agent_type: str = "simple",
                                   sample_size: int = None, observation_radius: int = 2) -> float:
        """
        현재 만족도를 계산합니다.

        Args:
            threshold: 만족 임계값
            agent_type: 행위자 유형
            sample_size: 샘플링할 행위자 수 (None이면 전체)
            observation_radius: 관찰 반경

        Returns:
            float: 0-1 사이의 만족도
        """
        if not self.agents:
            return 0.0

        agents_list = list(self.agents.values())

        # 샘플링으로 성능 최적화
        if sample_size and len(agents_list) > sample_size:
            agents_to_check = random.sample(agents_list, sample_size)
            print(f"만족도 계산 중... (샘플링: {sample_size}/{len(agents_list)})")
        else:
            agents_to_check = agents_list
            print(f"만족도 계산 중... (전체: {len(agents_list)})")

        satisfied_count = 0
        total_count = len(agents_to_check)

        for i, agent in enumerate(agents_to_check):
            if agent_type == "simple":
                is_satisfied = agent.decide_satisfaction_simple(self.grid, threshold, observation_radius)
            elif agent_type == "group":
                is_satisfied = agent.decide_satisfaction_group(self.grid, threshold, observation_radius)
            else:
                is_satisfied = agent.decide_satisfaction_simple(self.grid, threshold, observation_radius)

            if is_satisfied:
                satisfied_count += 1

            # API 호출 제한 (샘플링 시에만 적용)
            if client and sample_size:
                time.sleep(API_DELAY)

        return satisfied_count / total_count

    def run_simulation(self, steps: int = MAX_STEPS, threshold: float = THRESHOLD_VALUE,
                      agent_type: str = "simple", observation_radius: int = 2) -> List[np.ndarray]:
        """
        시뮬레이션을 실행합니다.

        Args:
            steps: 최대 단계 수
            threshold: 만족 임계값
            agent_type: 행위자 유형
            observation_radius: 관찰 반경

        Returns:
            List[np.ndarray]: 각 단계별 격자 상태
        """
        self.initialize_grid()
        self.frames = [self.grid.copy()]

        print(f"=== LLM 쉘링 시뮬레이션 시작 ({agent_type} 모드) ===")
        print(f"격자 크기: {self.width}x{self.height}")
        print(f"관찰 영역: {observation_radius*2+1}x{observation_radius*2+1}")
        print(f"임계값: {threshold}")
        api_available = bool(client)
        print(f"API 키 상태: {'설정됨' if api_available else '미설정 (fallback 모드)'}")

        if not api_available:
            print("주의: API 키가 없어 전통적 규칙 기반 방식으로 실행됩니다.")
            print("실제 LLM 기능을 사용하려면 OPENAI_API_KEY 환경변수를 설정하세요.")

        for step in range(steps):
            print(f"\n--- {step + 1}단계 실행 중 ---")

            prev_grid = self.grid.copy()
            self.move_unsatisfied_agents(threshold, agent_type, observation_radius)
            self.frames.append(self.grid.copy())

            # 수렴 체크
            if np.array_equal(prev_grid, self.grid):
                print("수렴 달성. 시뮬레이션 종료.")
                break

            # 만족도 계산 (간헐적으로만 수행, 샘플링 사용)
            if step % 3 == 0:  # 더 자주 체크하되 샘플링 사용
                sample_size = min(50, len(self.agents) // 2)  # 최대 50개 또는 절반
                satisfaction_rate = self.calculate_satisfaction_rate(threshold, agent_type, sample_size, observation_radius)
                print(f"전체 만족도 (추정): {satisfaction_rate:.3f}")

                if satisfaction_rate >= 0.95:
                    print("대부분의 행위자가 만족함. 시뮬레이션 종료.")
                    break

        return self.frames

    def get_agent_responses(self) -> List[Dict]:
        """
        모든 행위자의 응답 히스토리를 반환합니다.

        Returns:
            List[Dict]: 행위자별 응답 히스토리
        """
        all_responses = []
        for (x, y), agent in self.agents.items():
            for response in agent.satisfaction_history:
                all_responses.append({
                    "position": (x, y),
                    "agent_type": agent.type_name,
                    "response": response
                })
        return all_responses

def visualize_llm_simulation(frames: List[np.ndarray],
                           title: str = "쉘링 분리 모형 (LLM 기반)",
                           save_path: str = None) -> animation.FuncAnimation:
    """
    LLM 시뮬레이션 결과를 시각화합니다.

    Args:
        frames: 각 단계별 격자 상태
        title: 애니메이션 제목
        save_path: 저장할 경로

    Returns:
        animation.FuncAnimation: 애니메이션 객체
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 색상 맵 설정
    colors = ['white', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    def animate(frame_num):
        ax.clear()
        ax.imshow(frames[frame_num], cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f"{title} - {frame_num}단계", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        # 통계 정보는 간단히 표시
        ax.text(0.02, 0.98, f"단계: {frame_num}/{len(frames)-1}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=12)

    anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                 interval=500, repeat=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
        print(f"애니메이션이 {save_path}에 저장되었습니다.")

    return anim

def estimate_cost(width: int, height: int, empty_ratio: float, max_steps: int,
                 model: str = "gpt-4.1-nano-2025-04-14") -> dict:
    """시뮬레이션 예상 비용을 계산합니다."""
    try:
        from cost_calculator import calculate_simulation_cost
        return calculate_simulation_cost(width, height, empty_ratio, max_steps,
                                       agent_type="simple", model=model)
    except ImportError:
        # 간단한 추정치 계산
        agent_count = int(width * height * (1 - empty_ratio))
        expected_steps = max_steps * 0.7  # 70% 확률로 조기 수렴
        total_calls = int(agent_count * expected_steps)

        # 기본 토큰 추정
        tokens_per_call = 210  # 입력 200 + 출력 10
        total_tokens = total_calls * tokens_per_call

        # gpt-4.1-nano 가격 (2025년 기준)
        cost_per_1k_tokens = 0.00025  # 평균 입력/출력 비용
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return {
            "agent_count": agent_count,
            "total_api_calls": total_calls,
            "total_cost": estimated_cost,
            "model": model
        }

def show_cost_warning(cost_info: dict):
    """비용 경고를 표시합니다."""
    print("\n" + "=" * 60)
    print("💰 예상 비용 정보")
    print("=" * 60)
    print(f"📊 에이전트 수: {cost_info['agent_count']:,}개")
    print(f"🔢 예상 API 호출: {cost_info['total_api_calls']:,}회")
    print(f"💵 예상 비용: ${cost_info['total_cost']:.4f} USD")
    print(f"🤖 사용 모델: {cost_info['model']}")
    print()

    if cost_info['total_cost'] < 0.01:
        print("🟢 매우 저렴한 비용입니다. 안전하게 실행할 수 있습니다.")
    elif cost_info['total_cost'] < 0.05:
        print("🟡 저렴한 비용입니다. 일반적인 실험용으로 적합합니다.")
    elif cost_info['total_cost'] < 0.25:
        print("🟠 보통 비용입니다. 비용을 고려하여 실행하세요.")
    else:
        print("🔴 비싼 비용입니다. 신중하게 고려하세요.")

    print()
    print("💡 비용 절약 팁:")
    print("  - 더 저렴한 모델 사용 (gpt-4.1-nano 현재 사용 중)")
    print("  - 격자 크기 줄이기 (15x15 → 10x10)")
    print("  - 최대 단계 수 줄이기 (5 → 3)")
    print("  - 데모 모드 사용 (demo_schelling_llm.py)")
    print("  - 상세한 비용 분석: python cost_calculator.py --compare")
    print("=" * 60)

def main():
    """메인 실행 함수"""
    print("=== LLM 기반 쉘링 분리 모형 ===")

    # 비용 추정 및 경고 표시
    cost_info = estimate_cost(WIDTH, HEIGHT, EMPTY_RATIO, MAX_STEPS)
    show_cost_warning(cost_info)

    # 사용자 확인
    if cost_info['total_cost'] > 0.01:  # 1센트 이상인 경우
        response = input("\n계속 진행하시겠습니까? (y/N): ").lower().strip()
        if response not in ['y', 'yes', '예', 'ㅇ']:
            print("❌ 시뮬레이션이 취소되었습니다.")
            print("💡 무료 데모를 원하시면 demo_schelling_llm.py를 실행하세요!")
            return None, None

    print("\n🚀 시뮬레이션을 시작합니다...")

    # 시뮬레이션 생성
    sim = LLMSchellingSimulation(WIDTH, HEIGHT, EMPTY_RATIO)

    # 단순 LLM 행위자 시뮬레이션
    print("\n1. 단순 LLM 행위자 시뮬레이션")
    frames_simple = sim.run_simulation(MAX_STEPS, THRESHOLD_VALUE, "simple")

    # 집단 LLM 행위자 시뮬레이션
    print("\n2. 집단 LLM 행위자 시뮬레이션")
    sim2 = LLMSchellingSimulation(WIDTH, HEIGHT, EMPTY_RATIO)
    frames_group = sim2.run_simulation(MAX_STEPS, THRESHOLD_VALUE, "group")

    # 결과 시각화
    print("\n시각화 중...")
    anim1 = visualize_llm_simulation(frames_simple,
                                   "쉘링 분리 모형 (단순 LLM)",
                                   "schelling_llm_simple.gif")

    anim2 = visualize_llm_simulation(frames_group,
                                   "쉘링 분리 모형 (집단 LLM)",
                                   "schelling_llm_group.gif")

    # 행위자 응답 분석
    if client:
        print("\n=== 행위자 응답 분석 ===")
        responses = sim.get_agent_responses()
        print(f"총 응답 수: {len(responses)}")

        # 응답 예시 출력
        for i, resp in enumerate(responses[:5]):
            print(f"\n응답 {i+1}:")
            print(f"  위치: {resp['position']}")
            print(f"  유형: {resp['agent_type']}")
            print(f"  응답: {resp['response']['response']}")

    plt.show()

    return frames_simple, frames_group

if __name__ == "__main__":
    frames_simple, frames_group = main()