#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전통적 방법으로 구현한 쉘링 분리 모형
Traditional Schelling Segregation Model

이 모듈은 토마스 쉘링의 분리 모형을 전통적인 규칙 기반 방법으로 구현합니다.
각 행위자는 명시적으로 정의된 규칙에 따라 행동하며, 주변 이웃의 구성에 따라
만족도를 계산하고 이주 결정을 내립니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import sys
import platform
import matplotlib.font_manager as fm
from typing import List, Tuple

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

# 모형 매개변수 설정
WIDTH, HEIGHT = 50, 50          # 격자 크기
EMPTY_RATIO = 0.2               # 빈 공간 비율
THRESHOLD_VALUE = 0.3           # 만족 임계값
MAX_STEPS = 100                 # 최대 시뮬레이션 단계 수

def initialize_grid(width: int = WIDTH, height: int = HEIGHT,
                   empty_ratio: float = EMPTY_RATIO) -> np.ndarray:
    """
    격자를 초기화하고 행위자들을 무작위로 배치합니다.

    Args:
        width: 격자 너비
        height: 격자 높이
        empty_ratio: 빈 공간 비율

    Returns:
        np.ndarray: 초기화된 격자 (0: 빈 공간, 1: 타입 A, 2: 타입 B)
    """
    agent_ratio = (1 - empty_ratio) / 2
    grid = np.random.choice([1, 2, 0], size=(width, height),
                           p=[agent_ratio, agent_ratio, empty_ratio])
    return grid

def is_satisfied(grid: np.ndarray, x: int, y: int,
                threshold: float = THRESHOLD_VALUE) -> bool:
    """
    특정 위치의 행위자가 현재 상황에 만족하는지 판단합니다.

    Args:
        grid: 현재 격자 상태
        x, y: 행위자의 위치
        threshold: 만족 임계값

    Returns:
        bool: 만족 여부
    """
    # 빈 공간은 항상 만족
    if grid[x, y] == 0:
        return True

    agent_type = grid[x, y]
    similar_neighbors = 0
    total_neighbors = 0

    # 주변 8개 칸 검사 (무어 이웃)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # 자기 자신 제외
                continue

            nx, ny = x + dx, y + dy
            # 격자 경계 확인
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] == agent_type:
                    similar_neighbors += 1
                if grid[nx, ny] != 0:
                    total_neighbors += 1

    # 이웃이 없으면 만족
    if total_neighbors == 0:
        return True

    # 같은 유형 이웃 비율이 임계값 이상이면 만족
    return similar_neighbors / total_neighbors >= threshold

def move_unsatisfied(grid: np.ndarray, threshold: float = THRESHOLD_VALUE) -> np.ndarray:
    """
    불만족한 행위자들을 빈 공간으로 이동시킵니다.

    Args:
        grid: 현재 격자 상태
        threshold: 만족 임계값

    Returns:
        np.ndarray: 이동 후 격자 상태
    """
    # 불만족한 행위자 찾기
    unsatisfied_agents = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != 0 and not is_satisfied(grid, x, y, threshold):
                unsatisfied_agents.append((x, y))

    # 불만족한 행위자들을 무작위로 이동
    for x, y in unsatisfied_agents:
        # 빈 공간 찾기
        empty_spaces = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 0:
                    empty_spaces.append((i, j))

        if empty_spaces:
            # 무작위로 빈 공간 선택
            new_x, new_y = random.choice(empty_spaces)
            # 이동 실행
            grid[new_x, new_y] = grid[x, y]
            grid[x, y] = 0

    return grid

def simulate_schelling(grid: np.ndarray, steps: int = MAX_STEPS,
                      threshold: float = THRESHOLD_VALUE) -> List[np.ndarray]:
    """
    쉘링 모형 시뮬레이션을 실행합니다.

    Args:
        grid: 초기 격자 상태
        steps: 최대 시뮬레이션 단계
        threshold: 만족 임계값

    Returns:
        List[np.ndarray]: 각 단계별 격자 상태
    """
    frames = []
    frames.append(grid.copy())  # 초기 상태 저장

    for step in range(steps):
        prev_grid = grid.copy()
        grid = move_unsatisfied(grid, threshold)
        frames.append(grid.copy())

        # 조기 종료 조건 (모든 행위자가 만족하거나 변화가 없을 때)
        if np.array_equal(prev_grid, grid):
            print(f"수렴 달성. {step+1}단계에서 종료.")
            break

        # 만족도 체크
        if all(is_satisfied(grid, x, y, threshold)
               for x in range(grid.shape[0]) for y in range(grid.shape[1])):
            print(f"모든 행위자가 만족함. {step+1}단계에서 종료.")
            break

    return frames

def calculate_segregation_index(grid: np.ndarray) -> float:
    """
    분리 지수를 계산합니다 (Duncan & Duncan Index 변형).

    Args:
        grid: 격자 상태

    Returns:
        float: 0-1 사이의 분리 지수 (높을수록 더 분리됨)
    """
    total_dissimilarity = 0
    total_agents = 0

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != 0:
                agent_type = grid[x, y]
                different_neighbors = 0
                total_neighbors = 0

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            if grid[nx, ny] != 0:
                                total_neighbors += 1
                                if grid[nx, ny] != agent_type:
                                    different_neighbors += 1

                if total_neighbors > 0:
                    total_dissimilarity += different_neighbors / total_neighbors
                total_agents += 1

    return total_dissimilarity / total_agents if total_agents > 0 else 0

def calculate_satisfaction_rate(grid: np.ndarray, threshold: float = THRESHOLD_VALUE) -> float:
    """
    전체 만족도를 계산합니다.

    Args:
        grid: 격자 상태
        threshold: 만족 임계값

    Returns:
        float: 0-1 사이의 만족도
    """
    satisfied_agents = 0
    total_agents = 0

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != 0:
                total_agents += 1
                if is_satisfied(grid, x, y, threshold):
                    satisfied_agents += 1

    return satisfied_agents / total_agents if total_agents > 0 else 0

def visualize_simulation(frames: List[np.ndarray],
                        title: str = "쉘링 분리 모형 (전통적 방법)",
                        save_path: str = None) -> animation.FuncAnimation:
    """
    시뮬레이션 결과를 애니메이션으로 시각화합니다.

    Args:
        frames: 각 단계별 격자 상태
        title: 애니메이션 제목
        save_path: 저장할 경로 (None이면 저장하지 않음)

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

        # 통계 정보 표시
        seg_index = calculate_segregation_index(frames[frame_num])
        satisfaction = calculate_satisfaction_rate(frames[frame_num], THRESHOLD_VALUE)

        ax.text(0.02, 0.98, f"분리지수: {seg_index:.3f}\n만족도: {satisfaction:.3f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12)

    anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                 interval=200, repeat=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"애니메이션이 {save_path}에 저장되었습니다.")

    return anim

def analyze_results(frames: List[np.ndarray], threshold: float = THRESHOLD_VALUE) -> dict:
    """
    시뮬레이션 결과를 분석합니다.

    Args:
        frames: 각 단계별 격자 상태
        threshold: 만족 임계값

    Returns:
        dict: 분석 결과
    """
    initial_segregation = calculate_segregation_index(frames[0])
    final_segregation = calculate_segregation_index(frames[-1])
    initial_satisfaction = calculate_satisfaction_rate(frames[0], threshold)
    final_satisfaction = calculate_satisfaction_rate(frames[-1], threshold)

    return {
        "initial_segregation": initial_segregation,
        "final_segregation": final_segregation,
        "segregation_change": final_segregation - initial_segregation,
        "initial_satisfaction": initial_satisfaction,
        "final_satisfaction": final_satisfaction,
        "satisfaction_change": final_satisfaction - initial_satisfaction,
        "total_steps": len(frames) - 1
    }

def main():
    """메인 실행 함수"""
    print("=== 전통적 쉘링 분리 모형 시뮬레이션 ===")
    print(f"격자 크기: {WIDTH}x{HEIGHT}")
    print(f"빈 공간 비율: {EMPTY_RATIO}")
    print(f"만족 임계값: {THRESHOLD_VALUE}")
    print(f"최대 단계: {MAX_STEPS}")
    print()

    # 격자 초기화
    grid = initialize_grid()
    print("격자 초기화 완료")

    # 시뮬레이션 실행
    print("시뮬레이션 시작...")
    frames = simulate_schelling(grid, MAX_STEPS, THRESHOLD_VALUE)
    print(f"시뮬레이션 완료 (총 {len(frames)-1}단계)")

    # 결과 분석
    results = analyze_results(frames, THRESHOLD_VALUE)
    print("\n=== 시뮬레이션 결과 ===")
    print(f"초기 분리지수: {results['initial_segregation']:.3f}")
    print(f"최종 분리지수: {results['final_segregation']:.3f}")
    print(f"분리지수 변화: {results['segregation_change']:.3f}")
    print(f"초기 만족도: {results['initial_satisfaction']:.3f}")
    print(f"최종 만족도: {results['final_satisfaction']:.3f}")
    print(f"만족도 변화: {results['satisfaction_change']:.3f}")

    # 시각화
    print("\n시각화 중...")
    anim = visualize_simulation(frames, save_path="schelling_traditional.gif")
    plt.show()

    return frames, results

if __name__ == "__main__":
    frames, results = main()