#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 유틸리티 함수들
Common Utility Functions

이 모듈은 쉘링 모형 시뮬레이션에서 공통으로 사용되는 유틸리티 함수들을 제공합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime

def calculate_segregation_metrics(grid: np.ndarray) -> Dict[str, float]:
    """
    다양한 분리 지수를 계산합니다.

    Args:
        grid: 격자 상태

    Returns:
        Dict[str, float]: 분리 지수들
    """
    metrics = {}

    # Duncan & Duncan Index (전통적 분리 지수)
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

    metrics['duncan_index'] = total_dissimilarity / total_agents if total_agents > 0 else 0

    # Moran's I (공간적 자기상관)
    metrics['morans_i'] = calculate_morans_i(grid)

    # 클러스터링 계수
    metrics['clustering_coefficient'] = calculate_clustering_coefficient(grid)

    return metrics

def calculate_morans_i(grid: np.ndarray) -> float:
    """
    Moran's I 통계량을 계산합니다 (공간적 자기상관).

    Args:
        grid: 격자 상태

    Returns:
        float: Moran's I 값
    """
    n = grid.shape[0] * grid.shape[1]

    # 가중치 행렬 생성 (인접한 셀들만 고려)
    weights = np.zeros((n, n))
    positions = []

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            positions.append((i, j))

    for idx1, (x1, y1) in enumerate(positions):
        for idx2, (x2, y2) in enumerate(positions):
            if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 and (x1, y1) != (x2, y2):
                weights[idx1, idx2] = 1

    # 값들을 1차원으로 변환
    values = grid.flatten()

    # 빈 공간 제외
    non_empty_mask = values != 0
    values = values[non_empty_mask]
    weights = weights[non_empty_mask][:, non_empty_mask]

    if len(values) == 0:
        return 0

    # Moran's I 계산
    n = len(values)
    mean_val = np.mean(values)

    numerator = 0
    denominator = 0
    w_sum = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                w_ij = weights[i, j]
                numerator += w_ij * (values[i] - mean_val) * (values[j] - mean_val)
                w_sum += w_ij
        denominator += (values[i] - mean_val) ** 2

    if w_sum == 0 or denominator == 0:
        return 0

    morans_i = (n / w_sum) * (numerator / denominator)
    return morans_i

def calculate_clustering_coefficient(grid: np.ndarray) -> float:
    """
    클러스터링 계수를 계산합니다.

    Args:
        grid: 격자 상태

    Returns:
        float: 클러스터링 계수
    """
    total_clustering = 0
    agent_count = 0

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != 0:
                agent_type = grid[x, y]

                # 같은 유형의 이웃들 찾기
                same_neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]
                            and grid[nx, ny] == agent_type):
                            same_neighbors.append((nx, ny))

                # 클러스터링 계수 계산
                if len(same_neighbors) >= 2:
                    possible_connections = len(same_neighbors) * (len(same_neighbors) - 1) / 2
                    actual_connections = 0

                    for i, (x1, y1) in enumerate(same_neighbors):
                        for j, (x2, y2) in enumerate(same_neighbors[i+1:], i+1):
                            if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                                actual_connections += 1

                    total_clustering += actual_connections / possible_connections

                agent_count += 1

    return total_clustering / agent_count if agent_count > 0 else 0

def compare_simulations(frames_traditional: List[np.ndarray],
                       frames_llm: List[np.ndarray],
                       labels: List[str] = None) -> Dict:
    """
    두 시뮬레이션 결과를 비교합니다.

    Args:
        frames_traditional: 전통적 방법 결과
        frames_llm: LLM 방법 결과
        labels: 비교 대상 레이블

    Returns:
        Dict: 비교 결과
    """
    if labels is None:
        labels = ["Traditional", "LLM"]

    comparison = {
        "labels": labels,
        "steps": [len(frames_traditional) - 1, len(frames_llm) - 1],
        "metrics": {
            "initial": {},
            "final": {}
        }
    }

    # 초기 상태 비교
    initial_metrics_trad = calculate_segregation_metrics(frames_traditional[0])
    initial_metrics_llm = calculate_segregation_metrics(frames_llm[0])

    comparison["metrics"]["initial"] = {
        labels[0]: initial_metrics_trad,
        labels[1]: initial_metrics_llm
    }

    # 최종 상태 비교
    final_metrics_trad = calculate_segregation_metrics(frames_traditional[-1])
    final_metrics_llm = calculate_segregation_metrics(frames_llm[-1])

    comparison["metrics"]["final"] = {
        labels[0]: final_metrics_trad,
        labels[1]: final_metrics_llm
    }

    return comparison

def create_comparison_plot(comparison: Dict, save_path: str = None):
    """
    시뮬레이션 비교 결과를 시각화합니다.

    Args:
        comparison: 비교 결과 딕셔너리
        save_path: 저장할 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.rcParams['font.family'] = 'DejaVu Sans'

    labels = comparison["labels"]

    # Duncan Index 비교
    ax1 = axes[0, 0]
    initial_duncan = [comparison["metrics"]["initial"][label]["duncan_index"] for label in labels]
    final_duncan = [comparison["metrics"]["final"][label]["duncan_index"] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, initial_duncan, width, label='Initial', alpha=0.7)
    ax1.bar(x + width/2, final_duncan, width, label='Final', alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Duncan Index')
    ax1.set_title('Duncan Segregation Index Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Moran's I 비교
    ax2 = axes[0, 1]
    initial_morans = [comparison["metrics"]["initial"][label]["morans_i"] for label in labels]
    final_morans = [comparison["metrics"]["final"][label]["morans_i"] for label in labels]

    ax2.bar(x - width/2, initial_morans, width, label='Initial', alpha=0.7)
    ax2.bar(x + width/2, final_morans, width, label='Final', alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel("Moran's I")
    ax2.set_title("Moran's I Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    # 클러스터링 계수 비교
    ax3 = axes[1, 0]
    initial_clustering = [comparison["metrics"]["initial"][label]["clustering_coefficient"] for label in labels]
    final_clustering = [comparison["metrics"]["final"][label]["clustering_coefficient"] for label in labels]

    ax3.bar(x - width/2, initial_clustering, width, label='Initial', alpha=0.7)
    ax3.bar(x + width/2, final_clustering, width, label='Final', alpha=0.7)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Clustering Coefficient')
    ax3.set_title('Clustering Coefficient Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()

    # 단계 수 비교
    ax4 = axes[1, 1]
    steps = comparison["steps"]
    ax4.bar(labels, steps, alpha=0.7)
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Steps to Convergence')
    ax4.set_title('Convergence Speed Comparison')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"비교 차트가 {save_path}에 저장되었습니다.")

    plt.show()

def save_simulation_data(frames: List[np.ndarray],
                        metadata: Dict,
                        filename: str):
    """
    시뮬레이션 데이터를 저장합니다.

    Args:
        frames: 시뮬레이션 프레임들
        metadata: 메타데이터
        filename: 저장할 파일명
    """
    data = {
        "metadata": metadata,
        "frames": [frame.tolist() for frame in frames],
        "timestamp": datetime.now().isoformat()
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"시뮬레이션 데이터가 {filename}에 저장되었습니다.")

def load_simulation_data(filename: str) -> Tuple[List[np.ndarray], Dict]:
    """
    시뮬레이션 데이터를 불러옵니다.

    Args:
        filename: 불러올 파일명

    Returns:
        Tuple[List[np.ndarray], Dict]: 프레임들과 메타데이터
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = [np.array(frame) for frame in data["frames"]]
    metadata = data["metadata"]

    return frames, metadata

def create_side_by_side_animation(frames1: List[np.ndarray],
                                 frames2: List[np.ndarray],
                                 titles: List[str],
                                 save_path: str = None) -> animation.FuncAnimation:
    """
    두 시뮬레이션을 나란히 보여주는 애니메이션을 생성합니다.

    Args:
        frames1: 첫 번째 시뮬레이션 프레임들
        frames2: 두 번째 시뮬레이션 프레임들
        titles: 각 시뮬레이션의 제목
        save_path: 저장할 경로

    Returns:
        animation.FuncAnimation: 애니메이션 객체
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 색상 맵 설정
    colors = ['white', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    # 프레임 수 맞추기
    max_frames = max(len(frames1), len(frames2))

    def animate(frame_num):
        ax1.clear()
        ax2.clear()

        # 첫 번째 시뮬레이션
        frame1_idx = min(frame_num, len(frames1) - 1)
        ax1.imshow(frames1[frame1_idx], cmap=cmap, vmin=0, vmax=2)
        ax1.set_title(f"{titles[0]} - {frame1_idx}단계", fontsize=12)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # 두 번째 시뮬레이션
        frame2_idx = min(frame_num, len(frames2) - 1)
        ax2.imshow(frames2[frame2_idx], cmap=cmap, vmin=0, vmax=2)
        ax2.set_title(f"{titles[1]} - {frame2_idx}단계", fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # 통계 정보
        metrics1 = calculate_segregation_metrics(frames1[frame1_idx])
        metrics2 = calculate_segregation_metrics(frames2[frame2_idx])

        ax1.text(0.02, 0.98, f"Duncan: {metrics1['duncan_index']:.3f}",
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax2.text(0.02, 0.98, f"Duncan: {metrics2['duncan_index']:.3f}",
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    anim = animation.FuncAnimation(fig, animate, frames=max_frames,
                                 interval=300, repeat=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=3)
        print(f"비교 애니메이션이 {save_path}에 저장되었습니다.")

    return anim

def generate_parameter_study(parameter_ranges: Dict,
                           simulation_function,
                           base_params: Dict) -> pd.DataFrame:
    """
    매개변수 연구를 수행합니다.

    Args:
        parameter_ranges: 매개변수 범위 딕셔너리
        simulation_function: 시뮬레이션 함수
        base_params: 기본 매개변수

    Returns:
        pd.DataFrame: 결과 데이터프레임
    """
    results = []

    for param_name, param_values in parameter_ranges.items():
        for param_value in param_values:
            print(f"실행 중: {param_name} = {param_value}")

            # 매개변수 설정
            params = base_params.copy()
            params[param_name] = param_value

            # 시뮬레이션 실행
            frames = simulation_function(**params)

            # 결과 분석
            initial_metrics = calculate_segregation_metrics(frames[0])
            final_metrics = calculate_segregation_metrics(frames[-1])

            result = {
                'parameter': param_name,
                'value': param_value,
                'steps': len(frames) - 1,
                'initial_duncan': initial_metrics['duncan_index'],
                'final_duncan': final_metrics['duncan_index'],
                'duncan_change': final_metrics['duncan_index'] - initial_metrics['duncan_index'],
                'initial_morans': initial_metrics['morans_i'],
                'final_morans': final_metrics['morans_i'],
                'initial_clustering': initial_metrics['clustering_coefficient'],
                'final_clustering': final_metrics['clustering_coefficient']
            }

            results.append(result)

    return pd.DataFrame(results)

def create_parameter_study_plot(df: pd.DataFrame, save_path: str = None):
    """
    매개변수 연구 결과를 시각화합니다.

    Args:
        df: 결과 데이터프레임
        save_path: 저장할 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.rcParams['font.family'] = 'DejaVu Sans'

    parameters = df['parameter'].unique()

    for i, param in enumerate(parameters):
        param_data = df[df['parameter'] == param]

        ax = axes[i // 2, i % 2]
        ax.plot(param_data['value'], param_data['final_duncan'], 'o-', label='Final Duncan Index')
        ax.plot(param_data['value'], param_data['initial_duncan'], 's--', alpha=0.7, label='Initial Duncan Index')
        ax.set_xlabel(param)
        ax.set_ylabel('Duncan Index')
        ax.set_title(f'Effect of {param} on Segregation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"매개변수 연구 차트가 {save_path}에 저장되었습니다.")

    plt.show()

def print_summary_statistics(frames: List[np.ndarray],
                           title: str = "시뮬레이션 결과"):
    """
    시뮬레이션 결과의 요약 통계를 출력합니다.

    Args:
        frames: 시뮬레이션 프레임들
        title: 제목
    """
    print(f"\n=== {title} ===")
    print(f"총 단계 수: {len(frames) - 1}")

    initial_metrics = calculate_segregation_metrics(frames[0])
    final_metrics = calculate_segregation_metrics(frames[-1])

    print(f"\n초기 상태:")
    print(f"  Duncan Index: {initial_metrics['duncan_index']:.4f}")
    print(f"  Moran's I: {initial_metrics['morans_i']:.4f}")
    print(f"  Clustering Coefficient: {initial_metrics['clustering_coefficient']:.4f}")

    print(f"\n최종 상태:")
    print(f"  Duncan Index: {final_metrics['duncan_index']:.4f}")
    print(f"  Moran's I: {final_metrics['morans_i']:.4f}")
    print(f"  Clustering Coefficient: {final_metrics['clustering_coefficient']:.4f}")

    print(f"\n변화량:")
    print(f"  Duncan Index: {final_metrics['duncan_index'] - initial_metrics['duncan_index']:.4f}")
    print(f"  Moran's I: {final_metrics['morans_i'] - initial_metrics['morans_i']:.4f}")
    print(f"  Clustering Coefficient: {final_metrics['clustering_coefficient'] - initial_metrics['clustering_coefficient']:.4f}")

if __name__ == "__main__":
    # 테스트 코드
    print("유틸리티 함수 테스트")

    # 샘플 격자 생성
    test_grid = np.random.choice([0, 1, 2], size=(10, 10), p=[0.2, 0.4, 0.4])

    # 메트릭 계산 테스트
    metrics = calculate_segregation_metrics(test_grid)
    print("계산된 메트릭:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")