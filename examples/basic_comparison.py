#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 비교 실습 - 최적화된 조건
Basic Comparison Exercise - Optimized Conditions

전통적 쉘링 모형과 LLM 기반 쉘링 모형의 기본적인 차이를 비교하는 실습입니다.
최적 조건: threshold=0.6, empty_ratio=0.10
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schelling_traditional import *
from schelling_llm import *
from utils import *
import matplotlib.pyplot as plt
import time
import json

def main():
    """기본 비교 실습 메인 함수"""
    print("=== 기본 비교 실습 (최적화된 조건) ===")
    print("전통적 방법과 LLM 방법을 비교합니다.")

    # 최적화된 매개변수 설정
    grid_size = 15
    empty_ratio = 0.10  # 10% 빈 공간 (제약된 공간)
    threshold = 0.6     # 60% 임계값 (극적 분리)
    max_steps = 30

    print(f"\n매개변수 설정 (최적화된 조건):")
    print(f"- 격자 크기: {grid_size}x{grid_size}")
    print(f"- 빈 공간 비율: {empty_ratio} (제약된 공간 전략)")
    print(f"- 만족 임계값: {threshold} (극적 분리 조건)")
    print(f"- 최대 단계: {max_steps}")

    # 출력 디렉토리 생성
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. 전통적 방법 실행
    print("\n1. 전통적 쉘링 모형 실행 중...")
    start_time = time.time()
    grid_traditional = initialize_grid(grid_size, grid_size, empty_ratio)
    initial_grid_traditional = grid_traditional.copy()
    frames_traditional = simulate_schelling(grid_traditional, max_steps, threshold)
    traditional_time = time.time() - start_time

    # 2. LLM 방법 실행
    print("\n2. LLM 쉘링 모형 실행 중...")
    start_time = time.time()
    sim_llm = LLMSchellingSimulation(grid_size, grid_size, empty_ratio)
    sim_llm.initialize_grid()  # 격자 초기화
    initial_grid_llm = sim_llm.grid.copy()
    frames_llm = sim_llm.run_simulation(max_steps, threshold, "simple")
    llm_time = time.time() - start_time

    # 3. 결과 분석
    print("\n3. 결과 분석 중...")

    # 전통적 방법 결과
    results_traditional = analyze_results(frames_traditional, threshold)
    print_summary_statistics(frames_traditional, "전통적 방법")

    # LLM 방법 결과
    print_summary_statistics(frames_llm, "LLM 방법")

    # 4. 상세 비교 분석
    print("\n4. 상세 비교 분석 중...")

    # 초기 상태 비교
    initial_stats_trad = calculate_segregation_metrics(initial_grid_traditional)
    initial_stats_llm = calculate_segregation_metrics(initial_grid_llm)

    # 최종 상태 비교
    final_stats_trad = calculate_segregation_metrics(frames_traditional[-1])
    final_stats_llm = calculate_segregation_metrics(frames_llm[-1])

    # 진행 과정 분석
    trad_history = []
    llm_history = []

    for i, frame in enumerate(frames_traditional):
        stats = calculate_segregation_metrics(frame)
        trad_history.append({
            'step': i,
            'duncan_index': stats['duncan_index'],
            'morans_i': stats['morans_i'],
            'clustering_coefficient': stats['clustering_coefficient']
        })

    for i, frame in enumerate(frames_llm):
        stats = calculate_segregation_metrics(frame)
        llm_history.append({
            'step': i,
            'duncan_index': stats['duncan_index'],
            'morans_i': stats['morans_i'],
            'clustering_coefficient': stats['clustering_coefficient']
        })

    # 5. 시각화
    print("\n5. 시각화 중...")

    # 나란히 애니메이션
    anim_comparison = create_side_by_side_animation(
        frames_traditional, frames_llm,
        ["전통적 방법", "LLM 방법"],
        save_path="outputs/basic_comparison_optimized.gif"
    )

    # 상세 비교 차트 생성
    create_detailed_comparison_plot(trad_history, llm_history, traditional_time, llm_time)

    # 초기/최종 상태 비교 시각화
    create_initial_final_comparison(initial_grid_traditional, frames_traditional[-1],
                                  initial_grid_llm, frames_llm[-1])

    # 6. 결과 저장
    print("\n6. 결과 저장 중...")

    # 전통적 방법 데이터 저장
    save_simulation_data(frames_traditional, {
        "method": "traditional",
        "grid_size": grid_size,
        "empty_ratio": empty_ratio,
        "threshold": threshold,
        "max_steps": max_steps,
        "execution_time": traditional_time,
        "initial_stats": initial_stats_trad,
        "final_stats": final_stats_trad
    }, "data/traditional_results_optimized.json")

    # LLM 방법 데이터 저장
    save_simulation_data(frames_llm, {
        "method": "llm_simple",
        "grid_size": grid_size,
        "empty_ratio": empty_ratio,
        "threshold": threshold,
        "max_steps": max_steps,
        "execution_time": llm_time,
        "initial_stats": initial_stats_llm,
        "final_stats": final_stats_llm
    }, "data/llm_results_optimized.json")

    # 7. 주요 차이점 요약
    print("\n=== 주요 차이점 요약 (최적화된 조건) ===")

    print(f"실행 시간:")
    print(f"  전통적 방법: {traditional_time:.2f}초")
    print(f"  LLM 방법: {llm_time:.2f}초")
    print(f"  속도 비율: {llm_time/traditional_time:.1f}x")

    print(f"\n수렴 속도:")
    print(f"  전통적 방법: {len(frames_traditional)-1} 단계")
    print(f"  LLM 방법: {len(frames_llm)-1} 단계")

    print(f"\n최종 분리 지수 (Duncan Index):")
    print(f"  전통적 방법: {final_stats_trad['duncan_index']:.4f}")
    print(f"  LLM 방법: {final_stats_llm['duncan_index']:.4f}")
    print(f"  차이: {abs(final_stats_trad['duncan_index'] - final_stats_llm['duncan_index']):.4f}")

    print(f"\n공간적 자기상관 (Moran's I):")
    print(f"  전통적 방법: {final_stats_trad['morans_i']:.4f}")
    print(f"  LLM 방법: {final_stats_llm['morans_i']:.4f}")
    print(f"  차이: {abs(final_stats_trad['morans_i'] - final_stats_llm['morans_i']):.4f}")

    print(f"\n클러스터링 계수:")
    print(f"  전통적 방법: {final_stats_trad['clustering_coefficient']:.4f}")
    print(f"  LLM 방법: {final_stats_llm['clustering_coefficient']:.4f}")
    print(f"  차이: {abs(final_stats_trad['clustering_coefficient'] - final_stats_llm['clustering_coefficient']):.4f}")

    # 8. LLM 응답 분석 (API 키가 있는 경우)
    if hasattr(sim_llm, 'get_agent_responses'):
        responses = sim_llm.get_agent_responses()
        if responses:
            print(f"\n=== LLM 응답 분석 ===")
            print(f"총 응답 수: {len(responses)}")

            # 만족/불만족 비율
            satisfied_count = sum(1 for r in responses if r['response']['satisfied'])
            print(f"만족 응답 비율: {satisfied_count/len(responses):.2%}")

            # 응답 예시
            print("\n응답 예시:")
            for i, resp in enumerate(responses[:3]):
                print(f"  {i+1}. 위치 {resp['position']}, 유형 {resp['agent_type']}")
                print(f"     응답: {resp['response']['response']}")

    # 9. 비교 리포트 생성
    print("\n9. 비교 리포트 생성 중...")
    create_comparison_report(
        traditional_time, llm_time,
        initial_stats_trad, final_stats_trad,
        initial_stats_llm, final_stats_llm,
        len(frames_traditional), len(frames_llm),
        grid_size, empty_ratio, threshold
    )

    print("\n=== 실습 완료 ===")
    print("결과 파일들:")
    print("- outputs/basic_comparison_optimized.gif: 비교 애니메이션")
    print("- outputs/detailed_comparison_plot.png: 상세 비교 차트")
    print("- outputs/initial_final_comparison.png: 초기/최종 상태 비교")
    print("- outputs/comparison_report.md: 상세 비교 리포트")
    print("- data/traditional_results_optimized.json: 전통적 방법 데이터")
    print("- data/llm_results_optimized.json: LLM 방법 데이터")

    return frames_traditional, frames_llm, trad_history, llm_history

def create_detailed_comparison_plot(trad_history, llm_history, trad_time, llm_time):
    """상세 비교 차트 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('전통적 방법 vs LLM 방법 상세 비교 (최적화된 조건)', fontsize=16, fontweight='bold')

    # Duncan Index 비교
    trad_duncan = [h['duncan_index'] for h in trad_history]
    llm_duncan = [h['duncan_index'] for h in llm_history]

    axes[0, 0].plot(range(len(trad_duncan)), trad_duncan, 'b-', label='전통적 방법', linewidth=2)
    axes[0, 0].plot(range(len(llm_duncan)), llm_duncan, 'r-', label='LLM 방법', linewidth=2)
    axes[0, 0].set_title('Duncan Index 변화', fontweight='bold')
    axes[0, 0].set_xlabel('단계')
    axes[0, 0].set_ylabel('Duncan Index')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Moran's I 비교
    trad_moran = [h['morans_i'] for h in trad_history]
    llm_moran = [h['morans_i'] for h in llm_history]

    axes[0, 1].plot(range(len(trad_moran)), trad_moran, 'b-', label='전통적 방법', linewidth=2)
    axes[0, 1].plot(range(len(llm_moran)), llm_moran, 'r-', label='LLM 방법', linewidth=2)
    axes[0, 1].set_title('Moran\'s I 변화', fontweight='bold')
    axes[0, 1].set_xlabel('단계')
    axes[0, 1].set_ylabel('Moran\'s I')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 클러스터링 계수 비교
    trad_clustering = [h['clustering_coefficient'] for h in trad_history]
    llm_clustering = [h['clustering_coefficient'] for h in llm_history]

    axes[1, 0].plot(range(len(trad_clustering)), trad_clustering, 'b-', label='전통적 방법', linewidth=2)
    axes[1, 0].plot(range(len(llm_clustering)), llm_clustering, 'r-', label='LLM 방법', linewidth=2)
    axes[1, 0].set_title('클러스터링 계수 변화', fontweight='bold')
    axes[1, 0].set_xlabel('단계')
    axes[1, 0].set_ylabel('클러스터링 계수')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 실행 시간 비교
    methods = ['전통적 방법', 'LLM 방법']
    times = [trad_time, llm_time]
    colors = ['blue', 'red']

    bars = axes[1, 1].bar(methods, times, color=colors, alpha=0.7)
    axes[1, 1].set_title('실행 시간 비교', fontweight='bold')
    axes[1, 1].set_ylabel('시간 (초)')

    # 막대 위에 값 표시
    for bar, time_val in zip(bars, times):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{time_val:.2f}초', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/detailed_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_initial_final_comparison(initial_trad, final_trad, initial_llm, final_llm):
    """초기/최종 상태 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('초기 상태 vs 최종 상태 비교', fontsize=16, fontweight='bold')

    # 전통적 방법 초기 상태
    im1 = axes[0, 0].imshow(initial_trad, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('전통적 방법 - 초기 상태', fontweight='bold')
    axes[0, 0].axis('off')

    # 전통적 방법 최종 상태
    im2 = axes[0, 1].imshow(final_trad, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[0, 1].set_title('전통적 방법 - 최종 상태', fontweight='bold')
    axes[0, 1].axis('off')

    # LLM 방법 초기 상태
    im3 = axes[1, 0].imshow(initial_llm, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1, 0].set_title('LLM 방법 - 초기 상태', fontweight='bold')
    axes[1, 0].axis('off')

    # LLM 방법 최종 상태
    im4 = axes[1, 1].imshow(final_llm, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('LLM 방법 - 최종 상태', fontweight='bold')
    axes[1, 1].axis('off')

    # 컬러바 추가
    cbar = plt.colorbar(im4, ax=axes, shrink=0.6)
    cbar.set_label('에이전트 타입 (빨강: 그룹 A, 파랑: 그룹 B, 흰색: 빈 공간)', fontsize=12)

    plt.tight_layout()
    plt.savefig('outputs/initial_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_report(trad_time, llm_time, initial_trad, final_trad,
                           initial_llm, final_llm, trad_steps, llm_steps,
                           grid_size, empty_ratio, threshold):
    """상세 비교 리포트 생성"""

    report = f"""# 전통적 방법 vs LLM 방법 비교 리포트

## 🎯 실험 조건 (최적화된 설정)
- **격자 크기**: {grid_size}×{grid_size}
- **빈 공간 비율**: {empty_ratio} (제약된 공간 전략)
- **임계값**: {threshold} (극적 분리 조건)
- **최대 단계**: 30

## ⏱️ 성능 비교

### 실행 시간
- **전통적 방법**: {trad_time:.2f}초
- **LLM 방법**: {llm_time:.2f}초
- **속도 비율**: {llm_time/trad_time:.1f}x (LLM이 {'빠름' if llm_time < trad_time else '느림'})

### 수렴 속도
- **전통적 방법**: {trad_steps-1} 단계
- **LLM 방법**: {llm_steps-1} 단계
- **차이**: {abs(trad_steps - llm_steps)} 단계

## 📊 분리 패턴 분석

### 초기 상태 비교
| 지표 | 전통적 방법 | LLM 방법 | 차이 |
|------|-------------|----------|------|
| Duncan Index | {initial_trad['duncan_index']:.4f} | {initial_llm['duncan_index']:.4f} | {abs(initial_trad['duncan_index'] - initial_llm['duncan_index']):.4f} |
| Moran's I | {initial_trad['morans_i']:.4f} | {initial_llm['morans_i']:.4f} | {abs(initial_trad['morans_i'] - initial_llm['morans_i']):.4f} |
| 클러스터링 계수 | {initial_trad['clustering_coefficient']:.4f} | {initial_llm['clustering_coefficient']:.4f} | {abs(initial_trad['clustering_coefficient'] - initial_llm['clustering_coefficient']):.4f} |

### 최종 상태 비교
| 지표 | 전통적 방법 | LLM 방법 | 차이 |
|------|-------------|----------|------|
| Duncan Index | {final_trad['duncan_index']:.4f} | {final_llm['duncan_index']:.4f} | {abs(final_trad['duncan_index'] - final_llm['duncan_index']):.4f} |
| Moran's I | {final_trad['morans_i']:.4f} | {final_llm['morans_i']:.4f} | {abs(final_trad['morans_i'] - final_llm['morans_i']):.4f} |
| 클러스터링 계수 | {final_trad['clustering_coefficient']:.4f} | {final_llm['clustering_coefficient']:.4f} | {abs(final_trad['clustering_coefficient'] - final_llm['clustering_coefficient']):.4f} |

## 📁 생성된 파일들
- `basic_comparison_optimized.gif`: 나란히 애니메이션
- `detailed_comparison_plot.png`: 상세 비교 차트
- `initial_final_comparison.png`: 초기/최종 상태 비교
- `traditional_results_optimized.json`: 전통적 방법 데이터
- `llm_results_optimized.json`: LLM 방법 데이터

---
*이 리포트는 최적화된 조건(threshold=0.6, empty_ratio=0.10)에서의 비교 분석 결과입니다.*
"""

    with open('outputs/comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    frames_traditional, frames_llm, trad_history, llm_history = main()