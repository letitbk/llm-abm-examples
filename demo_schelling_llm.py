#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 기반 쉘링 분리 모형 (데모 버전)
LLM-based Schelling Segregation Model (Demo Version)

이 모듈은 API 호출 없이 빠르게 실행되는 데모 버전입니다.
실제 LLM 기능은 사용하지 않고 fallback 모드로만 동작합니다.
"""

import os
import sys

# API 키를 임시로 비활성화하여 fallback 모드로 실행
original_api_key = os.environ.get('OPENAI_API_KEY')
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

# 메인 모듈 임포트
from schelling_llm import LLMSchellingSimulation, setup_korean_font, visualize_llm_simulation
import time

# 데모용 매개변수
DEMO_WIDTH, DEMO_HEIGHT = 12, 12
DEMO_EMPTY_RATIO = 0.4
DEMO_THRESHOLD = 0.3
DEMO_MAX_STEPS = 8

def run_demo():
    """데모 시뮬레이션 실행"""
    print("=== LLM 쉘링 분리 모형 데모 ===")
    print("주의: 이 데모는 API 호출 없이 fallback 모드로 실행됩니다.")
    print("실제 LLM 기능을 사용하려면 원본 schelling_llm.py를 사용하세요.")
    print()

    print(f"격자 크기: {DEMO_WIDTH}x{DEMO_HEIGHT}")
    print(f"빈 공간 비율: {DEMO_EMPTY_RATIO}")
    print(f"임계값: {DEMO_THRESHOLD}")
    print(f"최대 단계: {DEMO_MAX_STEPS}")
    print()

    # 시뮬레이션 생성
    sim = LLMSchellingSimulation(DEMO_WIDTH, DEMO_HEIGHT, DEMO_EMPTY_RATIO)
    sim.initialize_grid()

    print(f"총 행위자 수: {len(sim.agents)}")
    print()

    # 단순 모드 시뮬레이션
    print("1. 단순 모드 시뮬레이션 (fallback)")
    start_time = time.time()
    frames_simple = sim.run_simulation(DEMO_MAX_STEPS, DEMO_THRESHOLD, "simple")
    end_time = time.time()

    print(f"실행 시간: {end_time - start_time:.2f}초")
    print(f"총 프레임 수: {len(frames_simple)}")
    print()

    # 집단 모드 시뮬레이션
    print("2. 집단 모드 시뮬레이션 (fallback)")
    sim2 = LLMSchellingSimulation(DEMO_WIDTH, DEMO_HEIGHT, DEMO_EMPTY_RATIO)

    start_time = time.time()
    frames_group = sim2.run_simulation(DEMO_MAX_STEPS, DEMO_THRESHOLD, "group")
    end_time = time.time()

    print(f"실행 시간: {end_time - start_time:.2f}초")
    print(f"총 프레임 수: {len(frames_group)}")
    print()

    # 시각화
    print("3. 시각화 생성 중...")
    anim1 = visualize_llm_simulation(frames_simple,
                                   "쉘링 분리 모형 데모 (단순 모드)",
                                   "demo_simple.gif")

    anim2 = visualize_llm_simulation(frames_group,
                                   "쉘링 분리 모형 데모 (집단 모드)",
                                   "demo_group.gif")

    print("데모 완료!")
    print()
    print("생성된 파일:")
    print("- demo_simple.gif: 단순 모드 애니메이션")
    print("- demo_group.gif: 집단 모드 애니메이션")
    print()
    print("실제 LLM 기능을 사용하려면:")
    print("1. OPENAI_API_KEY 환경변수를 설정하세요")
    print("2. python schelling_llm.py를 실행하세요")

    return frames_simple, frames_group

if __name__ == "__main__":
    # 한글 폰트 설정
    setup_korean_font()

    # 데모 실행
    frames_simple, frames_group = run_demo()

    # 원래 API 키 복원
    if original_api_key:
        os.environ['OPENAI_API_KEY'] = original_api_key