#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 쉘링 시뮬레이션 비용 계산기
LLM Schelling Simulation Cost Calculator

이 스크립트는 OpenAI API를 사용한 LLM 쉘링 시뮬레이션의 예상 비용을 계산합니다.
"""

import argparse
from typing import Dict, Tuple

# OpenAI API 가격 정보 (2025년 6월 기준, USD)
# 출처: OpenAI 공식 API 문서 및 검증된 제3자 소스
# 주의: 가격은 수시로 변동될 수 있으므로 https://openai.com/pricing 에서 최신 정보 확인 필요
#
# 🚨 주요 업데이트 (2025년 6월):
# - o3 모델 가격 80% 인하 적용
# - 새로운 모델들 (GPT-4.1, o3-pro 등) 추가
# - GPT-4.5는 2025년 7월 14일 폐지 예정
API_PRICING = {
    "gpt-4o-mini": {
        "input": 0.00015, # per 1K tokens (verified: $0.15 per 1M tokens)
        "output": 0.0006  # per 1K tokens (verified: $0.60 per 1M tokens)
    },
    "gpt-3.5-turbo": {
        "input": 0.0015, # per 1K tokens
        "output": 0.002  # per 1K tokens
    },
    "gpt-4o": {
        "input": 0.0025, # per 1K tokens ($2.50 per 1M tokens)
        "output": 0.005  # per 1K tokens ($5.00 per 1M tokens)
    },
    "gpt-4.1": {
        "input": 0.002,  # per 1K tokens ($2.00 per 1M tokens)
        "output": 0.008  # per 1K tokens ($8.00 per 1M tokens)
    },
    "gpt-4.1-mini": {
        "input": 0.0004, # per 1K tokens ($0.40 per 1M tokens)
        "output": 0.0016 # per 1K tokens ($1.60 per 1M tokens)
    },
    "gpt-4.1-nano": {
        "input": 0.0001, # per 1K tokens ($0.10 per 1M tokens) - ultra-low cost nano model
        "output": 0.0004 # per 1K tokens ($0.40 per 1M tokens) - ultra-low cost nano model
    },
    "gpt-4.1-nano-2025-04-14": {
        "input": 0.0001, # per 1K tokens ($0.10 per 1M tokens) - ultra-low cost nano model
        "output": 0.0004 # per 1K tokens ($0.40 per 1M tokens) - ultra-low cost nano model
    },
    "o3": {
        "input": 0.002,  # per 1K tokens ($2.00 per 1M tokens) - 80% 인하!
        "output": 0.008  # per 1K tokens ($8.00 per 1M tokens) - 80% 인하!
    },
    "o3-pro": {
        "input": 0.02,   # per 1K tokens ($20.00 per 1M tokens)
        "output": 0.08   # per 1K tokens ($80.00 per 1M tokens)
    },
    "o1": {
        "input": 0.015,  # per 1K tokens ($15.00 per 1M tokens)
        "output": 0.06   # per 1K tokens ($60.00 per 1M tokens)
    },
    "gpt-4": {
        "input": 0.03,   # per 1K tokens
        "output": 0.06   # per 1K tokens
    }
}

def estimate_tokens_per_call(agent_type: str = "simple") -> Tuple[int, int]:
    """
    API 호출당 예상 토큰 수를 계산합니다.

    Args:
        agent_type: 에이전트 유형 ("simple" 또는 "group")

    Returns:
        Tuple[int, int]: (입력 토큰, 출력 토큰)
    """
    if agent_type == "simple":
        # 단순 에이전트: 짧은 프롬프트 + 간단한 응답
        input_tokens = 200   # 격자 정보 + 기본 프롬프트
        output_tokens = 10   # "만족" 또는 "불만족"
    elif agent_type == "group":
        # 집단 에이전트: 긴 프롬프트 + 상세한 응답
        input_tokens = 350   # 격자 정보 + 집단 특성 + 상세 프롬프트
        output_tokens = 50   # 만족도 + 이유 설명
    else:
        # 기본값
        input_tokens = 250
        output_tokens = 25

    return input_tokens, output_tokens

def calculate_simulation_cost(width: int, height: int, empty_ratio: float,
                            max_steps: int, threshold: float = 0.3,
                            agent_type: str = "simple",
                            model: str = "gpt-4.1-nano-2025-04-14") -> Dict:
    """
    시뮬레이션 전체 비용을 계산합니다.

    Args:
        width, height: 격자 크기
        empty_ratio: 빈 공간 비율
        max_steps: 최대 단계 수
        threshold: 만족 임계값
        agent_type: 에이전트 유형
        model: 사용할 OpenAI 모델

    Returns:
        Dict: 비용 분석 결과
    """
    # 기본 계산
    total_cells = width * height
    agent_count = int(total_cells * (1 - empty_ratio))

    # 토큰 추정
    input_tokens, output_tokens = estimate_tokens_per_call(agent_type)

    # 시뮬레이션 패턴 추정
    # 일반적으로 초기 단계에서 더 많은 에이전트가 이동함
    avg_unsatisfied_ratio = 0.3  # 평균 30%의 에이전트가 불만족
    convergence_factor = 0.7     # 70% 확률로 조기 수렴

    # API 호출 수 계산
    calls_per_step = agent_count  # 모든 에이전트가 만족도 확인
    expected_steps = max_steps * convergence_factor
    total_api_calls = int(calls_per_step * expected_steps)

    # 만족도 계산 추가 호출 (샘플링)
    satisfaction_checks = int(expected_steps / 3)  # 3단계마다 체크
    sample_size = min(50, agent_count // 2)
    satisfaction_calls = satisfaction_checks * sample_size

    total_api_calls += satisfaction_calls

    # 토큰 계산
    total_input_tokens = total_api_calls * input_tokens
    total_output_tokens = total_api_calls * output_tokens

    # 비용 계산
    if model not in API_PRICING:
        raise ValueError(f"Unknown model: {model}")

    pricing = API_PRICING[model]
    input_cost = (total_input_tokens / 1000) * pricing["input"]
    output_cost = (total_output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    # 결과 반환
    return {
        "grid_size": f"{width}x{height}",
        "agent_count": agent_count,
        "empty_ratio": empty_ratio,
        "max_steps": max_steps,
        "expected_steps": expected_steps,
        "agent_type": agent_type,
        "model": model,
        "total_api_calls": total_api_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def print_cost_analysis(cost_info: Dict):
    """비용 분석 결과를 출력합니다."""
    print("=" * 60)
    print("🔍 LLM 쉘링 시뮬레이션 비용 분석")
    print("=" * 60)
    print("⚠️  가격 정보: 2025년 6월 기준, 실제 가격은 변동될 수 있습니다")
    print("📍 최신 가격 확인: https://openai.com/pricing")
    print("🔄 주요 업데이트: o3 모델 가격 80% 인하, 새로운 모델들 추가")
    print()

    print("📊 시뮬레이션 설정:")
    print(f"  - 격자 크기: {cost_info['grid_size']}")
    print(f"  - 에이전트 수: {cost_info['agent_count']:,}개")
    print(f"  - 빈 공간 비율: {cost_info['empty_ratio']:.1%}")
    print(f"  - 최대 단계: {cost_info['max_steps']}단계")
    print(f"  - 예상 실행 단계: {cost_info['expected_steps']:.1f}단계")
    print(f"  - 에이전트 유형: {cost_info['agent_type']}")
    print(f"  - 사용 모델: {cost_info['model']}")
    print()

    print("🔢 API 사용량:")
    print(f"  - 총 API 호출: {cost_info['total_api_calls']:,}회")
    print(f"  - 입력 토큰: {cost_info['total_input_tokens']:,}개")
    print(f"  - 출력 토큰: {cost_info['total_output_tokens']:,}개")
    print()

    print("💰 예상 비용 (USD):")
    print(f"  - 입력 토큰 비용: ${cost_info['input_cost']:.4f}")
    print(f"  - 출력 토큰 비용: ${cost_info['output_cost']:.4f}")
    print(f"  - 총 비용: ${cost_info['total_cost']:.4f}")
    print()

    # 비용 수준 평가
    if cost_info['total_cost'] < 0.01:
        level = "🟢 매우 저렴"
        advice = "테스트나 학습용으로 적합합니다."
    elif cost_info['total_cost'] < 0.05:
        level = "🟡 저렴"
        advice = "일반적인 실험용으로 적합합니다."
    elif cost_info['total_cost'] < 0.25:
        level = "🟠 보통"
        advice = "연구용으로 사용 가능하지만 비용을 고려하세요."
    else:
        level = "🔴 비싸다"
        advice = "대규모 연구나 상용 목적에만 권장합니다."

    print(f"📈 비용 수준: {level}")
    print(f"💡 권장사항: {advice}")
    print()

def compare_models(width: int, height: int, empty_ratio: float,
                  max_steps: int, agent_type: str = "simple"):
    """여러 모델의 비용을 비교합니다."""
    print("=" * 80)
    print("🔄 모델별 비용 비교")
    print("=" * 80)
    print()

    results = []
    for model in API_PRICING.keys():
        cost_info = calculate_simulation_cost(
            width, height, empty_ratio, max_steps,
            agent_type=agent_type, model=model
        )
        results.append((model, cost_info['total_cost']))

    # 비용 순으로 정렬
    results.sort(key=lambda x: x[1])

    print(f"{'모델':<15} {'비용 (USD)':<12} {'상대 비용':<10}")
    print("-" * 40)

    min_cost = results[0][1]
    for model, cost in results:
        relative = cost / min_cost if min_cost > 0 else 1
        print(f"{model:<15} ${cost:<11.4f} {relative:.1f}x")

    print()
    print(f"💡 가장 저렴한 모델: {results[0][0]} (${results[0][1]:.4f})")
    print(f"💡 가장 비싼 모델: {results[-1][0]} (${results[-1][1]:.4f})")
    print(f"💡 비용 차이: {results[-1][1] / results[0][1]:.1f}배")
    print()

def print_pricing_disclaimer():
    """가격 정보 면책 조항을 출력합니다."""
    print("\n" + "=" * 60)
    print("⚠️  중요 공지")
    print("=" * 60)
    print("• 가격 정보: 2024년 12월 기준으로 웹 검색을 통해 확인")
    print("• 실제 가격은 OpenAI의 정책에 따라 수시로 변동될 수 있습니다")
    print("• 정확한 최신 가격은 https://openai.com/pricing 에서 확인하세요")
    print("• 이 계산기는 참고용이며 실제 청구 금액과 다를 수 있습니다")
    print("=" * 60)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="LLM 쉘링 시뮬레이션 비용 계산기 (2024.12 기준)"
    )
    parser.add_argument("--width", type=int, default=15, help="격자 너비")
    parser.add_argument("--height", type=int, default=15, help="격자 높이")
    parser.add_argument("--empty-ratio", type=float, default=0.3, help="빈 공간 비율")
    parser.add_argument("--max-steps", type=int, default=5, help="최대 단계 수")
    parser.add_argument("--agent-type", choices=["simple", "group"],
                       default="simple", help="에이전트 유형")
    parser.add_argument("--model", choices=list(API_PRICING.keys()),
                                               default="gpt-4.1-nano-2025-04-14", help="OpenAI 모델")
    parser.add_argument("--compare", action="store_true", help="모델별 비용 비교")
    parser.add_argument("--disclaimer", action="store_true", help="가격 정보 면책 조항 표시")

    args = parser.parse_args()

    if args.disclaimer:
        print_pricing_disclaimer()
        return

    if args.compare:
        compare_models(args.width, args.height, args.empty_ratio,
                      args.max_steps, args.agent_type)
    else:
        cost_info = calculate_simulation_cost(
            args.width, args.height, args.empty_ratio, args.max_steps,
            agent_type=args.agent_type, model=args.model
        )
        print_cost_analysis(cost_info)

    # 간단한 면책 조항 표시
    print("\n💡 최신 가격 정보는 https://openai.com/pricing 에서 확인하세요")
    print("📅 이 계산기의 가격 정보: 2025년 6월 기준")
    print("🔄 주요 업데이트: o3 모델 80% 인하, 새로운 모델들 추가")

if __name__ == "__main__":
    main()