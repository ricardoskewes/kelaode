"""
Enhanced benchmark problems for language efficiency testing.
"""

from enhanced_benchmarks.MATH.math_problems import MATH_PROBLEMS
from enhanced_benchmarks.BBH.bbh_problems import BBH_PROBLEMS
from enhanced_benchmarks.HotpotQA.hotpotqa_problems import HOTPOTQA_PROBLEMS
from enhanced_benchmarks.ARC.arc_problems import ARC_PROBLEMS
from enhanced_benchmarks.GSM8K.gsm8k_problems import GSM8K_PROBLEMS

# Combine all enhanced benchmark problems
ENHANCED_BENCHMARK_PROBLEMS = (
    MATH_PROBLEMS +
    BBH_PROBLEMS +
    HOTPOTQA_PROBLEMS +
    ARC_PROBLEMS +
    GSM8K_PROBLEMS
)
