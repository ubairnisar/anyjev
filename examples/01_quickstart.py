"""Is a document crop STRUCTURED or PURE PROSE?  (when in doubt: structured)

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-logprobs 20
    python examples/01_quickstart.py crop1.png crop2.png
"""
import sys

from anyjev import Judge, presets

judge = Judge("http://localhost:8000/v1")      # model is auto-detected
task = presets.structured_vs_prose(threshold=0.30)

for path in sys.argv[1:]:
    r = judge.classify(task, image=path)
    print(f"{path}: {r.label}  P(structured)={r.p_positive:.3f}  gap={r.order_gap:.3f}"
          + ("  <- check manually" if r.uncertain else ""))
