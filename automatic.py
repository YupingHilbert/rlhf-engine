#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABC Prompt Self-Improve Loop
- Read 10 cases (image_url + expert_json)
- LLM A generates critiques using prompt_a
- LLM B diffs A vs expert, outputs structured issues + prompt fixes
- Aggregate B results into round summary
- LLM C edits prompt_a -> new_prompt_a
- Iterate N rounds, pick best prompt by score
- Write EVERYTHING into run_log.json

OpenAI Responses API is recommended (OpenAI Python SDK).
Docs: https://platform.openai.com/docs/guides/text
"""

import os
import json
import time
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI


# -------------------------
# Config
# -------------------------
load_dotenv("./.env")
MODEL_A = os.getenv("MODEL_A", "gpt-4o")
MODEL_B = os.getenv("MODEL_B", "gpt-4o")
MODEL_C = os.getenv("MODEL_C", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # low/medium/high
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "2"))

INPUT_CASES_PATH = os.getenv("CASES_PATH", "/automatic-training/cases.json")
OUTPUT_LOG_PATH = os.getenv("OUTPUT_LOG_PATH", "/automatic-training/run_log.json")

# If you want to be stricter about JSON-only outputs, keep these on.
STRICT_JSON = True

# Optional: Stop early if score doesn't improve for K rounds
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "2"))


# -------------------------
# Prompts (A/B/C templates)
# -------------------------

DEFAULT_PROMPT_A = """你是专业形象顾问的AI线上分身。你必须只根据图片中可观察到的信息判断，不确定就明确写“不确定”。

对每张图片输出一个JSON，字段固定为：
color, material, silhouette, style(最多2个), do(2条), dont(2条), evidence(2条)

硬规则：
1) 每个结论必须对应至少1条 evidence（先写 evidence，再写结论也可以）。
2) 禁止编造品牌、成分比例、具体价格等图片无法确认的信息。
3) style 只能从这个列表选：{可爱少年型, 罗曼型, 经典型, 自然型, 前卫型, 优雅型, 运动休闲型, 甜酷, 极简}（最多2个）。
4) 输出必须是严格JSON，不要多余解释文字。
"""

PROMPT_B = """你是严苛的“对齐评审员”。你不会重新评价图片，你只做：对比 Model A 输出 与 Expert 输出的差异。

输入包含：case_id, expert_json, model_a_json
请输出严格JSON：
case_id
diff: 数组，每项包含 field, issue_type(wrong/missing/overclaim/unverifiable), severity(1-3), expert, model_a, why_wrong, prompt_fix
summary_prompt_fix: 汇总3-8条“应该如何修改 Prompt A 的规则”，必须可执行、可写进prompt。

注意：prompt_fix 必须是“改prompt的规则”，不是“改答案”。
"""

PROMPT_C = """你是Prompt工程师。你的任务：根据评审建议，改写 Prompt A，使其在下一轮更贴近 expert 标准。

输入包含：
- current_prompt_a
- all_cases_summary_prompt_fix（来自10个case的汇总）
输出包含两个部分（严格JSON）：
1) new_prompt_a：一份完整可替换的 Prompt A（中文，<=400字，最多12条规则）
2) change_log：用要点列出你新增/删除/改写了哪些规则（<=10条）

要求：
- 优先解决高频错误和severity=3的问题
- 保持结构化输出字段不变
- 不要加入与任务无关的长解释
"""


# -------------------------
# Utilities
# -------------------------

def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("cases.json must be a non-empty list")
    return data


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_json_loads(s: str) -> Any:
    """
    Best-effort JSON parsing:
    - If model returns extra text, try to extract first JSON object.
    """
    s_strip = s.strip()
    try:
        return json.loads(s_strip)
    except Exception:
        # Try to extract substring between first { and last }
        l = s_strip.find("{")
        r = s_strip.rfind("}")
        if l != -1 and r != -1 and r > l:
            candidate = s_strip[l:r+1]
            return json.loads(candidate)
        raise


def score_from_b_outputs(b_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute an objective score from B's diff severities.
    Higher is better.
    """
    total_issues = 0
    severity_sum = 0
    severe3 = 0
    by_issue_type = {}
    by_field = {}

    for bo in b_outputs:
        diffs = bo.get("diff", []) or []
        for d in diffs:
            total_issues += 1
            sev = int(d.get("severity", 1))
            severity_sum += sev
            if sev >= 3:
                severe3 += 1
            itype = d.get("issue_type", "unknown")
            field = d.get("field", "unknown")
            by_issue_type[itype] = by_issue_type.get(itype, 0) + 1
            by_field[field] = by_field.get(field, 0) + 1

    # Simple scoring heuristic:
    # Start from 1000, subtract penalties
    score = 1000
    score -= total_issues * 10
    score -= severity_sum * 15
    score -= severe3 * 30

    return {
        "score": score,
        "total_issues": total_issues,
        "severity_sum": severity_sum,
        "severe3": severe3,
        "by_issue_type": by_issue_type,
        "by_field": by_field,
    }


def aggregate_prompt_fixes(b_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate summary_prompt_fix across all cases:
    - Count frequency
    - Return top items + also pass raw list to C
    """
    freq = {}
    raw = []
    for bo in b_outputs:
        fixes = bo.get("summary_prompt_fix", []) or []
        for fx in fixes:
            fx_norm = " ".join(str(fx).strip().split())
            if not fx_norm:
                continue
            raw.append(fx_norm)
            freq[fx_norm] = freq.get(fx_norm, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    top = [{"fix": k, "count": v} for k, v in ranked[:12]]

    return {
        "raw_fixes": raw,
        "top_fixes": top,
        "freq_map": freq,
    }


# -------------------------
# OpenAI Calls (Responses API)
# -------------------------

@dataclass
class LLMResult:
    output_text: str
    parsed_json: Optional[Any]
    raw_response_id: Optional[str]


class ABCLoop:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _responses_text(self, model: str, instructions: str, input_payload: Any) -> LLMResult:
        """
        input_payload can be:
        - a string
        - or a list of messages / multimodal parts
        """
        resp = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=input_payload,
        )

        text = getattr(resp, "output_text", None) or ""
        parsed = None
        if STRICT_JSON:
            try:
                parsed = safe_json_loads(text)
            except Exception:
                parsed = None
        return LLMResult(
            output_text=text,
            parsed_json=parsed,
            raw_response_id=getattr(resp, "id", None),
        )

    def run_a_for_case(self, prompt_a: str, case: Dict[str, Any]) -> LLMResult:
        """
        Vision input (image_url) + short task directive.
        """
        image_url = case.get("image_url")
        if not image_url:
            raise ValueError(f"case missing image_url: {case.get('case_id')}")

        case_id = case.get("case_id", "")

        # Responses API multimodal input:
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"case_id={case_id}\n请对这张图做点评，按JSON输出。"},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ]
        return self._responses_text(model=MODEL_A, instructions=prompt_a, input_payload=input_payload)

    def run_b_for_case(self, expert_json: Any, model_a_json: Any, case_id: str) -> LLMResult:
        input_payload = json.dumps(
            {
                "case_id": case_id,
                "expert_json": expert_json,
                "model_a_json": model_a_json,
            },
            ensure_ascii=False,
        )
        return self._responses_text(model=MODEL_B, instructions=PROMPT_B, input_payload=input_payload)

    def run_c_edit_prompt(self, current_prompt_a: str, agg_fixes: Dict[str, Any]) -> LLMResult:
        input_payload = json.dumps(
            {
                "current_prompt_a": current_prompt_a,
                "all_cases_summary_prompt_fix": {
                    "top_fixes": agg_fixes.get("top_fixes", []),
                    "raw_fixes": agg_fixes.get("raw_fixes", []),
                },
            },
            ensure_ascii=False,
        )
        return self._responses_text(model=MODEL_C, instructions=PROMPT_C, input_payload=input_payload)


# -------------------------
# Main Loop
# -------------------------

def main():
    print(f"Loading cases from {INPUT_CASES_PATH}")
    cases = load_cases(INPUT_CASES_PATH)

    # Validate: each case must have case_id, image_url, expert
    for i, c in enumerate(cases):
        if "case_id" not in c:
            c["case_id"] = f"{i+1:03d}"
        if "expert" not in c:
            raise ValueError(f"case {c['case_id']} missing expert")

    runner = ABCLoop()

    run_log: Dict[str, Any] = {
        "meta": {
            "created_at": now_iso(),
            "cases_path": INPUT_CASES_PATH,
            "models": {"A": MODEL_A, "B": MODEL_B, "C": MODEL_C},
            "reasoning_effort": REASONING_EFFORT,
            "max_rounds": MAX_ROUNDS,
            "strict_json": STRICT_JSON,
        },
        "rounds": [],
        "best": None,
    }

    prompt_a = DEFAULT_PROMPT_A
    best_score = None
    best_prompt = None
    best_round_index = None
    no_improve = 0

    for r in range(1, MAX_ROUNDS + 1):
        round_entry: Dict[str, Any] = {
            "round": r,
            "started_at": now_iso(),
            "prompt_a_in": prompt_a,
            "cases": [],
            "a_outputs": [],
            "b_outputs": [],
            "aggregate": None,
            "c_output": None,
            "prompt_a_out": None,
            "score": None,
            "ended_at": None,
        }

        # --- A: generate for each case ---
        a_case_jsons: List[Dict[str, Any]] = []
        a_raw: List[Dict[str, Any]] = []

        for case in cases:
            cid = case["case_id"]
            a_res = runner.run_a_for_case(prompt_a, case)

            # If strict JSON parsing failed, keep raw text and continue;
            # B will still run using raw text as fallback if needed.
            a_json = a_res.parsed_json if a_res.parsed_json is not None else {"_raw_text": a_res.output_text}

            a_case_jsons.append(a_json)
            a_raw.append(
                {
                    "case_id": cid,
                    "response_id": a_res.raw_response_id,
                    "output_text": a_res.output_text,
                    "parsed_json": a_res.parsed_json,
                }
            )

        round_entry["a_outputs"] = a_raw

        # --- B: diff each case ---
        b_raw: List[Dict[str, Any]] = []
        b_parsed: List[Dict[str, Any]] = []

        for idx, case in enumerate(cases):
            cid = case["case_id"]
            expert = case["expert"]
            model_a_json = a_case_jsons[idx]

            b_res = runner.run_b_for_case(expert_json=expert, model_a_json=model_a_json, case_id=cid)
            b_json = b_res.parsed_json if b_res.parsed_json is not None else {"_raw_text": b_res.output_text}

            b_raw.append(
                {
                    "case_id": cid,
                    "response_id": b_res.raw_response_id,
                    "output_text": b_res.output_text,
                    "parsed_json": b_res.parsed_json,
                }
            )
            # For scoring, we need something shaped like B JSON
            if isinstance(b_json, dict):
                b_parsed.append(b_json)
            else:
                b_parsed.append({"case_id": cid, "diff": [], "summary_prompt_fix": [], "_non_dict": True})

        round_entry["b_outputs"] = b_raw

        # --- Aggregate / Score ---
        agg = aggregate_prompt_fixes(b_parsed)
        score_obj = score_from_b_outputs(b_parsed)

        round_entry["aggregate"] = {
            "prompt_fix_aggregate": agg,
            "score_breakdown": score_obj,
        }
        round_entry["score"] = score_obj

        # Track best
        score_val = score_obj["score"]
        improved = (best_score is None) or (score_val > best_score)

        if improved:
            best_score = score_val
            best_prompt = prompt_a
            best_round_index = r
            no_improve = 0
        else:
            no_improve += 1

        # --- C: edit prompt for next round (unless last round) ---
        if r < MAX_ROUNDS:
            c_res = runner.run_c_edit_prompt(current_prompt_a=prompt_a, agg_fixes=agg)
            c_json = c_res.parsed_json if c_res.parsed_json is not None else None

            round_entry["c_output"] = {
                "response_id": c_res.raw_response_id,
                "output_text": c_res.output_text,
                "parsed_json": c_res.parsed_json,
            }

            # Extract new prompt
            new_prompt = None
            if isinstance(c_json, dict):
                new_prompt = c_json.get("new_prompt_a")
            if not new_prompt:
                # fallback: keep current prompt if C failed
                new_prompt = prompt_a

            round_entry["prompt_a_out"] = new_prompt
            prompt_a = new_prompt

        round_entry["ended_at"] = now_iso()
        run_log["rounds"].append(round_entry)

        # Save incremental log each round (crash-safe)
        with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(run_log, f, ensure_ascii=False, indent=2)

        # Early stop
        if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
            break

    run_log["best"] = {
        "best_score": best_score,
        "best_prompt_a": best_prompt,
        "best_round": best_round_index,
        "finished_at": now_iso(),
    }

    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    print("✅ Done.")
    print(f"Best round: {best_round_index}, best_score: {best_score}")
    print(f"Log written to: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()
