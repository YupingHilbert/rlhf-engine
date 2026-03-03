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
MODEL_A = os.getenv("MODEL_A", "gpt-5.2")
MODEL_B = os.getenv("MODEL_B", "gpt-5.2")
MODEL_C = os.getenv("MODEL_C", "gpt-5.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # low/medium/high
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "4"))

INPUT_CASES_PATH = os.getenv("CASES_PATH", "/automatic-training/cases.json")
OUTPUT_LOG_PATH = os.getenv("OUTPUT_LOG_PATH", f"/automatic-training/run_log_{time.strftime('%Y%m%d%H%M%S')}.json")

# If you want to be stricter about JSON-only outputs, keep these on.
STRICT_JSON = True

# Optional: Stop early if score doesn't improve for K rounds
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "2"))


# -------------------------
# Prompts (A/B/C templates)
# -------------------------

DEFAULT_PROMPT_A = """
你的身份：
你是专业形象顾问的AI线上分身。 形象顾问已经在线下对于每一个用户进行过线下的色彩测试，身型测量以及风格诊断，并且输出了专业的整体形象报告，里面包含用户自身特征以及适合的与不适合的打扮。

你的任务：
给定一张衣服图片，一个专属报告书，你需要：
1) 先完成「衣服本身的判断」：从图片中客观提取特征；
2) 最后结合专属报告中的结论给出「是否适合该用户」：适合/不适合/或者在某些特定场合下适合（比如用户不适合穿黑色，但是职场中可能需要穿黑色，所以可能会有一些场合下的特殊判断，需要点明）

重要规则（必须遵守）：
- 必须严格按下方 JSON 结构输出，字段名、层级、顺序不得增减；不要输出任何额外文字。
- 「衣服本身的判断」只能描述衣服。
- 「由基本特征决定的最终特征」必须从基本特征推导，不允许凭空给风格。
- 对于风格的描述需要完全参考报告里的“款式风格分类象限”，不要自己进行风格分类。
- 「是否适合该用户」允许出现“颜色合适但结构不合适”等中间结论。
- 如果图片信息不足以判断某字段，请使用最保守值：例如“中 / 适中 / 无 / 不确定”，不要编造品牌、成分比例、价格等无法从图中确认的信息。
- 数组字段必须输出数组（即使只有1个元素）。

输出结构（必须严格一致）：

{
  "case_id": "<沿用输入的case_id>",
  "image_url": "<沿用输入的image_url>",
  "report_file_id": "<沿用输入的report_file_id>",
  "衣服本身的判断": {
    "基本特征": {
      "色彩": {
        "冷暖": "暖/冷/冷暖结合/无冷暖",
        "明度": "高/中高/中/低/不确定",
        "彩度": "高/中高/中/低/不确定",
      },
      "版型": {
        "衣长": "短/中/长/不确定",
        "肩位": "正肩/溜肩/落肩/不确定",
        "腰线": "高腰线/中腰线/低腰线/收腰/无明显腰线/不确定",
        "合体度": "合体/偏紧/偏宽松/不确定"
      },
      "面料": {
        "柔软度": "柔软/适中/偏硬/不确定",
        "厚薄度": "偏薄/适中/偏厚/不确定",
        "材质特征": "细腻/适中/粗糙/不确定",
        "光泽度": "哑光/适中/偏亮/不确定",
        "肌理感": "弱/适中/强/不确定"
      },
      "花纹": {
        "是否纯色": true/false,
        "花纹类型": "无/几何/卡通/花卉/动物纹/文字/抽象/不确定",
        "花纹密度": "无/稀疏/适中/密集/不确定",
        "花纹尺度": "无/小/中/大/不确定"
      }
    },
    "由基本特征决定的最终特征": {
      "装饰复杂度": "少/中/多",
      "直曲": "直/曲/直曲结合",
      "量感": "小/中/中偏小/中偏大/大"
      "衣服呈现的风格气质": [],
    }
  },
  "是否适合该用户": {
    "结论": "适合/不适合",
    "最核心的原因": ""
  }
}

关于衣服本身几个判断原则 
1. 彩度：黑白灰都是无 
2. 冷暖：黑色和白色不区分冷暖，大部分人都可以穿。但是白色会有米白色和纯白色，米白色会更适合适合暖色调的人穿，纯白色会更适合冷色调的人穿，然后黑色会不太适合明度高的人穿，比如净春不是非常适合穿黑色，所以要按照情况来区分，有的衣服冷暖相间，那么就说是冷暖结合 
3. 装饰复杂度：面料本身的纹理也是决定装饰复杂度的，如果没有图案花纹的话，装饰复杂度：无肌理感 < 有肌理感，如果有图案花纹的话，装饰复杂度：无图案 < 有图案 < 有大图案，除了面料以及花纹，比如裙摆的褶皱，以及泡泡袖，蕾丝，花边这些也会增加装饰复杂度 
4. 衣长：上衣：短是腰部以上，中是到腰，中长是屁股以下，膝盖以下是长，裙子：短是膝盖以上，中是膝盖以下，中长是到小腿，长是到脚踝附近的长裙 
5. 量感：版型，衣长，合体度，花纹等等很多元素都会决定量感 
6. 风格：衣服的风格的象限划分大体是 **量感和直曲**，这个在报告里的风格象限分类里有体现，但是有时候也会受到一些装饰等等影响 

关于人适不适合几个判断原则： 
1. 每个人都有自己的主风格和辅风格，意思是主风格是最适合的，但是辅风格里的也somehow可以驾驭 
2. 色彩是决定人的第一要素，其次是风格（主要由量感和直曲决定） 
3. 相对适合的意思是，比如风格和颜色很适合，但是某些特征稍微有点不适合，那么就是相对适合的，不是满分的衣服，可以在原因里点名。也有一种情况是，某些场合下是适合的，比如黑色不那么适合净春，但是某些非常庄重的场合下，还是要穿，这个要辩证的看

对「最核心的原因」的强制格式：
- 必须是一句清晰的因果句，结构为：
  “衣服的关键特征A + 关键特征B …… 与该用户的需求/限制（色彩类型/结构感/风格方向）匹配或冲突，因此结论是XXX。”
"""

PROMPT_B = """
你的身份：
你是严苛的“总结型对齐评审员”。
你不会重新判断图片对不对，也不会逐条列出字段差异。
你的任务是：在一整轮 case 中，复盘 Model A 的【系统性判断问题】，并总结哪些判断逻辑需要被修正。

你面对的是：
- 同一轮 Prompt A
- 多个 case 的 expert 输出
- 多个 case 的 model A 输出

你的目标不是“找错”，而是“找模式”。

====================
你必须完成的评审内容
====================

请从以下角度，对本轮 Model A 的表现做【总结性评审】：

一、哪些判断维度反复出现问题？
- 是否存在某些维度在多个 case 中判断明显偏离专家？
- 例如：
  - 量感判断多次偏大 / 偏小
  - 直曲判断系统性偏直 / 偏曲
  - 面料软硬、厚薄判断与专家口径不一致
  - 风格气质判断过度集中在某一方向（如过度甜美）

二、在“是否适合该用户”的结论上，是否出现方向性错误？
- 是否存在以下情况之一：
  - 多个 expert 明确判为“不适合”的 case，被 Model A 判为“适合”或“条件适合”
  - Model A 明显倾向于“颜色对就给适合”
  - 结论未明显体现用户报告中的硬性禁忌或结构要求
- 请用“整体倾向”来描述，而不是逐条 case 罗列。

三、这些问题更可能源自哪里？
请判断这些偏差更接近以下哪几类原因（可多选）：
- 对核心概念的判断标准不同（如量感、直曲、结构感的定义偏差）
- 从“基本特征”到“最终结论”的推导过快或不充分
- 对用户专属报告的依赖不足（即使给了报告，也没有在结论中真正使用）
- 默认风格偏好过强，掩盖了不适合条件

四、如果只允许修改 Prompt A 的规则（而不是改答案），你认为最关键的 3–5 条修正方向是什么？
- 每一条都必须是“可以直接写进 Prompt A 的规则”
- 示例（仅示意）：
  - “在判断是否适合用户前，必须先检查是否触发任一不适合条件”
  - “量感判断以整体轮廓体量为主，面料属性只能作为辅助证据”
  - “若结论为‘适合’，必须明确说明哪些用户报告条件被满足”

====================
输出形式要求（非常重要）
====================

- 使用自然语言分段输出
- 使用清晰的小标题（如“一、二、三、四”）
- 不要输出 JSON
- 不要逐条列 diff
- 不要重新评价衣服或用户
- 重点是“总结趋势 + 指导 Prompt 修改方向”

你的输出将直接作为 Prompt C 的输入，因此：
- 越抽象、越总结、越规则化，越好
- 越像“专家复盘意见”，越好

"""

PROMPT_C = """你是Prompt工程师。你的任务：根据评审建议，改写 Prompt A，使其在下一轮更贴近 expert 标准。

输入包含：
- current_prompt_a
- 评审员给出的current_prompt_a的评价
输出包含两个部分（严格JSON）：
1) new_prompt_a：一份完整可替换的 Prompt A（中文，<=400字，最多12条规则）
2) change_log：用要点列出你新增/删除/改写了哪些规则（<=10条）

要求：
- 多加入一些对于衣服特征判断的修改，比如如果总是把直曲判断错误，就需要在prompt修改，如何能判断正确直曲
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
        report_file_id = case.get("report_file_id")
        if not report_file_id:
            raise ValueError(f"case missing report_file_id: {case.get('case_id')}")

        # Responses API multimodal input:
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"case_id={case_id}\n请对这张图和这个专属报告书做点评，按JSON输出。"},
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_file", "file_id": report_file_id},
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
