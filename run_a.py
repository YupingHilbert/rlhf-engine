#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Config
# -------------------------
load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_A = os.getenv("MODEL_A", "gpt-5.2")

INPUT_CASES_PATH = os.getenv("CASES_PATH", "/automatic-training/cases.json")
OUTPUT_PATH = os.getenv(
    "A_OUTPUT_PATH",
    f"/automatic-training/a_outputs_{time.strftime('%Y%m%d%H%M%S')}.json"
)

STRICT_JSON = os.getenv("STRICT_JSON", "1") == "1"


# -------------------------
# Prompt A (use your latest)
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


# -------------------------
# Helpers
# -------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("cases.json must be a non-empty list")
    return data


def safe_json_loads(s: str) -> Optional[Any]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(s[l:r+1])
            except Exception:
                return None
        return None


def save_json_crash_safe(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -------------------------
# Main
# -------------------------
def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    print(f"Loading cases: {INPUT_CASES_PATH}")
    cases = load_cases(INPUT_CASES_PATH)

    # minimal validation
    for i, c in enumerate(cases):
        if "case_id" not in c:
            c["case_id"] = f"{i+1:03d}"
        if not c.get("image_url"):
            raise ValueError(f"case {c['case_id']} missing image_url")
        if not c.get("report_file_id"):
            raise ValueError(f"case {c['case_id']} missing report_file_id")

    client = OpenAI(api_key=OPENAI_API_KEY)

    run_log: Dict[str, Any] = {
        "meta": {
            "created_at": now_iso(),
            "model_a": MODEL_A,
            "cases_path": INPUT_CASES_PATH,
            "prompt_a": DEFAULT_PROMPT_A,
            "strict_json": STRICT_JSON,
        },
        "results": [],
    }

    # write initial file immediately
    save_json_crash_safe(OUTPUT_PATH, run_log)
    print(f"Writing outputs to: {OUTPUT_PATH}")

    for idx, case in enumerate(cases, start=1):
        cid = case["case_id"]
        image_url = case["image_url"]
        report_file_id = case["report_file_id"]

        print(f"[{idx}/{len(cases)}] Running A for case_id={cid}")

        input_payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"case_id={cid}\n"
                            f"image_url={image_url}\n"
                            f"report_file_id={report_file_id}\n"
                            f"请严格按JSON结构输出。"
                        ),
                    },
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_file", "file_id": report_file_id},
                ],
            }
        ]

        t0 = time.time()
        resp = client.responses.create(
            model=MODEL_A,
            instructions=DEFAULT_PROMPT_A,
            input=input_payload,
        )
        dt = round(time.time() - t0, 3)

        output_text = getattr(resp, "output_text", "") or ""
        parsed = safe_json_loads(output_text) if STRICT_JSON else None

        item = {
            "case_id": cid,
            "image_url": image_url,
            "report_file_id": report_file_id,
            "response_id": getattr(resp, "id", None),
            "elapsed_sec": dt,
            "output_text": output_text,
            "parsed_json": parsed,
            "parsed_ok": parsed is not None,
            "written_at": now_iso(),
        }

        run_log["results"].append(item)

        # ✅ 关键：每个 case 完成后立刻落盘
        save_json_crash_safe(OUTPUT_PATH, run_log)

        print(f"  done. parsed_ok={item['parsed_ok']} elapsed={dt}s")

    print("✅ All cases done.")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
