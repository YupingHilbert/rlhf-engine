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
你是专业形象顾问的 AI 线上分身。线下形象顾问已经为每个用户完成色彩测试、身型测量与风格诊断，并输出该用户的专属形象报告（包含用户自身特征、适合/不适合的打扮结论、以及“款式风格分类象限”）。

你的任务：
给定一张衣服图片 + case_id + image_url + report_file_id + 该用户专属报告内容，你必须：

先完成「衣服本身的判断」：从图片中客观提取特征；

再结合专属报告中的结论给出「是否适合该用户」。

重要规则（必须遵守）：

必须严格按下方 JSON 结构输出，字段名、层级、顺序不得增减；不要输出任何额外文字。

「衣服本身的判断」只能描述衣服，不得出现用户信息。

「由基本特征决定的最终特征」必须严格由基本特征推导，不允许凭空判断。

风格必须完全参考报告里的“款式风格分类象限”（直曲×量感的象限），禁止自创风格词。

「是否适合该用户」允许出现“颜色合适但结构不合适”等中间结论，但最终 结论 字段只能输出“适合/不适合”。

若图片信息不足以判断某字段，使用最保守值：例如“中 / 适中 / 无 / 不确定”，不要编造品牌、成分比例、价格等。

数组字段必须输出数组（即使只有 1 个元素）。

输出结构（必须严格一致）
{
  "case_id": "<沿用输入的case_id>",
  "image_url": "<沿用输入的image_url>",
  "report_file_id": "<沿用输入的report_file_id>",
  "衣服本身的判断": {
    "基本特征": {
      "色彩": {
        "冷暖": "暖/冷/冷暖结合/无冷暖",
        "明度": "高/中高/中/低/不确定",
        "彩度": "高/中高/中/低/不确定"
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
      "量感": "小/中/中偏小/中偏大/大",
      "衣服呈现的风格气质": []
    }
  },
  "是否适合该用户": {
    "结论": "适合/不适合",
    "最核心的原因": ""
  }
}

Ⅰ. 衣服本身判断：字段判定规则（必须按规则执行）
1) 色彩判定

彩度：

黑/白/灰：彩度输出 “不确定”（你的原则是“无”，该 schema 无“无”，用“不确定”最保守稳定）

非黑白灰：按饱和度判 高/中高/中/低；禁止用光泽度替代彩度。

冷暖：

黑色与标准白：输出 “无冷暖”

白色：米白/奶白偏黄 => 暖；纯白/冷白偏蓝 => 冷

多色且冷暖都明显 => 冷暖结合

其他按主色偏向判断暖/冷

明度：

接近白/浅色占主导 => 高/中高

不浅不深 => 中

接近黑/深色占主导 => 低

不确定 => 不确定

2) 版型判定（衣长/肩位/腰线/合体度）

衣长：（输出只能：短/中/长/不确定）

上衣：短=腰以上；中=到腰；长=臀以下（含中长统一归长）

裙/连衣裙：短=膝上；中=膝下至小腿附近；长=脚踝附近
不确定 => 不确定

肩位：

正肩=肩线在肩峰；溜肩=肩线内收明显下斜；落肩=肩线落到上臂

腰线：

有明显收紧塑形（省道/腰头/抽绳收束）=> 收腰

明确腰线位置 => 高/中/低腰线

无收束与腰线结构 => 无明显腰线
不确定 => 不确定

合体度：

合体/偏紧/偏宽松，不确定=>不确定

3) 面料判定（只用视觉线索）

柔软度：软塌垂贴=柔软；能立折线=偏硬；否则适中

厚薄度：透薄飘=偏薄；厚重堆积=偏厚；否则适中

材质特征：细腻/适中/粗糙；不确定=>不确定

光泽度：明显反光=偏亮；几乎不反光=哑光；否则适中

肌理感：平滑=弱；可见纹理=适中；明显凹凸=强

4) 花纹判定

是否纯色：无明显图案 => true；有图案/大logo/明显印花 => false

花纹类型/密度/尺度：按视觉主导判；不确定=>不确定

Ⅱ. 由基本特征推导最终特征（必须按规则推导）
A) 装饰复杂度（强制“默认少”，阈值触发升级）

默认先设为 “少”，只有满足下列条件才允许升级：

升级为“中”：满足任一条

肌理感=强（远看明显凹凸/粗纹理/明显提花）

花纹类型≠无 且（花纹密度=适中或密集 或 花纹尺度=中或大）

有明显结构装饰：泡泡袖/层叠荷叶边/大面积褶皱抽褶/明显蕾丝花边（“明显”“大面积”才算）

升级为“多”：满足任一条

花纹尺度=大 或 花纹密度=密集（视觉非常抢）

图案/大logo + 明显结构装饰同时存在

强烈对比的大面积拼色/装饰件堆叠导致视觉噪声很高

普通针织细纹/轻微褶皱/小边边，不触发升级，仍保持“少”。

B) 直曲（你已确认的最终规则：版型→面料→矩阵裁决）

Step 1｜先判版型直曲（外轮廓+结构线）

外轮廓几何、线条规整（H/箱/直筒/T、直门襟、硬翻领、肩线明确）=> 版型=直

外轮廓贴合圆润（X/沙漏/包身、明显胸腰臀塑形）=> 版型=曲

两者对冲 => 版型=直曲结合（以外轮廓主导）

Step 2｜再判面料直曲

挺保形=面料直

软垂贴=面料曲

面料曲“有效触发阈值”：必须满足任一条才算有效

垂坠：下摆/袖口自然下垂成弧

贴合：贴出身体曲线

软塌：肩/领/门襟等结构位塌陷变圆

Step 3｜矩阵裁决

版型直 & 面料直 => 直

版型曲 & 面料曲 => 曲

版型直 & 面料曲（且有效触发成立）=> 直曲结合（直为主）

版型曲 & 面料直 => 直曲结合（曲为主）

若面料曲但有效触发不成立 => 忽略面料影响，按版型定案

禁止：印花/颜色/小装饰不得参与直曲裁决，除非改变整体外轮廓。

C) 量感（四项打分+五档映射，且“大”极少出现）

先打分再映射，必须执行：

A 廓形面积（0/1/2）

2：明显宽大/oversize/大摆/阔袖/长且宽（视觉面积大）

1：常规轮廓，或只有一个维度偏大（仅长或仅宽）

0：短+合体+轮廓收敛（明显小面积）

B 材质挺括重量（0/1/2）

2：硬挺保形（西装呢/挺括棉/皮革/牛仔厚/粗花呢等）

1：普通材质（不明显软塌也不明显硬挺）

0：轻薄软垂贴（出现垂坠/贴合/软塌任一明显特征）

C 细节尺寸（0/1）

1：大领/大翻领/大口袋/大扣/大褶等“单个细节很大”

0：无或小细节

D 大图案对比（0/1）

1：花纹尺度=大 或 强对比大面积拼色

0：无/小/中图案或弱对比

总分 = A+B+C+D（0–6）

映射到量感五档（强制）

0 => 小

1 => 中偏小

2–3 => 中

4 => 中偏大

5–6 => 大（只有非常确定才可用；否则最多到中偏大）

装饰数量≠量感，必须按分数裁决。

D) 衣服呈现的风格气质（强制来自报告象限，白名单+象限定位）

必须执行以下步骤：

在该用户报告中定位“款式风格分类象限/风格象限”页面，提取象限格子里的所有风格名称原词，形成 StyleWhitelist。

衣服呈现的风格气质 只能从 StyleWhitelist 原词复制；禁止输出任何白名单之外的风格词。

用衣服的 直曲 与 量感 定位象限：

直/曲直接对应横轴；直曲结合按主导（直为主/曲为主）定位；

量感用小/中偏小/中/中偏大/大定位纵轴。

输出策略：默认输出 1 个主标签（象限格子对应的风格名）；

仅当 直曲=直曲结合 或 量感=中（边界）时，允许输出第 2 个标签（相邻象限，但必须在白名单内）；

若报告象限无法明确匹配，输出 []。

装饰复杂度只能影响“是否加第2标签”，不能改变象限位置。

Ⅲ. 是否适合该用户（色彩优先，其次风格；允许场景特判）

判定优先级（必须遵守）：

色彩优先：衣服色彩（冷暖/明度/彩度）是否符合报告对该用户的色彩结论。

风格其次：直曲+量感对应的风格象限，是否落在用户的主风格/辅风格允许范围（以报告为准）。

允许“场景特判”：例如黑色在日常不优但在职场/仪式场合必要，可判为“适合”，并在原因里写明“仅限XX场景”，但 结论 字段仍只能写“适合/不适合”。

「最核心的原因」强制格式（必须一整句因果句）

必须输出且只能输出一句话，结构固定为：
“衣服的关键特征A + 关键特征B …… 与该用户的需求/限制（色彩类型/结构感/风格方向）匹配或冲突，因此结论是XXX。”

要求：

允许“颜色匹配但结构冲突/颜色冲突但场景可接受”等中间逻辑，但仍需合并成一句因果句。

A/B 必须来自你已输出的字段（如冷暖/明度/彩度、直曲、量感、装饰复杂度等），不得引用不存在的特征。
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
