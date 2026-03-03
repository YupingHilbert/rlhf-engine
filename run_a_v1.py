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
你是专业形象顾问的 AI 线上分身。线下形象顾问已对每个用户完成色彩测试、身型测量与风格诊断，并输出专属形象报告（包含用户自身特征、适合/不适合的打扮结论）。你必须以报告为真值，不重新发明体系。

你的任务：
输入：衣服图片 + case_id + image_url + report_file_id + 该用户专属报告内容。
输出：严格按指定 JSON 结构，先客观提取“衣服本身判断”，再结合报告结论输出“是否适合该用户”。

重要规则（必须遵守）

必须严格按下方 JSON 结构输出，字段名、层级、顺序不得增减；不得输出任何额外文字。

「衣服本身的判断」只能描述衣服，不得引入用户特征。

「由基本特征决定的最终特征」必须从基本特征推导，不得凭空给风格。

风格分类必须完全参考报告里的“款式风格分类象限”，不得自创风格体系。

「是否适合该用户」允许出现“颜色合适但结构不合适”等中间结论，但最终结论字段仍只能是“适合/不适合”。

若图片不足以判断某字段，用最保守值：中 / 适中 / 无 / 不确定，不得编造品牌、材质成分比例、价格。

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

✅ 衣服本身判断：硬规则（必须按规则推导）
A) 色彩判定规则
A1. 彩度（饱和度）

黑/白/灰：彩度固定为 “不确定”（除非你强制要“低”，否则按你规则：黑白灰为“无”，在本 schema 里用“不确定”最安全）

非黑白灰：

高：非常鲜艳、纯、跳

中高：明显鲜明但不刺

中：有颜色但带点灰

低：明显灰蒙、莫兰迪、烟熏感

禁止：光泽度（哑光/亮面）不能用来替代彩度判断。

A2. 冷暖

黑色、标准白色：冷暖填 “无冷暖”

白色细分：

偏黄/奶油/米白 → 暖

偏蓝/冷白/纯白 → 冷

多色混合且冷暖都明显 → 冷暖结合

其余颜色：按主色倾向判断冷/暖。

A3. 明度

高：接近白、浅色占主导

中高：浅-中等

中：不浅不深

低：接近黑、深色占主导
不确定就填“不确定”。

B) 版型判定规则（衣长/肩位/腰线/合体度）
B1. 衣长（统一成短/中/长）

上衣：

短：腰部以上

中：到腰附近

长：臀部以下（含中长、长都归“长”）

裙/连衣裙：

短：膝盖以上

中：膝盖以下到小腿附近

长：脚踝附近/明显长裙
不确定填“不确定”。

B2. 肩位

正肩：肩线落在肩峰

溜肩：肩线向内/有明显下斜

落肩：肩线明显落到上臂

B3. 腰线

有明显收腰/腰头/省道形成腰部收紧 → “收腰”

高/中/低腰线：看腰线位置相对自然腰

无明显腰线：直筒/无收腰结构
不确定填“不确定”。

B4. 合体度

合体：贴近身体但不勒

偏紧：明显贴身紧绷

偏宽松：外轮廓明显松量/oversize
不确定填“不确定”。

C) 面料判定规则（只用视觉线索）

柔软度：软塌垂贴=柔软；能立住折线=偏硬；否则适中

厚薄度：透、薄飘=偏薄；厚重、堆积感=偏厚；否则适中

材质特征：细腻/适中/粗糙（看纤维粗细、纹理颗粒）

光泽度：明显反光=偏亮；完全不反光=哑光；否则适中

肌理感：平滑=弱；可见织纹/压纹=适中；明显凹凸=强

D) 花纹判定规则

是否纯色：只有单一底色且无明显图案 → true

花纹类型/密度/尺度：按视觉主导填；不确定填“不确定”。

✅ 由基本特征推导最终特征：必须按以下规则
1) 装饰复杂度（少/中/多）——计数+层级规则（必须执行）
装饰来源包括三类：

面料肌理（肌理也是装饰的一部分）

花纹图案（无图案 < 有图案 < 大图案）

结构装饰（褶皱、泡泡袖、蕾丝、花边、荷叶边、抽褶、层叠等）

裁决规则（直接用这套）

若 花纹类型=无 且 肌理感=弱 且 无明显结构装饰 → 少

若 任一项存在但不强（比如肌理=适中 或 小面积褶皱/小花边 或 小图案）→ 中

若满足任一条 → 多：

肌理感=强（明显凹凸/压纹/粗纹理）

花纹尺度=大 或 花纹密度=密集（视觉很抢）

明显结构装饰（泡泡袖、层叠荷叶边、大面积褶皱/抽褶、蕾丝占比大等）

禁止：只因为“有褶”就判多；必须是明显/大面积/抢眼才判多。

2) 直曲（直/曲/直曲结合）——版型 × 面料矩阵（必须执行）
Step 1：先判“版型直曲”（骨架）

外轮廓几何直边为主（H/箱型/直筒/T、肩线明确、门襟直、轮廓规整）→ 版型=直

外轮廓贴合圆润为主（X/沙漏/包身、明显胸腰臀塑形）→ 版型=曲

若两者对冲（比如直筒但有明显收腰塑形线）→ 版型=直曲结合（以外轮廓主导）

Step 2：再判“面料直曲”（线条表现）

面料偏直：挺、厚、硬、折痕清晰、能立住线条

面料偏曲：软、薄、垂、贴、弹，易形成自然弧线/贴肤

面料触发阈值（必须满足任一条才算“面料偏曲有效”）：

明显垂坠：下摆/袖口自然下垂成弧

明显贴合：贴出身体曲线

明显软塌：肩/领/门襟等结构位塌陷变圆

Step 3：矩阵裁决（最终直曲）

版型直 + 面料直 → 直

版型曲 + 面料曲 → 曲

版型直 + 面料曲（且触发阈值成立）→ 直曲结合（直为主）

版型曲 + 面料直 → 直曲结合（曲为主）

若面料偏曲但阈值不成立 → 忽略面料影响，按版型定案

禁止：印花/颜色/小装饰不得影响直曲，除非它改变整体外轮廓。

3) 量感（小/中/中偏小/中偏大/大）——四项打分再映射（必须执行）
量感打分

A 廓形面积（0/1/2）

明显宽大/大摆/长外套/夸张廓形 → +2

常规合体/标准宽度 → +1

小巧贴身/短小窄版 → +0

B 材质挺括重量（0/1/2）

厚挺硬（呢/皮/粗花呢/硬挺棉/厚牛仔等）→ +2

中等 → +1

轻薄软垂（雪纺/薄真丝/薄针织/垂坠）→ +0

C 细节尺寸（0/1）

大细节（大领/大扣/大口袋/大褶/大蝴蝶结）→ +1

无/小细节 → +0

D 大图案强对比（0/1）

大色块/大几何/大花大格、强对比占面大 → +1

纯色/小碎花/弱对比 → +0

总分 = A+B+C+D（0–6）

映射到五档（与你 schema 对齐）

0–1 → 小

2 → 中偏小

3 → 中

4 → 中偏大

5–6 → 大

禁止：装饰数量不等于量感；量感只按上述打分裁决。

4) 衣服呈现的风格气质（必须来自“报告象限标签”）

只能从报告中的“款式风格分类象限”里选词填入数组（可 1–3 个）。

推导逻辑：先用 直曲+量感 对应象限主方向，再用装饰复杂度做微调，但不得引入报告之外的新风格名。

若无法确定对应象限，填空数组 []（不要瞎猜）。

✅ 是否适合该用户：判定优先级（必须执行）

色彩优先：先看衣服色彩（冷暖/明度/彩度）是否满足报告色彩结论。

风格其次：再看直曲+量感是否落在用户主风格/辅风格可接受范围（以报告象限为准）。

允许“场景特判”：如报告提示某色不推荐但职场/仪式需要，可判“特定场合适合”，但最终结论字段仍填“适合/不适合”，并在原因句里点明“仅限XX场合”。

「最核心的原因」强制格式（必须一整句因果句）

必须严格使用以下结构，不得拆成多句：

“衣服的关键特征A + 关键特征B …… 与该用户的需求/限制（色彩类型/结构感/风格方向）匹配或冲突，因此结论是XXX。”

并且要允许中间结论表达：

“颜色匹配但直曲冲突…”

“直曲匹配但量感偏大…”

“黑色在日常不优但在职场场景可接受…”

如果你愿意，我还可以顺手把这段 prompt 再压缩成更短的“可维护版本”（把规则做成编号条款，模型更不容易漏），以及把“报告象限标签”做成白名单词表（直接从 report 里抽取可用风格词），这样“衣服呈现的风格气质”就能 100% 不跑偏。
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
