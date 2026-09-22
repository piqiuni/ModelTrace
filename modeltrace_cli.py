#!/usr/bin/env python3
"""ModelTrace 终端在线评测：主动探测模型归因。

把 ModelTrace（xqy2006/ModelTrace）的三条长整数挑战 + 统一指纹库归因
收成一个零第三方依赖的单文件，可用 wget 管道直接跑：

    wget -qO- "https://raw.githubusercontent.com/<you>/<repo>/main/modeltrace_cli.py" \\
      | python3 - -m gpt-5.6-sol

或本地：

    python modeltrace_cli.py -m gpt-5.6-sol -u https://api.example.com/v1 -k sk-...

自动模式会向 OpenAI Chat Completions / Anthropic Messages 发挑战，
用统一指纹库做闭集归因。手动模式可粘贴/管道输入三份完整数字序列。

说明：
- 指纹库 unified_bank.json 首次运行从 GitHub raw 拉取并缓存（约 0.7MB）。
- 结果仅供参考，不是判断模型的决定性证据；未收录模型仍会归到最相近候选。
- 系统提示词会显著影响偏好，不建议在 Claude Code 等强系统提示环境里测。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量（与 ModelTrace fingerprint.py 对齐）
# ---------------------------------------------------------------------------

VALUE_MIN = 1
VALUE_MAX = 355
DIMENSION = VALUE_MAX - VALUE_MIN + 1  # 355
ALPHA = 0.5
ORDERED_BLOCK_WEIGHT = 0.25
ORDERED_FEATURE_DIM = 4 * 16 + 10  # 74
BANK_URL_DEFAULT = (
    "https://raw.githubusercontent.com/xqy2006/ModelTrace/main/data/unified_bank.json"
)
DEFAULT_UPSTREAM_USER_AGENT = (
    "Codex Desktop/0.147.0-alpha.1.2 (Windows 10.0.26200; x86_64) unknown "
    "(codex_exec; 0.147.0-alpha.1.2)"
)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# 纯 Python 数值工具
# ---------------------------------------------------------------------------


def standardize(values: list[float]) -> list[float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    scale = max(math.sqrt(variance), 1e-12)
    return [(value - mean) / scale for value in values]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = max(math.sqrt(sum(x * x for x in vector)), 1e-12)
    return [x / norm for x in vector]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """matrix: (k, d) @ vector: (d,) -> (k,)"""
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def project_out(vector: list[float], basis: list[list[float]]) -> list[float]:
    """v -= (v @ B.T) @ B；basis: (rank, d)。"""
    if not basis:
        return list(vector)
    coeffs = mat_vec(basis, vector)
    subtract = [0.0] * len(vector)
    for coeff, row in zip(coeffs, basis):
        for j, value in enumerate(row):
            subtract[j] += coeff * value
    return [v - s for v, s in zip(vector, subtract)]


def array_split(values: list, parts: int) -> list[list]:
    """等价于 numpy.array_split：前 (len % parts) 块多 1 个元素。"""
    length = len(values)
    base, extra = divmod(length, parts)
    chunks = []
    start = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        chunks.append(values[start : start + size])
        start += size
    return chunks


def histogram_16(values: list[float]) -> list[int]:
    """等价于 np.histogram(..., bins=16, range=(1.0, 356.0))。"""
    counts = [0] * 16
    lo, hi = 1.0, 356.0
    width = (hi - lo) / 16.0
    for value in values:
        if value < lo or value > hi:
            continue
        index = int((value - lo) / width)
        if index >= 16:
            index = 15
        if index < 0:
            index = 0
        counts[index] += 1
    return counts


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


# ---------------------------------------------------------------------------
# 指纹特征与打分（与 fingerprint.py 对齐）
# ---------------------------------------------------------------------------


def parse_numbers(text: str) -> list[int]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous_end = 0
    for match in re.finditer(r"\d+", text):
        separator = text[previous_end : match.start()]
        value = int(match.group())
        if current and any(character.isalpha() for character in separator):
            runs.append(current)
            current = []
        if VALUE_MIN <= value <= VALUE_MAX:
            current.append(value)
        previous_end = match.end()
    if current:
        runs.append(current)
    return max(runs, key=len) if runs else []


def count_numbers(numbers: list[int]) -> list[int]:
    counts = [0] * DIMENSION
    for number in numbers:
        counts[number - VALUE_MIN] += 1
    return counts


def hellinger_feature(counts: list[int]) -> list[float]:
    values = [c + ALPHA for c in counts]
    total = sum(values)
    return [math.sqrt(v / total) for v in values]


def ordered_block_feature(numbers: list[int]) -> list[float]:
    pieces: list[list[float]] = []
    values = [float(x) for x in numbers]
    for chunk in array_split(values, 4):
        counts = histogram_16(chunk)
        smoothed = [c + 0.5 for c in counts]
        total = sum(smoothed)
        pieces.append([math.sqrt(v / total) for v in smoothed])
    last_digits = [0] * 10
    for value in numbers:
        last_digits[int(value) % 10] += 1
    smoothed = [c + 0.5 for c in last_digits]
    total = sum(smoothed)
    pieces.append([math.sqrt(v / total) for v in smoothed])
    out: list[float] = []
    for piece in pieces:
        out.extend(piece)
    return out


def robust_score_counts(counts: list[int], bank: dict) -> dict[str, list[float]]:
    robust = bank["robust"]
    hellinger = robust["hellinger"]
    feature = hellinger_feature(counts)
    mean = hellinger["feature_mean"]
    scale = hellinger["feature_scale"]
    projected = [(f - m) / s for f, m, s in zip(feature, mean, scale)]
    projected = project_out(projected, hellinger.get("nuisance_basis") or [])
    projected = l2_normalize(projected)
    centroids = hellinger["centroids"]
    nuisance = standardize(mat_vec(centroids, projected))
    fused = standardize(list(nuisance))
    return {"fused": fused, "nuisance": nuisance}


def ordered_block_scores(numbers: list[int], bank: dict) -> list[float]:
    artifact = bank["robust"]["ordered_blocks"]
    feature = ordered_block_feature(numbers)
    mean = artifact["feature_mean"]
    scale = artifact["feature_scale"]
    standardized_feature = [(f - m) / s for f, m, s in zip(feature, mean, scale)]

    normalized = l2_normalize(standardized_feature)
    templates = artifact.get("environment_centroids") or []
    if templates:
        environment_scores = [mat_vec(centroids, normalized) for centroids in templates]
        n_models = len(environment_scores[0])
        template = [max(env[i] for env in environment_scores) for i in range(n_models)]
    else:
        template = mat_vec(artifact["centroids"], normalized)
    template = standardize(template)

    projected = project_out(standardized_feature, artifact.get("nuisance_basis") or [])
    projected = l2_normalize(projected)
    nuisance = standardize(mat_vec(artifact["centroids"], projected))

    return standardize([0.5 * t + 0.5 * n for t, n in zip(template, nuisance)])


def robust_score_numbers(numbers: list[int], bank: dict) -> dict[str, list[float]]:
    marginal = robust_score_counts(count_numbers(numbers), bank)
    artifact = bank["robust"].get("ordered_blocks")
    ordered_weight = float(artifact.get("weight", 0.0)) if artifact else 0.0
    if not artifact or ordered_weight == 0.0:
        return {
            **marginal,
            "marginal_fused": marginal["fused"],
            "ordered": marginal["fused"],
        }
    ordered = ordered_block_scores(numbers, bank)
    fused = [
        (1.0 - ordered_weight) * marginal_score + ordered_weight * ordered_score
        for marginal_score, ordered_score in zip(marginal["fused"], ordered)
    ]
    return {
        **marginal,
        "fused": fused,
        "marginal_fused": marginal["fused"],
        "ordered": ordered,
    }


def js_similarity(left: list[int], right: list[int]) -> float:
    left_total = sum(left)
    right_total = sum(right) + ALPHA * DIMENSION
    p = [value / left_total for value in left]
    q = [(value + ALPHA) / right_total for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]

    def divergence(values: list[float], middle: list[float]) -> float:
        return sum(value * math.log(value / target) for value, target in zip(values, middle) if value)

    js = (divergence(p, midpoint) + divergence(q, midpoint)) / 2.0
    return 1.0 - math.sqrt(js / math.log(2.0))


FAMILY_DISPLAY_NAMES = {"gpt": "GPT", "claude": "Claude"}


def analyze_outputs(outputs: list[dict], bank: dict) -> dict:
    model_ids = [model["id"] for model in bank["models"]]
    valid = []
    diagnostics = []
    for index, item in enumerate(outputs):
        text = str(item.get("text", ""))
        expected = int(item.get("expected_count") or 0)
        numbers = parse_numbers(text)
        minimum = max(80, math.ceil(expected * 0.55)) if expected else 80
        accepted = len(numbers) >= minimum
        diagnostics.append(
            {
                "index": index,
                "parsed_numbers": len(numbers),
                "minimum_numbers": minimum,
                "accepted": accepted,
            }
        )
        if accepted:
            counts = count_numbers(numbers)
            components = robust_score_numbers(numbers, bank)
            valid.append({"counts": counts, "scores": components["fused"], **components})

    if not valid:
        raise ValueError("没有可用回答：请粘贴完整数字序列；拒答或严重截断的回答不会计入。")

    combined_scores = [
        sum(item["scores"][index] for item in valid) / len(valid)
        for index in range(len(model_ids))
    ]
    combined_nuisance = [
        sum(item.get("nuisance", item["scores"])[index] for item in valid) / len(valid)
        for index in range(len(model_ids))
    ]
    calibration_key = str(min(len(valid), 3))
    beta = float(bank["calibration"][calibration_key]["beta"])
    probabilities = softmax([beta * value for value in combined_scores])
    pooled_counts = [
        sum(item["counts"][index] for item in valid) for index in range(DIMENSION)
    ]
    bank_models = {model["id"]: model for model in bank["models"]}
    results = [
        {
            "model": model_id,
            "display_name": bank_models[model_id]["display_name"],
            "probability": probabilities[index],
            "profile_similarity": js_similarity(pooled_counts, bank_models[model_id]["counts"]),
            "score": combined_scores[index],
            "nuisance_score": combined_nuisance[index],
        }
        for index, model_id in enumerate(model_ids)
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "prediction": results[0]["model"],
        "prediction_name": results[0]["display_name"],
        "probability": results[0]["probability"],
        "used_outputs": len(valid),
        "results": results,
        "diagnostics": diagnostics,
        "calibration": {
            "queries": calibration_key,
            "beta": beta,
            "cv_accuracy": bank["calibration"][calibration_key]["cv_accuracy"],
        },
        "method": bank.get("method", {}).get("name", "Ordered-block + nuisance-Hellinger"),
    }


def analyze_global_outputs(outputs: list[dict], bank: dict) -> dict:
    result = analyze_outputs(outputs, bank)
    model_entries = {model["id"]: model for model in bank["models"]}
    family_order = list(
        dict.fromkeys(model.get("family") or "models" for model in bank["models"])
    )
    family_names = {
        family_id: next(
            (
                model.get("family_name")
                for model in bank["models"]
                if (model.get("family") or "models") == family_id and model.get("family_name")
            ),
            FAMILY_DISPLAY_NAMES.get(family_id, family_id),
        )
        for family_id in family_order
    }
    family_probabilities = {
        family_id: sum(
            item["probability"]
            for item in result["results"]
            if (model_entries[item["model"]].get("family") or "models") == family_id
        )
        for family_id in family_order
    }
    for item in result["results"]:
        model = model_entries[item["model"]]
        family_id = model.get("family") or "models"
        item["family"] = family_id
        item["family_name"] = family_names[family_id]
        item["conditional_probability"] = (
            item["probability"] / family_probabilities[family_id]
            if family_probabilities[family_id]
            else 0.0
        )
    winning_family = max(family_order, key=family_probabilities.get)
    return {
        **result,
        "family_prediction": winning_family,
        "family_prediction_name": family_names[winning_family],
        "family_probability": family_probabilities[winning_family],
        "family_probabilities": [
            {
                "family": family_id,
                "display_name": family_names[family_id],
                "probability": family_probabilities[family_id],
            }
            for family_id in family_order
        ],
        "method": "统一全局稳健数字指纹",
    }


def generate_challenges(count: int = 3) -> list[dict]:
    rng = random.SystemRandom()
    lengths = rng.sample(range(292, 333), count)
    openings = [
        "这是一次独立的数值选择记录",
        "请完成下面的无语义整数选择任务",
        "执行一次第一反应取值记录",
        "生成一组不承载语义的整数选择",
        "进行一轮快速逐项取值",
    ]
    actions = [
        "为各个位置分别凭第一反应选择",
        "逐项选择",
        "每次只决定当前一项，共给出",
        "分别凭第一反应给出",
        "逐个直接选择",
    ]
    endings = [
        "允许某个数字再次出现；每项写出后不要回头排序、去重或替换。",
        "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
        "相同值可以再次出现；输出过程中不要整理或改写前面的项目。",
        "重复值无需删除；不要筛选、重排或补成某种规律。",
        "不必赋予数字任何含义；已经给出的值保持不变。",
    ]
    separator_hints = [
        "数字之间用逗号或空格分隔均可。",
        "使用一种一致的常见分隔符即可。",
        "可以用逗号、空格或换行分隔。",
        "只要每个整数边界清楚，格式可自行选择。",
    ]
    challenges = []
    for index, length in enumerate(lengths):
        prompt = (
            f"{rng.choice(openings)}。{rng.choice(actions)} {length} 个 1 到 355（含端点）的整数。"
            "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
            "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、"
            "计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
            f"{rng.choice(endings)}{rng.choice(separator_hints)}"
            "直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。"
        )
        challenge_id = secrets.token_hex(7)
        challenges.append(
            {
                "id": f"probe-{index + 1}-{challenge_id}",
                "expected_count": length,
                "prompt": prompt,
            }
        )
    return challenges


# ---------------------------------------------------------------------------
# 指纹库加载
# ---------------------------------------------------------------------------


def default_cache_dir() -> Path:
    override = os.environ.get("MODELTRACE_CACHE_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "modeltrace"
    return Path.home() / ".cache" / "modeltrace"


def load_bank(bank_url: str = BANK_URL_DEFAULT, cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "unified_bank.json"
    local_bank = cache_dir / "unified_bank.local.json"
    # 允许离线/自备指纹库
    for path in (local_bank, local):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    try:
        with urllib.request.urlopen(bank_url, timeout=60) as response:
            raw = response.read()
        bank = json.loads(raw.decode("utf-8"))
        local.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
        return bank
    except Exception as error:
        raise RuntimeError(
            f"无法获取指纹库：{error}\n"
            f"可手动下载到 {local}，或用 --bank-url / --bank-file 指定。"
        ) from error


# ---------------------------------------------------------------------------
# API 客户端
# ---------------------------------------------------------------------------


def upstream_user_agent() -> str:
    override = (
        os.environ.get("MODELTRACE_USER_AGENT") or os.environ.get("GPT56_USER_AGENT") or ""
    ).strip()
    return override or DEFAULT_UPSTREAM_USER_AGENT


def completion_url(base_url: str, api_format: str = "openai") -> str:
    normalized = base_url.rstrip("/")
    if api_format == "anthropic":
        if normalized.endswith("/messages"):
            return normalized
        if normalized.endswith("/v1"):
            return normalized + "/messages"
        return normalized + "/v1/messages"
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def _looks_like_waf_block(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("cloudflare", "just a moment", "cf-ray", "access denied", "attention required")
    )


def _compact_upstream_error(details: str, fallback: str) -> str:
    text = (details or fallback or "").strip()
    if not text:
        return "上游接口返回错误"
    if _looks_like_waf_block(text):
        return (
            "请求被上游网关拦截（Cloudflare/WAF 拦截页）。"
            "请确认 base_url 指向 API 端点而非网页地址、API Key 有效，"
            "或该服务是否限制当前网络/IP"
        )
    if text.startswith("<") or ("{" not in text and "html" in text.lower()):
        return text[:200]
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
            return json.dumps(payload, ensure_ascii=False)[:500]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return text[:500]


def _request_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str,
    system_prompt: str = "",
) -> str:
    if api_format == "anthropic":
        body_data = {
            "model": api_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body_data["system"] = system_prompt
        headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    else:
        body_data = {
            "model": api_model,
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    if temperature is not None:
        body_data["temperature"] = temperature
    body = json.dumps(body_data).encode("utf-8")
    url = completion_url(base_url, api_format)
    payload = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            message = _compact_upstream_error(details, error.reason)
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            if attempt < MAX_ATTEMPTS and error.code in RETRYABLE_STATUS:
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            raise RuntimeError(f"HTTP {error.code}: {message}{retried}") from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", str(error))
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            raise RuntimeError(f"无法连接接口：{reason}{retried}") from error
    if payload is None:
        raise RuntimeError("接口未返回有效 JSON")
    if api_format == "anthropic":
        content = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        stop_reason = payload.get("stop_reason")
        if stop_reason == "refusal":
            raise RuntimeError("模型拒绝生成，本次回答不计入")
        if stop_reason == "max_tokens":
            raise RuntimeError("回答因 max_tokens 截断，本次回答不计入")
    else:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise RuntimeError(
                f"回答未正常完成（{choice['finish_reason']}），本次回答不计入"
            )
    return str(content)


def request_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str = "auto",
    system_prompt: str = "",
) -> str:
    if api_format != "auto":
        return _request_completion(
            base_url, api_key, api_model, prompt, temperature, api_format, system_prompt
        )
    formats = ("openai", "anthropic")
    errors = []
    for candidate in formats:
        try:
            return _request_completion(
                base_url, api_key, api_model, prompt, temperature, candidate, system_prompt
            )
        except RuntimeError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("接口格式自动探测失败；" + "；".join(errors))


def test_automatic(
    base_url: str,
    api_key: str,
    api_model: str,
    temperature: float | None,
    bank: dict,
    api_format: str = "auto",
    target_count: int = 3,
    max_attempts: int | None = None,
    save_outputs: Path | None = None,
) -> dict:
    if max_attempts is None:
        max_attempts = max(target_count + 3, 6)
    challenges = generate_challenges(max_attempts)
    outputs = []
    raw_rows = []
    errors = []
    for index, challenge in enumerate(challenges, start=1):
        print(
            f"[{index}/{max_attempts}] 请求挑战 {challenge['id']} "
            f"(期望约 {challenge['expected_count']} 个数)…",
            file=sys.stderr,
        )
        try:
            text = request_completion(
                base_url,
                api_key,
                api_model,
                challenge["prompt"],
                temperature,
                api_format,
            )
            minimum = max(80, math.ceil(challenge["expected_count"] * 0.55))
            parsed_count = len(parse_numbers(text))
            accepted = parsed_count >= minimum
            raw_rows.append(
                {
                    "challenge_id": challenge["id"],
                    "expected_count": challenge["expected_count"],
                    "prompt": challenge["prompt"],
                    "text": text,
                    "parsed_numbers": parsed_count,
                    "minimum_numbers": minimum,
                    "accepted": accepted,
                }
            )
            if accepted:
                outputs.append(
                    {
                        "text": text,
                        "expected_count": challenge["expected_count"],
                    }
                )
                print(f"    接受：解析到 {parsed_count} 个数字", file=sys.stderr)
            else:
                errors.append(f"有效数字不足：{parsed_count}/{minimum}")
                print(f"    拒绝：解析到 {parsed_count}/{minimum}", file=sys.stderr)
        except Exception as error:
            errors.append(str(error))
            raw_rows.append(
                {
                    "challenge_id": challenge["id"],
                    "expected_count": challenge["expected_count"],
                    "prompt": challenge["prompt"],
                    "text": "",
                    "error": str(error),
                }
            )
            print(f"    失败：{error}", file=sys.stderr)
        if len(outputs) == target_count:
            break
    if save_outputs is not None:
        save_outputs.parent.mkdir(parents=True, exist_ok=True)
        with save_outputs.open("w", encoding="utf-8") as handle:
            for row in raw_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"原始输出已写入: {save_outputs}", file=sys.stderr)
    result = analyze_global_outputs(outputs, bank)
    result["api_test"] = {
        "requested": target_count,
        "attempted": len(raw_rows),
        "max_attempts": max_attempts,
        "received": len(outputs),
        "errors": errors,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str, str]:
    base_url = (
        args.base_url
        or os.environ.get("MODELTRACE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip()
    api_key = (
        args.api_key
        or os.environ.get("MODELTRACE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    model = (
        args.model
        or os.environ.get("MODELTRACE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or ""
    ).strip()
    return base_url, api_key, model


def split_manual_outputs(pasted: str) -> list[dict]:
    blocks = [
        block.strip()
        for block in re.split(r"(?m)^\s*===OUTPUT===\s*$", pasted)
        if block.strip()
    ]
    if not blocks:
        blocks = [pasted.strip()] if pasted.strip() else []
    return [{"text": block, "expected_count": 0} for block in blocks]


def format_report(result: dict, top: int = 8) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append("ModelTrace 归因结果")
    lines.append("=" * 56)
    lines.append(
        f"预测模型: {result['prediction_name']}  "
        f"P={result['probability']:.1%}"
    )
    lines.append(
        f"预测家族: {result['family_prediction_name']}  "
        f"P={result['family_probability']:.1%}"
    )
    lines.append(
        f"有效回答: {result['used_outputs']}  "
        f"校准: β={result['calibration']['beta']:.3f} "
        f"CV={result['calibration']['cv_accuracy']}"
    )
    lines.append("")
    lines.append(f"{'模型':<28} {'概率':>8} {'家族内':>8} {'相似度':>8}")
    lines.append("-" * 56)
    for item in result["results"][:top]:
        lines.append(
            f"{item['display_name']:<28} "
            f"{item['probability']:>7.1%} "
            f"{item.get('conditional_probability', 0):>7.1%} "
            f"{item.get('profile_similarity', 0):>8.3f}"
        )
    lines.append("")
    lines.append("家族概率:")
    for family in result.get("family_probabilities", []):
        lines.append(f"  {family['display_name']:<12} {family['probability']:>7.1%}")
    if result.get("api_test"):
        api = result["api_test"]
        lines.append("")
        lines.append(
            f"API 测试: 成功 {api['received']}/{api['requested']} "
            f"(尝试 {api['attempted']}/{api['max_attempts']})"
        )
        for err in api.get("errors", [])[:5]:
            lines.append(f"  - {err}")
    lines.append("")
    lines.append("注：结果仅供参考，不是判断模型的决定性证据。")
    lines.append("    未收录模型仍会被归到最相似的现有候选。")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modeltrace_cli.py",
        description="ModelTrace 终端在线评测：长整数挑战 + 统一指纹库模型归因",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动 API 测试（OpenAI 兼容）
  python modeltrace_cli.py -m gpt-5.6-sol -u https://api.example.com/v1 -k sk-xxx

  # 环境变量方式（适合 wget 管道，避免 key 进 shell history）
  export OPENAI_BASE_URL=https://api.example.com/v1
  export OPENAI_API_KEY=sk-xxx
  wget -qO- "https://raw.githubusercontent.com/you/repo/main/modeltrace_cli.py" \\
    | python3 - -m gpt-5.6-sol

  # 手动模式：三份完整输出用 ===OUTPUT=== 分隔
  cat outputs.txt | python3 modeltrace_cli.py --manual

  # 只要 JSON
  python3 modeltrace_cli.py -m gpt-5.6-sol --json
""",
    )
    parser.add_argument("-m", "--model", help="请求 API 时使用的模型名")
    parser.add_argument("-u", "--base-url", help="OpenAI/Anthropic 兼容 Base URL")
    parser.add_argument("-k", "--api-key", help="API Key（也可用环境变量）")
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=3,
        help="目标有效回答数（默认 3，对应校准表）",
    )
    parser.add_argument("-t", "--temperature", type=float, default=None, help="采样温度")
    parser.add_argument(
        "--format",
        choices=("auto", "openai", "anthropic"),
        default="auto",
        help="API 协议（默认 auto）",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="手动模式：从 stdin 读取回答（===OUTPUT=== 分隔）",
    )
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    parser.add_argument("--top", type=int, default=8, help="文本报告显示前 N 个候选")
    parser.add_argument("--bank-url", default=BANK_URL_DEFAULT, help="指纹库 URL")
    parser.add_argument("--bank-file", type=Path, help="本地指纹库 JSON 路径")
    parser.add_argument("--cache-dir", type=Path, help="指纹库缓存目录")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="自动模式最多发起几次挑战（默认 max(n+3, 6)）",
    )
    parser.add_argument(
        "--save-outputs",
        type=Path,
        help="把原始模型输出存成 JSONL，便于复测/手动归因",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.bank_file:
            bank = json.loads(Path(args.bank_file).read_text(encoding="utf-8"))
        else:
            bank = load_bank(args.bank_url, args.cache_dir)
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    base_url, api_key, model = resolve_credentials(args)
    has_api = bool(base_url and api_key and model)

    try:
        if args.manual or (not has_api and not sys.stdin.isatty() and not (args.model or args.base_url)):
            if sys.stdin.isatty() and args.manual:
                print(
                    "手动模式：粘贴完整输出，多份用 ===OUTPUT=== 分隔，EOF 结束：",
                    file=sys.stderr,
                )
            pasted = sys.stdin.read()
            outputs = split_manual_outputs(pasted)
            result = analyze_global_outputs(outputs, bank)
        else:
            if not has_api:
                parser.error(
                    "自动模式需要 base-url / api-key / model。\n"
                    "可用参数 -u -k -m，或环境变量 "
                    "OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL "
                    "（或 MODELTRACE_*）。"
                )
            if args.samples < 1:
                parser.error("-n/--samples 至少为 1")
            result = test_automatic(
                base_url=base_url,
                api_key=api_key,
                api_model=model,
                temperature=args.temperature,
                bank=bank,
                api_format=args.format,
                target_count=args.samples,
                max_attempts=args.max_attempts,
                save_outputs=args.save_outputs,
            )
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
