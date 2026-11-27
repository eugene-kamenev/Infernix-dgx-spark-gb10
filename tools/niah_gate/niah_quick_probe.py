#!/usr/bin/env python3
"""Fast (~15-60s) 262K NIAH premature-EOS probe using assistant continuation.

Replays the seed=1234, 2048-needle 262K NIAH trajectory up to needle 285
(`mesa-d-peninsula: 36738\\njag`, ~210K tokens) via assistant continuation so
regressions or fast-path interactions can be checked in seconds instead of a
full 5-10 minute NIAH Gate run.

Exits 0 if greedy `at_jag` and `before_jag` both continue past `jag`
(`finish_reason == "length"`), or 1 if either hits premature EOS.
"""

from __future__ import annotations

import argparse
from collections import Counter
import http.client
import json
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import niah_gate as ng  # noqa: E402


DEFAULT_FIXTURE_DIR = SCRIPT_DIR / "fixtures" / "seed1234_jag"
DEFAULT_PROMPT_CACHE = Path("/tmp/niah_quick_probe_prompt_seed1234.txt")


def wait_for_server(base_url: str, timeout_s: float = 120.0) -> str:
    host, port, use_tls = ng.parse_base_url(base_url)
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        conn = (
            http.client.HTTPSConnection(host, port, timeout=5.0)
            if use_tls
            else http.client.HTTPConnection(host, port, timeout=5.0)
        )
        try:
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200:
                data = json.loads(raw)
                models = data.get("data") or []
                return str(models[0]["id"]) if models else "default"
            last_err = f"HTTP {resp.status}: {raw[:200]}"
        except Exception as exc:
            last_err = str(exc)
        finally:
            conn.close()
        time.sleep(1.0)
    raise RuntimeError(f"Server at {base_url} not ready after {timeout_s}s: {last_err}")


def get_or_build_prompt(base_url: str, model: str, cache_path: Path) -> str:
    if cache_path.is_file():
        prompt = cache_path.read_text(encoding="utf-8")
        measured = ng.probe_prompt_tokens(
            base_url, model, prompt, seed=1234, reasoning_effort="medium"
        )
        print(f"[prompt] loaded cached prompt ({len(prompt)} chars), measured prompt_tokens={measured}")
        return prompt

    needles = ng.generate_needles(2048, seed=1234, prefix_collisions=False)
    paragraphs = ng.load_corpus_paragraphs(ng.DEFAULT_CORPUS_DIR)
    target_tokens = ng.target_needle_region_tokens(262144, None)
    prompt, _haystack, placement, measured, iters, converged = ng.calibrate_haystack(
        paragraphs=paragraphs,
        needles=needles,
        target_tokens=target_tokens,
        story_on=True,
        seed=1234,
        probe_fn=lambda p: ng.probe_prompt_tokens(
            base_url, model, p, seed=1234, reasoning_effort="medium"
        ),
    )
    print(
        f"[prompt] calibrated: measured={measured} iters={iters} converged={converged} "
        f"placement_valid={placement.valid}"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(prompt, encoding="utf-8")
    return prompt


def chat_non_stream(
    base_url: str,
    model: str,
    user_prompt: str,
    assistant_content: str,
    max_tokens: int,
    ignore_eos: bool = False,
    temperature: float = 0.0,
    top_p: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    seed: int = 1234,
) -> dict:
    host, port, use_tls = ng.parse_base_url(base_url)
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "reasoning_effort": "none",
        "ignore_eos": ignore_eos,
        "stream": False,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    if top_k is not None:
        payload["top_k"] = top_k

    body = json.dumps(payload).encode("utf-8")
    conn = (
        http.client.HTTPSConnection(host, port, timeout=300.0)
        if use_tls
        else http.client.HTTPConnection(host, port, timeout=300.0)
    )
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        elapsed = time.monotonic() - t0
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:500]}")
        obj = json.loads(raw)
        obj["_elapsed_s"] = elapsed
        return obj
    finally:
        conn.close()


def chat_stream(
    base_url: str,
    model: str,
    user_prompt: str,
    assistant_content: str,
    max_tokens: int,
) -> dict:
    host, port, use_tls = ng.parse_base_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 1234,
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode("utf-8")
    conn = (
        http.client.HTTPSConnection(host, port, timeout=180.0)
        if use_tls
        else http.client.HTTPConnection(host, port, timeout=180.0)
    )
    t0 = time.monotonic()
    content_parts: list[str] = []
    finish_reason = "unknown"
    usage: dict = {}
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raw = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {raw[:500]}")
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
        return {
            "content": "".join(content_parts),
            "finish_reason": finish_reason,
            "usage": usage,
            "elapsed_s": time.monotonic() - t0,
        }
    finally:
        conn.close()


def print_probe(name: str, obj: dict) -> tuple[str | None, str]:
    usage = obj.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    fr = choice.get("finish_reason")
    content = msg.get("content") or ""
    print(
        f"[{name}] elapsed={obj.get('_elapsed_s', 0):.2f}s "
        f"prompt_tokens={pt} cached_tokens={cached} completion_tokens={ct} "
        f"finish_reason={fr} content={content!r}"
    )
    return fr, content


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast 262K NIAH premature-EOS probe.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("jag", "after-think"), default="jag")
    parser.add_argument("--sample-dist", type=int, default=0, help="Run N T=1.0 samples at jag")
    parser.add_argument("--only-dist", action="store_true", help="Skip greedy probes (for warm servers)")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--prompt-cache", type=Path, default=DEFAULT_PROMPT_CACHE)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    model = wait_for_server(args.base_url, timeout_s=args.timeout)
    print(f"[server] ready model={model}")
    prompt = get_or_build_prompt(args.base_url, model, args.prompt_cache)
    reasoning = (args.fixture_dir / "reasoning.txt").read_text(encoding="utf-8")
    answer = (args.fixture_dir / "answer.txt").read_text(encoding="utf-8")
    if not answer.endswith("mesa-d-peninsula: 36738\njag"):
        raise RuntimeError(f"Unexpected fixture answer tail: {answer[-60:]!r}")

    passed = True
    if args.mode == "jag":
        asst_jag = f"<think>\n{reasoning}\n</think>\n\n{answer}"
        if not args.only_dist:
            p1 = chat_non_stream(args.base_url, model, prompt, asst_jag, max_tokens=32, ignore_eos=False)
            fr1, c1 = print_probe("at_jag", p1)

            p2 = chat_non_stream(args.base_url, model, prompt, asst_jag, max_tokens=16, ignore_eos=True)
            print_probe("at_jag_ignore_eos", p2)

            asst_before = f"<think>\n{reasoning}\n</think>\n\n{answer[:-3]}"
            p3 = chat_non_stream(args.base_url, model, prompt, asst_before, max_tokens=32, ignore_eos=False)
            fr3, c3 = print_probe("before_jag", p3)

            passed = (
                fr1 == "length"
                and c1.startswith("uar-n-ledge: 54516")
                and fr3 == "length"
                and c3.startswith("jaguar-n-ledge: 54516")
            )
            print(f"[verdict] {'PASS' if passed else 'FAIL'}")

        if args.sample_dist > 0:
            hist: Counter[tuple[str | None, str | None]] = Counter()
            for s in range(1, args.sample_dist + 1):
                res = chat_non_stream(
                    args.base_url,
                    model,
                    prompt,
                    asst_jag,
                    max_tokens=1,
                    ignore_eos=False,
                    temperature=1.0,
                    top_p=1.0,
                    min_p=0.0,
                    top_k=20,
                    seed=s,
                )
                ch = (res.get("choices") or [{}])[0]
                fr = ch.get("finish_reason")
                ct = (ch.get("message") or {}).get("content")
                hist[(fr, ct)] += 1
            print(f"[sample_dist N={args.sample_dist}] histogram={dict(hist)}")
    else:
        asst_after_think = f"<think>\n{reasoning}\n</think>\n\n"
        res = chat_stream(args.base_url, model, prompt, asst_after_think, max_tokens=4096)
        usage = res.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        lines = len(ng.ANSWER_LINE_RE.findall(res["content"]))
        passed = res["finish_reason"] == "length" and lines >= 285
        print(
            f"[after_think] elapsed={res['elapsed_s']:.2f}s prompt_tokens={pt} cached_tokens={cached} "
            f"completion_tokens={ct} finish_reason={res['finish_reason']} matched_lines={lines} "
            f"tail={res['content'][-160:]!r}"
        )
        print(f"[verdict] {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
