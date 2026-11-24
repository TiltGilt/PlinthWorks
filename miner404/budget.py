from __future__ import annotations

import json
import threading
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text())
CREDIT_LIMIT = PRICING["credits_per_prompt"]
CALL_LIMIT = PRICING["calls_per_prompt"]
LONG_CONTEXT = PRICING["long_context_above_input_tokens"]
SAFETY = 0.06
TOKENS_PER_IMAGE = 3000
CHARS_PER_TOKEN = 3.0


def call_credits(model: str, input_tokens: int, output_tokens: int) -> float:
    weights = PRICING["models"][model]
    long = input_tokens > LONG_CONTEXT
    return (input_tokens * weights["long_input_credits_per_token" if long else "input_credits_per_token"]
            + output_tokens * weights["long_output_credits_per_token" if long else "output_credits_per_token"])


RESERVE_OUTPUT = {("gpt-6-astra", 22000): 18500, ("gpt-6.1-sol", 44000): 36000, ("gpt-6.1-sol", 32768): 24000,
                  ("gpt-6.1-sol", 8192): 2048}


# Real-time budget: calls are planned and admitted on the measured p90 output of each call type and realistic input,
# against the usable budget; a second, hard check keeps actual spend plus the full output cap (and a conservative input
# estimate) of every call in flight within the credit limit, so no prompt can go over budget whatever its outputs.
EXPECT_OUTPUT = {("gpt-6-astra", 22000): 10200, ("gpt-6.1-sol", 44000): 24000, ("gpt-6.1-sol", 32768): 12700,
                 ("gpt-6.1-sol", 24000): 14700, ("gpt-6.1-sol", 20000): 14000, ("gpt-6.1-sol", 16384): 2500,
                 ("gpt-6.1-sol", 8192): 400, ("gpt-6.1-sol", 4096): 240, ("gpt-6.1-sol", 2000): 100,
                 ("gpt-6.1-sol", 1024): 150, ("gpt-6-luna", 2048): 350}
WORST_OUTPUT = {**RESERVE_OUTPUT, ("gpt-6.1-sol", 24000): 19500}


class Estimate(float):
    """An expected cost (the float) that also carries the call's worst case (.worst)."""
    worst: float


def worst_case(model: str, text_chars: int, images: int, max_output_tokens: int) -> float:
    input_tokens = int(text_chars / CHARS_PER_TOKEN) + images * TOKENS_PER_IMAGE
    worst = call_credits(model, input_tokens, max_output_tokens)  # the hard bound: output can never pass the call's cap
    expected_input = int(text_chars / 3.5) + images * 1050
    expected = call_credits(model, expected_input, EXPECT_OUTPUT.get((model, max_output_tokens), min(max_output_tokens, WORST_OUTPUT.get((model, max_output_tokens), max_output_tokens))))
    out = Estimate(min(expected, worst))
    out.worst = worst
    return out


def _worst(e: float) -> float:
    return getattr(e, "worst", e)


class Budget:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.credits = 0.0
        self._reserved_calls = 0
        self._reserved_credits = 0.0
        self._held_calls = 0
        self._held_credits = 0.0
        self._reserved_worst = 0.0

    def hold(self, calls: int, credits: float) -> None:
        with self._lock:
            self._held_calls, self._held_credits = calls, credits

    def release(self) -> None:
        with self._lock:
            self._held_calls, self._held_credits = 0, 0.0

    def allows(self, estimates: list[float], keep_calls: int = 0, keep_credits: float = 0.0) -> bool:
        with self._lock:
            calls = self.calls + self._reserved_calls + self._held_calls + len(estimates) + keep_calls
            credits = self.credits + self._reserved_credits + self._held_credits + sum(estimates) + keep_credits
            hard = self.credits + self._reserved_worst + sum(_worst(e) for e in estimates)
            return calls <= CALL_LIMIT and credits <= CREDIT_LIMIT * (1 - SAFETY) and hard <= CREDIT_LIMIT

    def reserve(self, estimate: float) -> bool:
        with self._lock:
            if (self.calls + self._reserved_calls + self._held_calls + 1 > CALL_LIMIT
                    or self.credits + self._reserved_credits + self._held_credits + estimate > CREDIT_LIMIT * (1 - SAFETY)
                    or self.credits + self._reserved_worst + _worst(estimate) > CREDIT_LIMIT):
                return False
            self._reserved_calls += 1
            self._reserved_credits += estimate
            self._reserved_worst += _worst(estimate)
            return True

    def settle(self, estimate: float, model: str, usage: object | None) -> None:
        if usage is not None:
            charged = call_credits(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        else:
            charged = estimate
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_credits -= estimate
            self._reserved_worst -= _worst(estimate)
            self.calls += 1
            self.credits += charged
