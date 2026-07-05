"""
Tests for the real TeacherModel backends (synth_common/clients.py) using an
injected fake SDK client — no network, no API key, no anthropic/openai install.
"""

from __future__ import annotations

import tempfile
import unittest

from sft import SFTConfig, synthesize_diverse
from synth_common import CachingTeacher, ClaudeTeacher, MockTeacher, OpenAICompatibleTeacher


# --- fake Anthropic SDK surface ------------------------------------------------
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(kwargs)


class FakeAnthropic:
    def __init__(self, handler):
        self.messages = _FakeMessages(handler)


class TestClaudeTeacher(unittest.TestCase):
    def test_does_not_forward_temperature_by_default(self):
        # Opus 4.8 rejects sampling params; the teacher must not send temperature.
        fake = FakeAnthropic(lambda kw: _Resp("hello"))
        t = ClaudeTeacher(client=fake)
        out = t.generate("hi", temperature=0.5)
        self.assertEqual(out, "hello")
        sent = fake.messages.calls[0]
        self.assertNotIn("temperature", sent)
        self.assertEqual(sent["thinking"], {"type": "adaptive"})   # adaptive by default
        self.assertEqual(sent["model"], "claude-opus-4-8")

    def test_forwards_temperature_when_enabled(self):
        fake = FakeAnthropic(lambda kw: _Resp("ok"))
        t = ClaudeTeacher(client=fake, forward_sampling=True, model="claude-haiku-4-5",
                          thinking=None, effort="medium")
        t.generate("hi", temperature=0.3)
        sent = fake.messages.calls[0]
        self.assertEqual(sent["temperature"], 0.3)
        self.assertNotIn("thinking", sent)                          # omitted when None
        self.assertEqual(sent["output_config"], {"effort": "medium"})

    def test_refusal_returns_empty(self):
        fake = FakeAnthropic(lambda kw: _Resp("", stop_reason="refusal"))
        t = ClaudeTeacher(client=fake)
        self.assertEqual(t.generate("hi"), "")

    def test_drives_a_real_synthesis_pipeline(self):
        # The fake SDK delegates content to a stateful MockTeacher, so ClaudeTeacher
        # produces coherent, executable answer/tests and the pipeline validates them.
        mock = MockTeacher()

        def handler(kw):
            user = kw["messages"][0]["content"]
            return _Resp(mock.generate(user))

        teacher = ClaudeTeacher(client=FakeAnthropic(handler))
        out = list(synthesize_diverse(["Learn Python lists."], SFTConfig().diverse, teacher, seed=1))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].meta["validated"])


# --- fake OpenAI-compatible surface -------------------------------------------
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _ChatResp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class FakeOpenAI:
    def __init__(self, content):
        self.calls = []
        chat = type("C", (), {})()
        completions = type("Comp", (), {})()

        def create(**kwargs):
            self.calls.append(kwargs)
            return _ChatResp(content)

        completions.create = create
        chat.completions = completions
        self.chat = chat


class TestOpenAICompatibleTeacher(unittest.TestCase):
    def test_vllm_style_call(self):
        fake = FakeOpenAI("def f(): return 1")
        t = OpenAICompatibleTeacher(model="Qwen2.5-Coder", client=fake)
        out = t.generate("write f", temperature=0.7)
        self.assertEqual(out, "def f(): return 1")
        sent = fake.calls[0]
        self.assertEqual(sent["temperature"], 0.7)               # open models accept it
        self.assertEqual(sent["model"], "Qwen2.5-Coder")


class _CountingTeacher:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, temperature=1.0, max_tokens=2048):
        self.calls += 1
        return f"resp:{prompt}"


class TestCachingTeacher(unittest.TestCase):
    def test_cache_hit_avoids_second_call(self):
        with tempfile.TemporaryDirectory() as d:
            inner = _CountingTeacher()
            cached = CachingTeacher(inner, cache_dir=d, model_tag="test")
            a = cached.generate("same prompt")
            b = cached.generate("same prompt")     # served from disk
            c = cached.generate("other prompt")
            self.assertEqual(a, b)
            self.assertEqual(inner.calls, 2)        # only 2 underlying calls (not 3)
            self.assertEqual(cached.hits, 1)
            self.assertEqual(cached.misses, 2)


if __name__ == "__main__":
    unittest.main()
