"""
Day 1 — LLM API Foundation
Test suite for student solutions and master keys.

Run from the 02-lab/ folder:
    pytest tests/ -v

All external API calls are mocked — no real API keys or internet connection required.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# Locate directories relative to this test file
TESTS_DIR = Path(__file__).parent
LAB_DIR = TESTS_DIR.parent

# Resolve module paths
SOLUTION_PATH = LAB_DIR / "solution-code" / "solution.py"
TEMPLATE_PATH = LAB_DIR / "starter-code" / "template.py"

def _load_module(path: Path, unique_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Target file not found at: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Dynamically load the solution module if it exists (for grading), otherwise test the starter template
if SOLUTION_PATH.exists():
    _m = _load_module(SOLUTION_PATH, "solution")
else:
    _m = _load_module(TEMPLATE_PATH, "template")

# Import target functions
call_openai = getattr(_m, "call_openai")
call_gemini = getattr(_m, "call_gemini")
call_anthropic = getattr(_m, "call_anthropic")
compare_models = getattr(_m, "compare_models")
streaming_chatbot = getattr(_m, "streaming_chatbot")
retry_with_backoff = getattr(_m, "retry_with_backoff")
batch_compare = getattr(_m, "batch_compare")
format_comparison_table = getattr(_m, "format_comparison_table")


# ---------------------------------------------------------------------------
# Mock Generators
# ---------------------------------------------------------------------------
def _make_openai_response(text: str = "Hello from OpenAI"):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 25
    resp.usage_metadata = usage
    return resp


def _make_anthropic_response(text: str = "Hello from Anthropic"):
    resp = MagicMock()
    content_part = MagicMock()
    content_part.text = text
    resp.content = [content_part]
    usage = MagicMock()
    usage.input_tokens = 15
    usage.output_tokens = 30
    resp.usage = usage
    return resp


def _make_stream_chunks(parts: list[str]):
    chunks = []
    for part in parts:
        chunk = MagicMock()
        chunk.text = part
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestCallOpenAI(unittest.TestCase):

    @patch("openai.OpenAI")
    def test_openai_returns_tuple_structure(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("OpenAI Mock Response")

        text, latency, usage = call_openai("Hello")

        self.assertEqual(text, "OpenAI Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 20)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    @patch("openai.OpenAI")
    def test_openai_sends_expected_chat_params(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response()

        call_openai("Hello", model="gpt-4o-mini", temperature=0.2, top_p=0.8, max_tokens=64)

        self.assertEqual(MockOpenAI.call_args.kwargs.get("api_key"), "test-openai-key")
        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertIn(kwargs["model"], {"gpt-4o-mini", "openai/gpt-4o-mini"})
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["top_p"], 0.8)
        self.assertEqual(kwargs["max_tokens"], 64)


class TestCallGemini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_gemini_returns_tuple_structure_new_sdk(self, MockGenAIClient):
        mock_client = MagicMock()
        MockGenAIClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response("Gemini Mock Response")

        text, latency, usage = call_gemini("Hello")

        self.assertEqual(text, "Gemini Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 25)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"})
    @patch("google.genai.Client")
    def test_gemini_sends_expected_generate_content_params(self, MockGenAIClient):
        mock_client = MagicMock()
        MockGenAIClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response()

        call_gemini("Hello", model=_m.GEMINI_MODEL, temperature=0.1, top_p=0.5, max_tokens=32)

        MockGenAIClient.assert_called_once_with(api_key="test-gemini-key")
        mock_client.models.generate_content.assert_called_once()
        kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], _m.GEMINI_MODEL)
        self.assertEqual(kwargs["contents"], "Hello")
        self.assertIn("config", kwargs)
        config = kwargs["config"]
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.top_p, 0.5)
        self.assertEqual(config.max_output_tokens, 32)


class TestCallAnthropic(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_anthropic_returns_tuple_structure(self, MockAnthropic):
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response("Claude Mock Response")

        text, latency, usage = call_anthropic("Hello")

        self.assertEqual(text, "Claude Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 15)
        self.assertEqual(usage["output_tokens"], 30)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_sends_expected_message_params(self, MockAnthropic):
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response()

        call_anthropic("Hello", model="claude-3-5-haiku", temperature=0.3, top_p=0.7, max_tokens=48)

        MockAnthropic.assert_called_once_with(api_key="test-anthropic-key")
        kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-3-5-haiku")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["top_p"], 0.7)
        self.assertEqual(kwargs["max_tokens"], 48)


class TestCompareModels(unittest.TestCase):

    def test_compare_models_evaluates_correct_costs(self):
        with patch.object(_m, "call_openai") as mock_openai, \
             patch.object(_m, "call_gemini") as mock_gemini:

            # Setup mock returns
            mock_openai.side_effect = [
                ("GPT-4o Response", 0.5, {"input_tokens": 10, "output_tokens": 20}),       # 1st call: gpt-4o
                ("GPT-4o-Mini Response", 0.3, {"input_tokens": 10, "output_tokens": 20}),  # 2nd call: gpt-4o-mini
            ]
            mock_gemini.return_value = (
                "Gemini Response",
                0.4,
                {"input_tokens": 10, "output_tokens": 20}
            )

            result = compare_models("Prompt")

            # Verify keys exist
            self.assertIn("gpt4o", result)
            self.assertIn("gpt4o_mini", result)
            self.assertIn("gemini_flash", result)

            # Verify models were called in the expected order
            self.assertEqual(mock_openai.call_count, 2)
            openai_models = [
                c.kwargs.get("model", c.args[1] if len(c.args) > 1 else None)
                for c in mock_openai.call_args_list
            ]
            self.assertEqual(openai_models[0], "gpt-4o")
            self.assertEqual(openai_models[1], "gpt-4o-mini")
            gemini_kwargs = mock_gemini.call_args.kwargs
            if "model" in gemini_kwargs:
                self.assertEqual(gemini_kwargs["model"], _m.GEMINI_MODEL)

            # Verify payload fields
            self.assertEqual(result["gpt4o"]["response"], "GPT-4o Response")
            self.assertEqual(result["gpt4o"]["latency"], 0.5)
            self.assertEqual(result["gpt4o"]["input_tokens"], 10)
            self.assertEqual(result["gpt4o"]["output_tokens"], 20)

            # Verify exact cost calculations
            # gpt-4o: (10 * 5.0 + 20 * 20.0) / 1,000,000 = 0.00045
            self.assertAlmostEqual(result["gpt4o"]["cost"], 0.00045, places=8)

            # gpt-4o-mini: (10 * 0.150 + 20 * 0.600) / 1,000,000 = 0.0000135
            self.assertAlmostEqual(result["gpt4o_mini"]["cost"], 0.0000135, places=8)

            # gemini-2.5-flash: (10 * 0.075 + 20 * 0.300) / 1,000,000 = 0.00000675
            self.assertAlmostEqual(result["gemini_flash"]["cost"], 0.00000675, places=8)


class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_immediately(self):
        val = retry_with_backoff(lambda: "success")
        self.assertEqual(val, "success")

    def test_succeeds_after_retries(self):
        state = {"attempts": 0}

        def flaky_fn():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("Fail")
            return "ok"

        val = retry_with_backoff(flaky_fn, max_retries=3, base_delay=0.001)
        self.assertEqual(val, "ok")
        self.assertEqual(state["attempts"], 3)

    def test_raises_permanent_exception(self):
        def permanent_fail():
            raise RuntimeError("Permanent")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(permanent_fail, max_retries=2, base_delay=0.001)

    @patch(f"{_m.__name__}.time.sleep")
    def test_uses_exponential_backoff_delays(self, mock_sleep):
        state = {"attempts": 0}

        def flaky_fn():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("Fail")
            return "ok"

        retry_with_backoff(flaky_fn, max_retries=3, base_delay=0.1)

        self.assertEqual(mock_sleep.call_count, 2)
        self.assertAlmostEqual(mock_sleep.call_args_list[0].args[0], 0.1)
        self.assertAlmostEqual(mock_sleep.call_args_list[1].args[0], 0.2)


class TestBatchCompareAndFormat(unittest.TestCase):

    def test_batch_runs_and_formats_table(self):
        with patch.object(_m, "compare_models") as mock_comp:
            def _get_mock(*_args, **_kwargs):
                return {
                    "gpt4o": {"response": "GPT-4o response content here", "latency": 0.5, "cost": 0.00045, "input_tokens": 10, "output_tokens": 20},
                    "gpt4o_mini": {"response": "GPT-4o-Mini response content here", "latency": 0.3, "cost": 0.0000135, "input_tokens": 10, "output_tokens": 20},
                    "gemini_flash": {"response": "Gemini response content here", "latency": 0.4, "cost": 0.00000675, "input_tokens": 10, "output_tokens": 20}
                }
            mock_comp.side_effect = _get_mock

            results = batch_compare(["Q1", "Q2"])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["prompt"], "Q1")
            self.assertEqual(results[1]["prompt"], "Q2")
            self.assertEqual(mock_comp.call_args_list, [call("Q1"), call("Q2")])

            table = format_comparison_table(results)
            self.assertIsInstance(table, str)
            self.assertIn("Prompt", table)
            self.assertIn("GPT-4o", table)
            self.assertIn("Gemini-Flash", table)
            self.assertIn("Q1", table)
            self.assertIn("Latency", table)
            self.assertIn("Tokens (In/Out)", table)
            self.assertIn("Cost (USD)", table)

    def test_format_table_truncates_long_responses(self):
        long_response = "A" * 80
        results = [{
            "prompt": "Long",
            "gpt4o": {"response": long_response, "latency": 0.5, "cost": 0.1, "input_tokens": 1, "output_tokens": 2},
            "gpt4o_mini": {"response": "short", "latency": 0.3, "cost": 0.01, "input_tokens": 1, "output_tokens": 2},
            "gemini_flash": {"response": "short", "latency": 0.4, "cost": 0.001, "input_tokens": 1, "output_tokens": 2},
        }]
        table = format_comparison_table(results)
        self.assertIn("A" * 50 + "...", table)
        self.assertNotIn("A" * 51, table)


class TestStreamingChatbot(unittest.TestCase):

    @patch("builtins.input", side_effect=["quit"])
    @patch("google.genai.Client")
    def test_exits_cleanly(self, MockClient, mock_input):
        # Ensure it quits cleanly without entering infinite loop
        try:
            streaming_chatbot()
        except SystemExit:
            pass
        MockClient.return_value.models.generate_content_stream.assert_not_called()

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["hello", "exit"])
    @patch("google.genai.Client")
    def test_streams_response_chunks(self, MockClient, mock_input, mock_print):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.return_value = _make_stream_chunks(["Xin ", "chao"])

        streaming_chatbot()

        mock_client.models.generate_content_stream.assert_called_once()
        kwargs = mock_client.models.generate_content_stream.call_args.kwargs
        self.assertEqual(kwargs["model"], _m.GEMINI_MODEL)
        printed = "".join(
            str(c.args[0]) if c.args else ""
            for c in mock_print.call_args_list
        )
        self.assertIn("Xin ", printed)
        self.assertIn("chao", printed)

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["t1", "t2", "t3", "t4", "quit"])
    @patch("google.genai.Client")
    def test_keeps_only_last_three_turns(self, MockClient, mock_input, mock_print):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.return_value = _make_stream_chunks(["ok"])

        streaming_chatbot()

        self.assertEqual(mock_client.models.generate_content_stream.call_count, 4)
        last_contents = mock_client.models.generate_content_stream.call_args_list[-1].kwargs["contents"]
        self.assertLessEqual(len(last_contents), 6)
        user_texts = [
            msg["parts"][0]["text"]
            for msg in last_contents
            if msg.get("role") == "user"
        ]
        self.assertEqual(user_texts, ["t2", "t3", "t4"])


if __name__ == "__main__":
    unittest.main()
