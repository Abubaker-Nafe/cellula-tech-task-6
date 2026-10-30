"""Offline regression checks: python -m unittest discover -s tests -v.

No real API keys, Redis writes, provider requests, or model downloads are used.
"""

import asyncio
import importlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import RunnableLambda
from langchain_redis import RedisCache

from evaluation_schema import (
    EvaluationResult,
    validate_evaluation_generation,
    validate_evaluation_response,
)
from llm_cache import LoggedRedisCache
import llm_config


class EvaluatorRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = {}
        cls.responses = []
        cls.calls = []
        cls.stack = ExitStack()
        cls.stack.enter_context(patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "synthetic-test-key",
            "OPENROUTER_MODEL": llm_config.DEFAULT_MODEL,
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
        }))

        def initialize(cache, *args, **kwargs):
            cache.prefix = kwargs.get("prefix", "redis")
            cache.ttl = kwargs.get("ttl")

        def lookup(cache, prompt, config):
            return cls.entries.get((cache.prefix, prompt, config))

        def update(cache, prompt, config, values):
            cls.entries[(cache.prefix, prompt, config)] = values

        def send(**kwargs):
            cls.calls.append(kwargs)
            result = cls.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        cls.stack.enter_context(patch.object(RedisCache, "__init__", initialize))
        cls.stack.enter_context(patch.object(RedisCache, "lookup", lookup))
        cls.stack.enter_context(patch.object(RedisCache, "update", update))
        cls.stack.enter_context(patch.object(
            llm_config, "OpenRouter",
            side_effect=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(send=send)),
        ))
        cls.evaluator = importlib.import_module("evaluator")
        cls.workflow = importlib.import_module("workflow")
        cls.original_generator = cls.workflow.generator_chain
        cls.workflow.generator_chain = RunnableLambda(lambda inputs: "4")

    @classmethod
    def tearDownClass(cls):
        cls.workflow.generator_chain = cls.original_generator
        cls.stack.close()

    def setUp(self):
        self.entries.clear()
        self.responses.clear()
        self.calls.clear()
        self.output = io.StringIO()
        self.capture = redirect_stdout(self.output)
        self.capture.__enter__()
        self.inputs = {
            "question": "What is the maximum?",
            "context": "The maximum is 4.",
            "answer": "4",
        }

    def tearDown(self):
        self.capture.__exit__(None, None, None)

    @staticmethod
    def response(text, finish="stop"):
        return {
            "id": "synthetic-test",
            "model": "synthetic",
            "choices": [{
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }],
        }

    def test_truncated_json_is_not_approved_even_if_langchain_repairs_it(self):
        raw = '{"approved":true,"feedback":"Supported'
        self.responses.append(self.response(raw, "length"))
        result = self.evaluator.structured_model.invoke(
            self.evaluator.prompt.invoke(self.inputs)
        )
        # Exercise the actual SDK response conversion and LangChain parser.
        self.assertTrue(result["parsed"].approved)
        self.assertFalse(self.evaluator.resolve_evaluation(result).approved)
        self.assertEqual(self.entries, {})
        schema = self.calls[0]["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["required"]), {"approved", "feedback"})

    def test_complete_json_with_length_finish_is_rejected_and_not_cached(self):
        self.responses.append(self.response(
            '{"approved":true,"feedback":"Supported"}', "length"
        ))
        self.assertFalse(self.evaluator.evaluator_chain.invoke(self.inputs).approved)
        self.assertEqual(self.entries, {})

    def test_invalid_shapes_remain_unvalidated_and_uncached(self):
        invalid = [
            'User Safety: safe\nResponse Safety: safe',
            '{"approved":"yes","feedback":"x"}',
            '{"approved":true}',
            '{"approved":true,"feedback":"x","extra":1}',
            '{"approved":false,"feedback":""}',
            '{"approved":true,"feedback":"   "}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.responses.append(self.response(raw))
                result = self.evaluator.evaluator_chain.invoke(self.inputs)
                self.assertFalse(result.approved)
                self.assertIn("unvalidated", result.feedback)
                self.assertEqual(self.entries, {})

    def test_valid_approvals_and_rejections_are_cacheable(self):
        for approved in (True, False):
            with self.subTest(approved=approved):
                self.entries.clear()
                before = len(self.calls)
                self.responses.append(self.response(json.dumps({
                    "approved": approved, "feedback": "  Evidence checked.  ",
                })))
                first = self.evaluator.evaluator_chain.invoke(self.inputs)
                second = self.evaluator.evaluator_chain.invoke(self.inputs)
                self.assertIs(first.approved, approved)
                self.assertEqual(second.feedback, "Evidence checked.")
                self.assertEqual(len(self.calls), before + 1)

    def test_invalid_then_valid_then_repeat_uses_two_provider_calls(self):
        self.responses.extend([
            self.response("not JSON"),
            self.response('{"approved":true,"feedback":"Supported"}'),
        ])
        self.assertFalse(self.evaluator.evaluator_chain.invoke(self.inputs).approved)
        self.assertTrue(self.evaluator.evaluator_chain.invoke(self.inputs).approved)
        self.assertTrue(self.evaluator.evaluator_chain.invoke(self.inputs).approved)
        self.assertEqual(len(self.calls), 2)

    def test_invalid_cached_generation_is_ignored_without_deletion(self):
        cache = self.evaluator.model.cache
        invalid = [ChatGeneration(
            message=AIMessage(content='{"approved":true,"feedback":"Supported"}'),
            generation_info={"finish_reason": "length"},
        )]
        key = (cache.prefix, "prompt", "config")
        self.entries[key] = invalid
        self.assertIsNone(cache.lookup("prompt", "config"))
        self.assertIs(self.entries[key], invalid)

    def test_async_updates_use_the_same_validation(self):
        generation = ChatGeneration(
            message=AIMessage(content='{"approved":true,"feedback":"   "}'),
            generation_info={"finish_reason": "stop"},
        )
        asyncio.run(self.evaluator.model.cache.aupdate("p", "m", [generation]))
        self.assertEqual(self.entries, {})

    def test_four_unfinished_assessments_return_latest_unvalidated_answer(self):
        from agent_memory import AgentMemories

        memories = AgentMemories()
        self.responses.extend([
            self.response('{"approved":true,"feedback":"Supported', "length")
            for _ in range(4)
        ])
        result = self.workflow.run_workflow("Maximum?", "Maximum is 4.", memories)
        self.assertEqual(result["iterations"], 4)
        self.assertFalse(result["approved"])
        self.assertEqual(result["status"], "max_iterations_reached")
        self.assertEqual(result["answer"], "4")
        self.assertIn("unvalidated", result["feedback"])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(memories.generator.messages), 8)
        self.assertEqual(len(memories.evaluator.messages), 8)
        self.assertIsNot(memories.generator.messages, memories.evaluator.messages)

    def test_feedback_and_early_approval_still_work(self):
        seen = []
        original = self.workflow.generator_chain
        self.workflow.generator_chain = RunnableLambda(
            lambda inputs: (seen.append(inputs) or "4")
        )
        self.responses.extend([
            self.response('{"approved":false,"feedback":"State the maximum."}'),
            self.response('{"approved":true,"feedback":"Supported."}'),
        ])
        try:
            result = self.workflow.run_workflow("Maximum?", "Maximum is 4.")
        finally:
            self.workflow.generator_chain = original
        self.assertEqual(result["iterations"], 2)
        self.assertTrue(result["approved"])
        self.assertEqual(seen[1]["feedback"], "State the maximum.")

    def test_transport_failure_is_not_disguised_as_approval(self):
        self.responses.append(httpx.ReadTimeout("synthetic timeout"))
        with self.assertRaises(httpx.ReadTimeout):
            self.evaluator.evaluator_chain.invoke(self.inputs)
        self.assertEqual(self.entries, {})


class ResponsePolicyTests(unittest.TestCase):
    def test_missing_finish_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_evaluation_response(AIMessage(
                content='{"approved":true,"feedback":"Supported"}'
            ))

    def test_native_truncation_and_refusal_are_rejected(self):
        for extra, metadata in [
            ({}, {"finish_reason": "stop", "native_finish_reason": "max_tokens"}),
            ({"refusal": "unavailable"}, {"finish_reason": "stop"}),
        ]:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    validate_evaluation_response(AIMessage(
                        content='{"approved":true,"feedback":"Supported"}',
                        additional_kwargs=extra,
                        response_metadata=metadata,
                    ))


class TransportConfigurationTests(unittest.TestCase):
    def test_actual_sdk_makes_one_attempt_for_503(self):
        from openrouter import OpenRouter

        attempts = []

        def respond(request):
            attempts.append(request)
            return httpx.Response(
                503, json={"error": {"code": 503, "message": "synthetic"}},
                request=request,
            )

        with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
            async_client = httpx.AsyncClient(
                transport=httpx.MockTransport(respond), trust_env=False
            )
            sdk = OpenRouter(
                api_key="synthetic-test-key", timeout_ms=60_000,
                retry_config=None, client=client, async_client=async_client,
            )
            try:
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "synthetic-test-key",
                    "LANGCHAIN_TRACING_V2": "false",
                    "LANGSMITH_TRACING": "false",
                }):
                    with patch.object(llm_config, "OpenRouter", return_value=sdk) as constructor:
                        model = llm_config.create_model(cache=False)
                        with self.assertRaises(Exception):
                            model.invoke("synthetic")
                        constructor.assert_called_once_with(
                            api_key="synthetic-test-key",
                            timeout_ms=60_000, retry_config=None,
                        )
            finally:
                asyncio.run(async_client.aclose())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].extensions["timeout"], {
            "connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0,
        })


class CacheSerializationTests(unittest.TestCase):
    def test_real_cache_serialization_preserves_completion_metadata(self):
        class MemoryRedis:
            def __init__(self):
                self.values = {}
                self.expirations = {}

            def client_setinfo(self, *args):
                pass

            def json(self):
                return self

            def set(self, key, path, value):
                self.values[key] = json.loads(json.dumps(value))

            def get(self, key):
                value = self.values.get(key)
                return json.loads(json.dumps(value)) if value is not None else None

            def expire(self, key, ttl):
                self.expirations[key] = ttl

        redis = MemoryRedis()
        cache = LoggedRedisCache(
            redis_client=redis, ttl=3600, prefix="test-only",
            validate_generation=validate_evaluation_generation,
        )
        valid = ChatGeneration(
            message=AIMessage(
                content='{"approved":false,"feedback":"Needs a correction."}',
                response_metadata={"finish_reason": "stop"},
            ),
            generation_info={"finish_reason": "stop"},
        )
        cache.update("p", "m", [valid])
        restored = cache.lookup("p", "m")
        self.assertIsInstance(restored[0], ChatGeneration)
        self.assertFalse(validate_evaluation_generation(restored[0]).approved)
        self.assertEqual(redis.expirations[cache._key("p", "m")], 3600)

    def test_conflicting_completion_metadata_is_rejected(self):
        generation = ChatGeneration(
            message=AIMessage(
                content='{"approved":true,"feedback":"Supported."}',
                response_metadata={"finish_reason": "stop"},
            ),
            generation_info={"finish_reason": "length"},
        )
        with self.assertRaises(ValueError):
            validate_evaluation_generation(generation)


class IngestionTests(unittest.TestCase):
    def test_chunks_preserve_metadata_and_fit_the_token_budget(self):
        from langchain_core.documents import Document
        import ingestion

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=True, **kwargs):
                return list(range(len(text) + (2 if add_special_tokens else 0)))

        tokenizer = CharacterTokenizer()
        text = "\n".join(f"line_{index} = {index};" for index in range(120))
        with patch.object(ingestion, "get_embedding_tokenizer", return_value=tokenizer):
            chunks = ingestion.split_documents([
                Document(page_content=text, metadata={"source": "fixture.py", "page": 2})
            ])
        self.assertGreater(len(chunks), 1)
        self.assertIn("line_119", chunks[-1].page_content)
        for chunk in chunks:
            self.assertLessEqual(len(tokenizer.encode(chunk.page_content)), 256)
            self.assertEqual(chunk.metadata["source"], "fixture.py")
            self.assertEqual(chunk.metadata["page"], 2)
            start = chunk.metadata["start_index"]
            self.assertEqual(text[start:start + len(chunk.page_content)], chunk.page_content)

    def test_ppt_conversion_preserves_original_source_and_slide(self):
        from pptx import Presentation
        import ingestion
        import ppt_ingestion

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.ppt"
            source.write_bytes(b"test-input")
            original_bytes = source.read_bytes()

            def convert(command, **kwargs):
                output_directory = Path(command[command.index("--outdir") + 1])
                presentation = Presentation()
                slide = presentation.slides.add_slide(presentation.slide_layouts[1])
                slide.shapes.title.text = "Four iterations"
                presentation.save(output_directory / "input.pptx")
                self.assertEqual(kwargs["timeout"], 120)
                self.assertTrue(any(arg.startswith("-env:UserInstallation=") for arg in command))
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(ppt_ingestion, "find_libreoffice", return_value="fixture-soffice"):
                with patch.object(ppt_ingestion.subprocess, "run", side_effect=convert):
                    documents = ingestion.load_file(source)
            self.assertIn("Four iterations", documents[0].page_content)
            self.assertEqual(documents[0].metadata, {"source": "original.ppt", "slide": 1})
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_conversion_timeout_is_an_actionable_error(self):
        import ppt_ingestion

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.ppt"
            source.write_bytes(b"fixture")
            with patch.object(ppt_ingestion, "find_libreoffice", return_value="fixture-soffice"):
                with patch.object(
                    ppt_ingestion.subprocess, "run",
                    side_effect=subprocess.TimeoutExpired("fixture", 120),
                ):
                    with self.assertRaisesRegex(ValueError, "120-second"):
                        with ppt_ingestion.converted_pptx(source):
                            self.fail("Conversion should not succeed")


if __name__ == "__main__":
    unittest.main()
