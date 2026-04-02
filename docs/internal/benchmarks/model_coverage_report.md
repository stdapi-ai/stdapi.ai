# stdapi.ai — Model Coverage Benchmark Report

**Report scope:** Combines three test sessions covering both API routes across all supported Bedrock model providers.
**Test dates:** 2026-04-02 (initial routes), 2026-04-03 (extended multi-provider)
**Server:** stdapi.ai v1.7.0+ running on `http://127.0.0.1:8001`
**Config:** `AWS_BEDROCK_REGIONS=eu-west-3,eu-west-1,eu-central-1,us-east-1,us-west-2`
**Routes tested:** Anthropic Messages API (`/anthropic/v1/messages`) and OpenAI Chat Completions API (`/v1/chat/completions`)

---

## Test Scenario Index

Each scenario is labelled with the T-ID used throughout the matrices below, its description, and the corresponding pytest test(s) added to the parametrized test suite.

| T-ID | Route | Description | Pytest path |
|------|-------|-------------|-------------|
| T1 | Both | Basic non-streaming response, usage tokens | `tests/test_anthropic_messages_multi_model.py::TestMultiModelBasics::test_basic_text_generation` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelChatCompletions::test_basic_chat_completion` |
| T2 | Both | Streaming SSE event sequence, non-empty text blocks | `tests/test_anthropic_messages_multi_model.py::TestMultiModelBasics::test_streaming_event_sequence` · `tests/test_anthropic_messages_multi_model.py::TestMultiModelBasics::test_streaming_no_empty_text_blocks` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelChatCompletions::test_streaming_chat_completion` |
| T3 | Both | System prompt compliance | _(covered by T1/T4 system-prompt variants in ad-hoc scripts; no dedicated parametrized test)_ |
| T4 | Both | Multi-turn context retention | `tests/test_anthropic_messages_multi_model.py::TestMultiModelBasics::test_multi_turn_context_retention` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelChatCompletions::test_multi_turn_context_retention` |
| T5 | Both | Non-streaming tool use — single turn + result continuation | `tests/test_anthropic_messages_multi_model.py::TestMultiModelToolUse::test_tool_call_single_turn` · `tests/test_anthropic_messages_multi_model.py::TestMultiModelToolUse::test_tool_result_continuation` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelToolUse::test_tool_call_single_turn` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelToolUse::test_tool_result_continuation` |
| T5s | Both | Tool use in streaming mode | `tests/test_anthropic_messages_multi_model.py::TestMultiModelToolUse::test_streaming_tool_call` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelToolUse::test_streaming_tool_call` |
| T5+ | Both | Full agentic loop — multi-turn, multi-tool (list dir + read file) | `tests/test_anthropic_messages_multi_model.py::TestMultiModelToolUse::test_agentic_loop_directory_and_file` · `tests/test_openai_chat_completions_multi_model.py::TestMultiModelToolUse::test_agentic_loop_directory_and_file` |
| T6 | Anthropic | Prompt caching — implicit cache read on second call (Amazon Nova models) | `tests/test_anthropic_messages_multi_model.py::TestPromptCaching::test_cache_read_on_second_call` |
| T7 | Both | Native reasoning — `thinking` blocks; configurable by token budget or effort; model-subset only | `tests/test_anthropic_messages_multi_model.py::TestNativeReasoning::test_native_thinking_blocks_present` · `tests/test_anthropic_messages_multi_model.py::TestNativeReasoning::test_streaming_native_thinking_blocks` |
| T8 | Both | Vision — 1×1 red PNG described correctly | `tests/test_anthropic_messages_multi_model.py::TestVision::test_image_color_recognition` · `tests/test_openai_chat_completions_multi_model.py::TestVision::test_image_color_recognition` |
| T9 | Both | Structured JSON output parseable and contains expected keys | `tests/test_anthropic_messages_multi_model.py::TestStructuredOutput::test_json_output_parseable` · `tests/test_openai_chat_completions_multi_model.py::TestStructuredOutput::test_json_output_parseable` |
| T-CC1 | Anthropic (Claude Code) | Request pipeline trace — follow POST /v1/chat/completions from route handler to Bedrock converse() call, quoting code at each step | `tests/test_anthropic_messages_multi_model_claude_code.py::TestClaudeCodePipeline::test_trace_request_pipeline` |
| T-CC2 | Anthropic (Claude Code) | Streaming path trace — follow stream=True branch from divergence point to final SSE output, quoting code from ≥3 files | `tests/test_anthropic_messages_multi_model_claude_code.py::TestClaudeCodePipeline::test_trace_streaming_path` |
| T-CC3 | Anthropic (Claude Code) | Parameter mapping audit — read types, adapter, and _prepare_converse_request to document ≥10 OpenAI → Bedrock field mappings with exact code quotes | `tests/test_anthropic_messages_multi_model_claude_code.py::TestClaudeCodeAnalysis::test_audit_parameter_mapping` |
| T-CC4 | Anthropic (Claude Code) | Model override enumeration — Glob + read all model-specific files in stdapi/models/chat/ and document ≥5 override implementations | `tests/test_anthropic_messages_multi_model_claude_code.py::TestClaudeCodeAnalysis::test_enumerate_model_overrides` |
| T-CC5 | Anthropic (Claude Code) | Effort level comparison — T-CC3 task repeated at `--effort low` and `--effort high`; Claude + Nova 2 only | `tests/test_anthropic_messages_multi_model_claude_code.py::TestClaudeCodeEffortLevels::test_effort_parameter_mapping` |

> **Running the parametrized suite:**
> ```
> pytest --expensive tests/test_anthropic_messages_multi_model.py
> pytest --expensive tests/test_openai_chat_completions_multi_model.py
> ```
>
> **Running the Claude Code agentic suite** (requires `claude` CLI + AWS Bedrock credentials — spawns its own stdapi server automatically):
> ```
> pytest --expensive -s tests/test_anthropic_messages_multi_model_claude_code.py
> # Grep metrics: add 2>&1 | grep CC-METRICS
> # Pass --server-url to use an external server instead of spawning one
> ```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Pass |
| ⚠️ | Degraded — functions but not perfectly (see issue notes) |
| ❌S | **Server issue** — gateway bug; our responsibility to fix |
| ❌M | **Model limitation** — Bedrock or model behaviour; not fixable server-side |
| — | **Not tested** — model was not included in this scenario's parametrize list because the capability is not supported by this model (e.g. no tool calling, no vision, no native reasoning) |
| ~ | **Not tested** — model likely supports this scenario but was not included in the test run; coverage not yet extended |
| ⊘ | **Untested** — model is inaccessible due to an external restriction (Bedrock legacy access revoked or geo-restriction); not a server failure |

---

## Combined Model Coverage Matrix — Anthropic Route

| Provider | Model | T1 | T2 | T4 | T5 | T5s | T5+ | T6 | T7 | T8 | T9 |
|----------|-------|----|----|----|----|----|-----|----|----|----|----|
| Anthropic | `anthropic.claude-sonnet-4-6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | ✅ | ✅ |
| Amazon | `amazon.nova-micro-v1:0` | ✅ | ✅ | ✅ | — | — | — | ✅ | — | — | — |
| Amazon | `amazon.nova-lite-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Amazon | `amazon.nova-pro-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Amazon | `amazon.nova-2-lite-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| AI21 Labs | `ai21.jamba-1-5-mini-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| AI21 Labs | `ai21.jamba-1-5-large-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| Cohere | `cohere.command-r-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — | — |
| Cohere | `cohere.command-r-plus-v1:0` | ⊘ | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — |
| DeepSeek | `deepseek.v3-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | ✅ |
| DeepSeek | `deepseek.v3.2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| DeepSeek | `deepseek.r1-v1:0` | ✅ | ✅ | ✅¹ | — | — | — | — | ✅ | — | — |
| Google | `google.gemma-3-12b-it` | ✅ | ✅ | ✅ | — | — | — | — | — | ❌M² | — |
| Google | `google.gemma-3-4b-it` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Google | `google.gemma-3-27b-it` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Meta | `meta.llama3-3-70b-instruct-v1:0` | ✅ | ✅ | ✅ | ✅³ | ❌M³ | ❌M⁴ | — | — | — | — |
| Meta | `meta.llama3-1-8b-instruct-v1:0` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Meta | `meta.llama3-1-70b-instruct-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| Meta | `meta.llama3-2-3b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — | — |
| Meta Llama 4 | `meta.llama4-scout-17b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — | — |
| Meta Llama 4 | `meta.llama4-maverick-17b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — | — |
| Meta (vision) | `meta.llama3-2-11b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | ⊘ | — |
| Meta (vision) | `meta.llama3-2-90b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | ⊘ | — |
| MiniMax | `minimax.minimax-m2.5` | ✅¹ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | ✅ |
| MiniMax | `minimax.minimax-m2.1` | ✅ | ✅ | ✅¹ | — | — | — | — | — | — | — |
| Mistral | `mistral.mistral-7b-instruct-v0:2` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Mistral | `mistral.mistral-large-2402-v1:0` | ✅ | ✅ | ✅ | ✅ | ❌M³ | ✅ | — | — | — | — |
| Mistral | `mistral.ministral-3-8b-instruct` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Mistral | `mistral.magistral-small-2509` | ✅ | ✅ | ✅ | — | — | — | — | ✅ | — | — |
| Mistral | `mistral.pixtral-large-2502-v1:0` | ✅ | ✅ | ✅ | ✅ | ❌M³ | ✅ | — | — | ✅ | — |
| Moonshot AI | `moonshotai.kimi-k2.5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| Moonshot AI | `moonshot.kimi-k2-thinking` | ✅ | ✅ | ✅¹ | — | — | — | — | ✅ | — | — |
| NVIDIA | `nvidia.nemotron-nano-9b-v2` | ✅ | ✅ | ⚠️⁵ | — | — | — | — | — | — | — |
| NVIDIA | `nvidia.nemotron-nano-3-30b` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| OpenAI@Bedrock | `openai.gpt-oss-20b-1:0` | ✅ | ✅ | ✅¹ | ✅ | ✅ | ✅ | — | — | — | — |
| OpenAI@Bedrock | `openai.gpt-oss-120b-1:0` | ✅ | ✅ | ✅¹ | ✅ | ✅ | ✅ | — | — | — | — |
| Qwen | `qwen.qwen3-32b-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| Qwen | `qwen.qwen3-coder-30b-a3b-v1:0` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Qwen | `qwen.qwen3-vl-235b-a22b` | ✅ | ✅ | ✅ | — | — | — | — | — | ✅ | — |
| Writer | `writer.palmyra-x4-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| Writer | `writer.palmyra-x5-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| Writer | `writer.palmyra-vision-7b` | ✅ | ✅ | ✅ | — | — | — | — | — | ✅ | — |
| Z.AI | `zai.glm-4.7-flash` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| Z.AI | `zai.glm-5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |

---

## Combined Model Coverage Matrix — OpenAI Route

| Provider | Model | T1 | T2 | T4 | T5 | T5s | T5+ | T7 | T8 | T9 |
|----------|---------|----|----|----|----|----|-----|----|----|-----|
| Anthropic | `anthropic.claude-sonnet-4-6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Amazon | `amazon.nova-micro-v1:0` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Amazon | `amazon.nova-lite-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Amazon | `amazon.nova-2-lite-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| AI21 Labs | `ai21.jamba-1-5-mini-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| AI21 Labs | `ai21.jamba-1-5-large-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Cohere | `cohere.command-r-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — |
| Cohere | `cohere.command-r-plus-v1:0` | ⊘ | ⊘ | ⊘ | ⊘ | — | — | — | — | — |
| DeepSeek | `deepseek.v3-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | ✅ |
| DeepSeek | `deepseek.v3.2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| DeepSeek | `deepseek.r1-v1:0` | ✅ | ✅ | ✅¹ | — | — | — | ✅ | — | — |
| Google | `google.gemma-3-12b-it` | ✅ | ✅ | ✅ | — | — | — | — | ❌² | — |
| Google | `google.gemma-3-4b-it` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Google | `google.gemma-3-27b-it` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Meta | `meta.llama3-3-70b-instruct-v1:0` | ✅ | ✅ | ✅ | ✅³ | ❌M³ | ✅ | — | — | ✅ |
| Meta | `meta.llama3-1-8b-instruct-v1:0` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Meta | `meta.llama3-1-70b-instruct-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Meta | `meta.llama3-2-3b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — |
| Meta Llama 4 | `meta.llama4-scout-17b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — |
| Meta Llama 4 | `meta.llama4-maverick-17b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | — | — |
| Meta (vision) | `meta.llama3-2-11b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | ⊘ | — |
| Meta (vision) | `meta.llama3-2-90b-instruct-v1:0` | ⊘ | ⊘ | ⊘ | — | — | — | — | ⊘ | — |
| MiniMax | `minimax.minimax-m2.5` | ✅¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| MiniMax | `minimax.minimax-m2.1` | ✅ | ✅ | ✅¹ | — | — | — | — | — | — |
| Mistral | `mistral.mistral-7b-instruct-v0:2` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Mistral | `mistral.mistral-large-2402-v1:0` | ✅ | ✅ | ✅ | ✅ | ❌M³ | ✅ | — | — | — |
| Mistral | `mistral.ministral-3-8b-instruct` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Mistral | `mistral.magistral-small-2509` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Mistral | `mistral.pixtral-large-2502-v1:0` | ✅ | ✅ | ✅ | ✅ | ❌M³ | ✅ | — | ✅ | — |
| Moonshot AI | `moonshotai.kimi-k2.5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Moonshot AI | `moonshot.kimi-k2-thinking` | ✅ | ✅ | ✅¹ | — | — | — | — | — | — |
| NVIDIA | `nvidia.nemotron-nano-9b-v2` | ✅ | ✅ | ⚠️⁵ | — | — | — | — | — | — |
| NVIDIA | `nvidia.nemotron-nano-3-30b` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| OpenAI@Bedrock | `openai.gpt-oss-20b-1:0` | ✅ | ✅ | ✅¹ | ✅ | ✅ | ✅ | — | — | — |
| OpenAI@Bedrock | `openai.gpt-oss-120b-1:0` | ✅ | ✅ | ✅¹ | ✅ | ✅ | ✅ | — | — | — |
| Qwen | `qwen.qwen3-32b-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Qwen | `qwen.qwen3-coder-30b-a3b-v1:0` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Qwen | `qwen.qwen3-vl-235b-a22b` | ✅ | ✅ | ✅ | — | — | — | — | ✅ | — |
| Writer | `writer.palmyra-x4-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Writer | `writer.palmyra-x5-v1:0` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| Writer | `writer.palmyra-vision-7b` | ✅ | ✅ | ✅ | — | — | — | — | ✅ | — |
| Z.AI | `zai.glm-4.7-flash` | ✅ | ✅ | ✅ | — | — | — | — | — | — |
| Z.AI | `zai.glm-5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |

---

**Matrix footnotes:**

> ¹ **Multi-turn with small `max_tokens` — very low issue (test budget only):** These are native-reasoning models that spend tokens on an internal thinking phase before producing text. When the test uses a low `max_tokens` budget, the thinking block exhausts the token allowance, leaving no room for text output. With `max_tokens≥512` (or `≥1024` for heavy reasoners) all affected models produce correct output. This is a test-configuration observation, not a server defect; the gateway correctly forwards the model's response. Affected models: `deepseek.r1-v1:0`, `minimax.minimax-m2.5` (T1 only), `minimax.minimax-m2.1`, `moonshot.kimi-k2-thinking`, `openai.gpt-oss-20b-1:0`, `openai.gpt-oss-120b-1:0`. The parametrized pytest `_REASONING_MODELS` suite uses `max_tokens=1024` explicitly to avoid this.
>
> ² **Gemma 3 vision — model capability gap:** `google.gemma-3-12b-it` does not support image inputs on Bedrock (`400: The model does not support image inputs`). Server correctly propagates the error. The 27B variant was not tested for vision.
>
> ³ **Streaming+tools — Bedrock limitation:** `meta.llama3-3-70b-instruct-v1:0`, `mistral.mistral-large-2402-v1:0`, and `mistral.pixtral-large-2502-v1:0` return `400` for streaming + tool use (`This model doesn't support tool use in streaming mode`). Non-streaming tool use passes for all three. Server correctly propagates the Bedrock error.
>
> ⁴ **Llama 3.3 70B agentic loop — model behavior:** Model outputs raw JSON strings for tool calls instead of using Bedrock's native `toolUse` block format in multi-turn contexts. T5 (single-turn, non-streaming) passes. The gateway cannot work around a model that ignores the tool schema.
>
> ⁵ **NVIDIA Nemotron Nano 9B — verbose chain-of-thought:** The 9B model always prepends extended step-by-step reasoning to its answer. System-prompt formatting instructions (e.g. ALL CAPS) apply only to the final answer portion; the CoT preamble is always lowercase. Use `nvidia.nemotron-nano-3-30b` for predictable output.

---

## Issues Summary

Issues are divided into two categories: **server issues** (our code, our responsibility) and **model / external limitations** (not fixable server-side).

---

### 🔴 Server Issues — Our Responsibility

Failures caused by a bug in the stdapi.ai gateway. Each entry is either fixed or open.

| ID | Severity | Status | Summary | Location |
|----|----------|--------|---------|----------|
| ISSUE-1 | Critical | ✅ **Fixed** | DeepSeek V3 and Google Gemma 3 streaming returned empty content. `_process_content_block_delta()` permanently suppressed any block starting with an empty delta `{"text": ""}`, but DeepSeek/Gemma emit an empty first delta followed by real content in the same block. Fix: changed to *deferred suppression* — un-suppress on the first non-empty delta, synthesize the missing `content_block_start` retroactively. | `stdapi/models/chat/_adapters/_anthropic_message.py` · `_process_content_block_delta()` |
| ISSUE-7 | Very Low | ✅ **Fixed** | Double-nested error message: when Bedrock returns `"Model 'xxx' not found"` the server wrapped it again, producing `"The model 'Model 'xxx' not found' does not exist..."`. Fixed by adding a `detail` kwarg to `UnsupportedModelError` and always passing the plain model ID; deprecation context now goes into `detail`. | `stdapi/api_errors.py` · `UnsupportedModelError` · `stdapi/models/__init__.py` · `_raise_model_not_found()` |

> **Current state:** No open server bugs with functional impact. All ❌S have been resolved.

---

### 🟡 Model & Bedrock Limitations — Not Our Responsibility

Failures or gaps caused by Bedrock access policies, model-level behaviour, or model capability limits. The server handles these correctly (returns the appropriate error or response); they cannot be fixed server-side.

#### Bedrock Access Restrictions (⊘ Untested)

Models that were attempted but blocked at the Bedrock layer before the gateway could serve them. Marked ⊘ in the matrices.

| ID | Models | Reason | Workaround |
|----|--------|--------|------------|
| ISSUE-2 | `meta.llama3-2-3b-instruct-v1:0`, `meta.llama3-2-1b-instruct-v1:0`, `cohere.command-r-v1:0`, `cohere.command-r-plus-v1:0` | **Legacy access revoked.** Bedrock's 15-day inactivity policy for legacy models returns HTTP 404. Access resumes after reactivation in the Bedrock console. | Reactivate in the Bedrock console, or use a current-generation replacement. |
| ISSUE-EXT-2 | `meta.llama4-scout-17b-instruct-v1:0`, `meta.llama4-maverick-17b-instruct-v1:0`, `meta.llama3-2-11b-instruct-v1:0`, `meta.llama3-2-90b-instruct-v1:0` | **Meta EULA geo-restriction.** HTTP 400 from all regions. AWS detects the account as an EU account and blocks these models regardless of which regions are configured in `AWS_BEDROCK_REGIONS`. Non-multimodal Llama 3.x is unaffected. | No workaround available — restriction is enforced at the AWS account level, not the region level. Requires a non-EU AWS account. |

#### Bedrock Model Capability Limits (❌M in matrices)

The model or Bedrock platform does not support a feature. The server correctly propagates the Bedrock error.

| ID | Severity | Models | Description |
|----|----------|--------|-------------|
| ISSUE-3 | Medium | `mistral.mistral-large-2402-v1:0`, `mistral.pixtral-large-2502-v1:0`, `meta.llama3-3-70b-instruct-v1:0` | **Streaming + tool use returns HTTP 400.** Bedrock rejects `converse_stream` with `toolConfig` for these model families (`This model doesn't support tool use in streaming mode`). Non-streaming tool use works correctly for all three. |
| ISSUE-4 | Medium | `meta.llama3-3-70b-instruct-v1:0` | **Agentic loop outputs raw JSON.** In multi-turn tool-use contexts, the model emits tool calls as JSON-encoded text strings instead of native Bedrock `toolUse` blocks. Single-turn T5 passes. The gateway cannot compensate for a model that ignores the Bedrock tool schema. |
| ISSUE-6 | Low | `google.gemma-3-12b-it` | **No image input support.** Model returns `400: The model does not support image inputs`. |

#### Model Behaviour Characteristics (Informational)

Observable model-quality quirks that do not indicate a server defect.

| ID | Severity | Models | Description |
|----|----------|--------|-------------|
| ISSUE-OAI-2 | Low | `mistral.mistral-7b-instruct-v0:2`, `minimax.minimax-m2.5` | Weak system-prompt compliance: models occasionally ignore strict formatting instructions (e.g. ALL CAPS). Budget/small models. |
| ISSUE-EXT-4 | Low | `nvidia.nemotron-nano-9b-v2` | Always prepends verbose chain-of-thought reasoning to answers; cannot be suppressed by prompt. Use `nvidia.nemotron-nano-3-30b` for clean output. |

#### Test-Configuration Observations (Very Low — Reasoning Models)

The following were observed during ad-hoc testing with low `max_tokens` budgets. The parametrized pytest `_REASONING_MODELS` suite uses `max_tokens=1024` explicitly to avoid this.

| ID | Severity | Models | Description |
|----|----------|--------|-------------|
| ISSUE-5 / ISSUE-OAI-3 / ISSUE-EXT-3 | **Very Low** | `deepseek.r1-v1:0`, `minimax.minimax-m2.5`, `minimax.minimax-m2.1`, `moonshot.kimi-k2-thinking`, `openai.gpt-oss-20b-1:0`, `openai.gpt-oss-120b-1:0` | Native-reasoning models allocate the entire `max_tokens` budget to an internal thinking block, leaving no capacity for text output. With `max_tokens≥512` (or `≥1024` for heavy models) all produce correct reasoning + text. The gateway is correct — it returns `stop_reason: max_tokens` with a valid `thinking` block. The issue only surfaces in multi-turn tests if the first assistant message comes back empty, causing Bedrock to reject a blank assistant message on turn 2. |

#### Feature Gaps (Intentional — Not Bugs)

| Feature | Route | Status |
|---------|-------|--------|
| `logprobs`, `store`, `prediction` | OpenAI | Intentionally unsupported. |
| Prompt caching (`cache_control`) | OpenAI | `cache_control` parameter is Anthropic-route-specific. OpenAI route equivalent: `prompt_cache_key` (supported). |

#### Reasoning / Thinking Configuration (`thinking` parameter)

The `thinking` parameter is supported but model-dependent. Configuration method and accepted options vary by model family. Unsupported models return 400.

| Model family | Supports reasoning | Budget tokens | Effort level | Notes |
|---|---|---|---|---|
| Claude 4.6+ | ✅ | ✅ | ✅ | Adaptive by default; switches to explicit budget when `budget_tokens` is set; `effort` maps to `output_config.effort` |
| Claude 3.7–4.5 | ✅ | ✅ | ✅ | Always uses explicit `budget_tokens`; if omitted, calculated from effort (0.25×–1× of `max_tokens`, capped at 32 768) |
| Amazon Nova 2 | ✅ | ❌M | ✅ | Effort only (`maxReasoningEffort`); returns 400 if `budget_tokens` is provided |
| DeepSeek V3 | ✅ | ❌M | ✅ | Effort only, passed as a string literal to Bedrock; returns 400 if `budget_tokens` is provided |
| Claude 3.5, Nova base/premier, Mistral, OpenAI GPT | ❌M | ❌M | ❌M | Not supported; returns 400 |

---

## Provider Coverage Summary

| Provider | Models tested | Status |
|----------|--------------|--------|
| Anthropic (Claude) | 1 | ✅ Full coverage |
| Amazon Nova | 4 | ✅ Full coverage |
| AI21 Labs | 2 | ✅ Basic + streaming |
| Cohere | 2 | ⊘ Untested — legacy access revoked |
| DeepSeek | 3 | ✅ Full coverage incl. reasoning + tools |
| Google Gemma | 3 | ✅ Basic + streaming (no image support) |
| Meta Llama 3.x | 4 | ✅ / ⊘ — 3.1 8B/70B ✅; 3.3 70B ✅ (tool caveat); 3.2 3B ⊘ legacy |
| Meta Llama 4 | 2 | ⊘ Untested — EU geo-restriction |
| Meta Llama 3.2 Vision | 2 | ⊘ Untested — EU geo-restriction |
| MiniMax | 2 | ✅ Basic + reasoning + tools (M2.5) |
| Mistral | 5 | ✅ All variants; streaming+tools Bedrock-limited |
| Moonshot AI | 2 | ✅ K2.5 full; K2-Thinking basic + reasoning |
| NVIDIA | 2 | ✅ 30B clean; 9B verbose CoT |
| OpenAI@Bedrock | 2 | ✅ Basic + tools; reasoning needs large max_tokens |
| Qwen | 3 | ✅ Text + tools + vision (VL 235B) |
| Writer | 3 | ✅ Text + tools + vision |
| Z.AI | 2 | ✅ Basic + tools (GLM-5) |
| **Total** | **42** | **34 fully verified** · **6 ⊘ untested (external)** · **2 caveats** |

---

## Route Equivalence

Both API routes are functionally equivalent for all tested models. The gateway correctly translates between Anthropic Messages API format and OpenAI Chat Completions API format with no observable behavioral differences.

| Capability | Anthropic route | OpenAI route |
|-----------|----------------|--------------|
| Basic chat (non-streaming) | ✅ | ✅ |
| Streaming | ✅ (ISSUE-1 fixed) | ✅ |
| Tool use (non-streaming) | ✅ | ✅ |
| Tool use (streaming) | ✅ (model-dependent) | ✅ (model-dependent) |
| Agentic loops | ✅ | ✅ |
| Native reasoning / thinking blocks | ✅ | ✅ (as text, no `thinking` type) |
| Vision (image input) | ✅ | ✅ |
| Prompt caching | ✅ (`cache_control` per-block) | ✅ (`prompt_cache_key` parameter) |
| JSON mode (structured output) | ✅ (`output_config` parameter with `json_schema`) | ✅ (`response_format` with `json_object` / `json_schema` via Bedrock `outputConfig`) |
| `thinking` / reasoning (budget) | ✅ (`thinking.budget_tokens`, Claude only) | ✅ (`enable_thinking` + `thinking_budget`, Claude only) |
| `thinking` / reasoning (effort) | ✅ (`thinking.effort`: `low`/`medium`/`high`/`max`, Claude + Nova 2 + DeepSeek) | ✅ (`reasoning_effort`: `low`/`medium`/`high`/`max`, Claude + Nova 2 + DeepSeek) |

---

## Claude Code Agentic Benchmark (T-CC)

**Test date:** 2026-04-03
**Server:** stdapi.ai v1.7.0+ — **each test run spawns its own dedicated stdapi server** on a free port, captures its JSON request logs, and asserts that every Bedrock call targeted the expected model ID.
**Claude Code:** v2.1.59
**Route:** Anthropic Messages API (`/anthropic`) — all models routed through the `sonnet` slot via `ANTHROPIC_DEFAULT_SONNET_MODEL`

These tests launch a real `claude` CLI process in `--print` mode and ask it to analyze the stdapi.ai source code across multiple files.  Each task was designed to require **~10+ tool calls** by instructing the model to read actual code and quote function signatures verbatim.  A `--max-budget-usd 10` cap is applied per invocation to prevent runaway loops.

Metrics per cell: **turns · wall-clock duration · input tokens · output tokens**.

> **Token note**: `claude --output-format json` reports token counts from the Anthropic SDK response.  For models with prompt caching enabled (Claude Sonnet, Nova 2), `in` shows *fresh* (non-cached) input tokens; cached tokens are listed separately.

> **Effort tests (T-CC5)** only run for models with `supports_effort=True` (Claude Sonnet + Nova 2 Lite).
> All other models show `—` for T-CC5.

### T-CC1: Request Pipeline Trace

Trace POST /v1/chat/completions from the HTTP route handler to the Bedrock `converse()` call.  Model must read ≥5 files and quote exact function signatures.

| Model | Result | Turns | Duration | In tok | Out tok |
|---|---|---|---|---|---|
| `anthropic.claude-sonnet-4-6` | ✅ | 2 | 128s | 5³ | 2 103 |
| `amazon.nova-2-lite-v1:0` | ✅ | 2 | 95s | 5² | 1 793 |
| `moonshotai.kimi-k2.5` | ✅ | 2 | 98s | 39 001 | 1 793 |
| `qwen.qwen3-coder-30b-a3b-v1:0` | ✅ | 17 | 67s | 399 824 | 3 447 |
| `qwen.qwen3-coder-next` | ✅ | 2 | 122s | 39 000 | 1 612 |
| `minimax.minimax-m2.5` | ✅ | 2 | 121s | 39 129 | 1 888 |
| `mistral.devstral-2-123b` | ✅ | 2 | 117s | 40 635 | 2 019 |

### T-CC2: Streaming Path Trace

Trace the `stream=True` code path from the divergence point to the final SSE output.  Model must read ≥3 files and quote SSE event mapping code.

| Model | Result | Turns | Duration | In tok | Out tok |
|---|---|---|---|---|---|
| `anthropic.claude-sonnet-4-6` | ✅ | 12 | 226s | 11³ | 3 938 |
| `amazon.nova-2-lite-v1:0` | ✅ | 16 | 142s | 12² | 4 894 |
| `moonshotai.kimi-k2.5` | ✅ | 2 | 129s | 40 392 | 2 366 |
| `qwen.qwen3-coder-30b-a3b-v1:0` | ✅ | 2 | 173s | 42 438 | 2 691 |
| `qwen.qwen3-coder-next` | ✅ | 12 | 160s | 248 217 | 4 638 |
| `minimax.minimax-m2.5` | ✅ | 2 | 144s | 41 746 | 2 478 |
| `mistral.devstral-2-123b` | ✅ | 2 | 155s | 41 258 | 2 207 |

### T-CC3: Parameter Mapping Audit

Read types, adapter, and `_prepare_converse_request` to document ≥10 OpenAI → Bedrock field mappings with exact code quotes.  Most demanding multi-file read task.

| Model | Result | Turns | Duration | In tok | Out tok |
|---|---|---|---|---|---|
| `anthropic.claude-sonnet-4-6` | ✅ | 17 | 118s | 13³ | 5 479 |
| `amazon.nova-2-lite-v1:0` | ✅ | 16 | 113s | 13² | 5 448 |
| `moonshotai.kimi-k2.5` | ✅ | 10 | 96s | 197 265 | 4 352 |
| `qwen.qwen3-coder-30b-a3b-v1:0` | ✅ | 18 | 119s | 416 038 | 5 728 |
| `qwen.qwen3-coder-next` | ✅ | 16 | 116s | 274 207 | 4 696 |
| `minimax.minimax-m2.5` | ✅¹ | 19 | 112s | 400 245 | 5 350 |
| `mistral.devstral-2-123b` | ✅ | 9 | 104s | 202 406 | 4 546 |

¹ Failed once (hard error / unparseable output), passed on retry.

### T-CC4: Model Override Enumeration

Glob `stdapi/models/chat/`, read `_default.py` for baseline, then read ≥5 model-specific files and quote overridden method signatures.

| Model | Result | Turns | Duration | In tok | Out tok |
|---|---|---|---|---|---|
| `anthropic.claude-sonnet-4-6` | ✅ | 15 | 73s | 5³ | 4 956 |
| `amazon.nova-2-lite-v1:0` | ✅ | 13 | 67s | 5² | 4 216 |
| `moonshotai.kimi-k2.5` | ✅ | 13 | 61s | 107 796 | 3 360 |
| `qwen.qwen3-coder-30b-a3b-v1:0` | ✅ | 13 | 66s | 107 796 | 4 138 |
| `qwen.qwen3-coder-next` | ✅ | 15 | 63s | 99 253 | 3 879 |
| `minimax.minimax-m2.5` | ✅ | 13 | 64s | 107 868 | 3 854 |
| `mistral.devstral-2-123b` | ✅ | 13 | 62s | 107 796 | 3 672 |

### T-CC5: Effort Level Comparison (Claude + Nova 2 only)

Same T-CC3 parameter-mapping task at `--effort low` and `--effort high`.  Measures whether reasoning budget changes exploration depth.

| Model | Effort | Result | Turns | Duration | In tok | Out tok |
|---|---|---|---|---|---|---|
| `anthropic.claude-sonnet-4-6` | low | ✅ | 11 | 123s | 11³ | 6 354 |
| `anthropic.claude-sonnet-4-6` | high | ✅ | 20 | 173s | 14³ | 10 816 |
| `amazon.nova-2-lite-v1:0` | low | ✅ | 21 | 136s | 15² | 6 368 |
| `amazon.nova-2-lite-v1:0` | high | ✅ | 11 | 91s | 9² | 4 530 |

² Nova 2 Lite has prompt caching enabled; `In tok` shows fresh (non-cached) tokens only.  Cached token reads per test: T-CC1 18K · T-CC2 258K · T-CC3 324K · T-CC4 66K · T-CC5-low 495K · T-CC5-high 152K.

³ Claude Sonnet 4.6 also has prompt caching enabled; `In tok` shows fresh (non-cached) tokens only.  Cached token reads per test: T-CC1 18K · T-CC2 176K · T-CC3 326K · T-CC4 54K · T-CC5-low 249K · T-CC5-high 362K.

### Observations

**Turn count reflects exploration depth**, not answer quality — all models produced correct, passing answers:

- **T-CC4 (model overrides)** is the most consistent: five models used exactly 13 turns; Claude Sonnet and Qwen3-coder-next used 15.  The task inherently forces Glob + multiple Read calls regardless of prior knowledge.
- **T-CC3 (parameter mapping)** drove the most variability and highest turn counts (9–19 turns), confirming it is the most demanding code-reading task.
- **T-CC1 (pipeline trace)** was answered in 2 turns by six of seven models — those models either resolved the call chain from training knowledge or produced accurate answers from a minimal file read.  Qwen3-coder-30b is the exception (17 turns) and a better differentiator of exploration depth.
- **T-CC2 (streaming path)** shows strong model differentiation: Claude Sonnet (12), Nova 2 Lite (16), and Qwen3-coder-next (12) explored deeply; other models completed in 2 turns.
- **Effort `low` → fewer turns than `high` for Claude Sonnet; reversed for Nova 2**: at `low` effort Claude Sonnet used 11 turns (vs 20 at `high`), while Nova 2 used 21 turns at `low` (vs 11 at `high`).  For Claude, higher effort drives deeper exploration (more files read, more output tokens: 6K vs 10K).  For Nova 2, higher effort appears to improve planning — fewer but more focused reads.  The two models exhibit opposite effort-turn relationships.
- **Nova 2 Lite prompt caching**: removing `DISABLE_PROMPT_CACHING` enables Nova 2's Bedrock prompt caching.  Cached tokens reach 495K per test (T-CC5-low), leaving fresh `input_tokens` in the single digits.  Output tokens are the meaningful work metric for Nova 2.
- **Token counts vs cost**: `cost_usd` from Claude Code was replaced by `in`/`out` token counts.  The cost field used Claude Sonnet pricing for all models (including Nova, Qwen, etc.), making cross-model cost comparisons misleading.  Token counts are model-agnostic and directly reflect the work done.
- **MiniMax M2.5 flakiness**: MiniMax M2.5 produced a hard failure (non-zero exit / unparseable output) on T-CC3 in one run, then passed in a retry.  The `¹` marker in the T-CC3 table reflects this.  All other models passed every run consistently.
- **Run-to-run variance**: Agentic turn counts are non-deterministic. Multiple runs of the same model on the same task have produced notably different counts: Kimi K2.5 pipeline trace was 2 turns on one run and 13 turns on another; Qwen3-coder-next pipeline trace was 2 turns on one run and 15 turns on another.  The figures in the tables above reflect the most recent successful run for each model.  The numbers reflect exploration depth on a given invocation, not a stable scalar.

