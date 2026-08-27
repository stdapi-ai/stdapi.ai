---
title: Models
description: Browse, filter, chart and compare every model stdapi.ai serves on AWS - modalities, regional availability, published AWS prices and independent leaderboard scores, side by side.
keywords: AWS AI model list, Amazon Bedrock model list, Bedrock model comparison, Bedrock pricing table, AI model catalogue AWS, compare LLM prices AWS, Bedrock model availability by region
hide:
  - navigation
  - toc
---

# :material-view-list: Models

Every model stdapi.ai serves on AWS — Amazon Bedrock, Bedrock Mantle, Polly,
Transcribe and Comprehend — what it does, where it runs,
what AWS charges for it, and how it scores on public leaderboards.

All of them answer on **one endpoint under one API key**, through the
[OpenAI, Anthropic and Cohere APIs](features.md) your tools already speak.

This list is what a deployment serves out of the box. A deployment can also
serve models of its own: any [Amazon Bedrock Marketplace model
endpoint](features.md#bedrock-marketplace-endpoints) or [Amazon SageMaker AI
endpoint](features.md#sagemaker-endpoints) you run in your account joins the
same catalogue, on the same APIs.

<link rel="stylesheet" href="../styles/models-table.min.css">
<link rel="preload" href="catalog.json" as="fetch" crossorigin>

<div class="models-app" data-models-app hidden markdown="0">
</div>

<script src="../js/models-table.min.js" defer></script>

<noscript>
<p><strong>The interactive table needs JavaScript.</strong> The same catalogue is
available from any running instance through
<a href="../api_search_models/">search_models</a> and
<a href="../api_model_pricing/">model_pricing</a>, and as raw JSON at
<a href="catalog.json">models/catalog.json</a>.</p>
<!-- catalog:noscript -->
<table><thead><tr><th>Model</th><th>ID</th><th>Provider</th><th>Input</th><th>Output</th><th>Regions</th></tr></thead><tbody>
<tr><td>Jamba 1.5 Large (legacy)</td><td><code>ai21.jamba-1-5-large-v1:0</code></td><td>AI21 Labs</td><td>TEXT</td><td>TEXT</td><td>1</td></tr>
<tr><td>Jamba 1.5 Mini (legacy)</td><td><code>ai21.jamba-1-5-mini-v1:0</code></td><td>AI21 Labs</td><td>TEXT</td><td>TEXT</td><td>1</td></tr>
<tr><td>Bedrock Guardrail Checks</td><td><code>amazon.bedrock-runtime-guardrail-checks</code></td><td>Amazon</td><td>TEXT</td><td>MODERATION</td><td>7</td></tr>
<tr><td>Comprehend Toxicity Detection</td><td><code>amazon.comprehend-toxicity</code></td><td>Amazon</td><td>TEXT</td><td>MODERATION</td><td>33</td></tr>
<tr><td>Nova 2 Lite</td><td><code>amazon.nova-2-lite-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE, VIDEO</td><td>TEXT</td><td>19</td></tr>
<tr><td>Amazon Nova Multimodal Embeddings</td><td><code>amazon.nova-2-multimodal-embeddings-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE, AUDIO, VIDEO</td><td>EMBEDDING</td><td>1</td></tr>
<tr><td>Nova 2 Sonic</td><td><code>amazon.nova-2-sonic-v1:0</code></td><td>Amazon</td><td>SPEECH</td><td>SPEECH, TEXT</td><td>4</td></tr>
<tr><td>Nova Canvas (legacy)</td><td><code>amazon.nova-canvas-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Nova Lite</td><td><code>amazon.nova-lite-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE, VIDEO</td><td>TEXT</td><td>19</td></tr>
<tr><td>Nova Micro</td><td><code>amazon.nova-micro-v1:0</code></td><td>Amazon</td><td>TEXT</td><td>TEXT</td><td>16</td></tr>
<tr><td>Nova Premier (legacy)</td><td><code>amazon.nova-premier-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE, VIDEO</td><td>TEXT</td><td>3</td></tr>
<tr><td>Nova Pro</td><td><code>amazon.nova-pro-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE, VIDEO</td><td>TEXT</td><td>18</td></tr>
<tr><td>Nova Reel (legacy)</td><td><code>amazon.nova-reel-v1:0</code></td><td>Amazon</td><td>TEXT, IMAGE</td><td>VIDEO</td><td>3</td></tr>
<tr><td>Nova Reel (legacy)</td><td><code>amazon.nova-reel-v1:1</code></td><td>Amazon</td><td>TEXT, IMAGE</td><td>VIDEO</td><td>1</td></tr>
<tr><td>Nova Sonic (legacy)</td><td><code>amazon.nova-sonic-v1:0</code></td><td>Amazon</td><td>SPEECH</td><td>SPEECH, TEXT</td><td>3</td></tr>
<tr><td>Polly Generative</td><td><code>amazon.polly-generative</code></td><td>Amazon</td><td>TEXT</td><td>SPEECH</td><td>10</td></tr>
<tr><td>Polly Long-form</td><td><code>amazon.polly-long-form</code></td><td>Amazon</td><td>TEXT</td><td>SPEECH</td><td>1</td></tr>
<tr><td>Polly Neural</td><td><code>amazon.polly-neural</code></td><td>Amazon</td><td>TEXT</td><td>SPEECH</td><td>18</td></tr>
<tr><td>Polly Standard</td><td><code>amazon.polly-standard</code></td><td>Amazon</td><td>TEXT</td><td>SPEECH</td><td>20</td></tr>
<tr><td>Rerank 1.0</td><td><code>amazon.rerank-v1:0</code></td><td>Amazon</td><td>TEXT</td><td>RERANKING</td><td>4</td></tr>
<tr><td>Titan Multimodal Embeddings G1</td><td><code>amazon.titan-embed-image-v1</code></td><td>Amazon</td><td>TEXT, IMAGE</td><td>EMBEDDING</td><td>10</td></tr>
<tr><td>Titan Embeddings G1 - Text</td><td><code>amazon.titan-embed-text-v1</code></td><td>Amazon</td><td>TEXT</td><td>EMBEDDING</td><td>4</td></tr>
<tr><td>Titan Text Embeddings V2</td><td><code>amazon.titan-embed-text-v2:0</code></td><td>Amazon</td><td>TEXT</td><td>EMBEDDING</td><td>19</td></tr>
<tr><td>Transcribe</td><td><code>amazon.transcribe</code></td><td>Amazon</td><td>SPEECH</td><td>TEXT</td><td>1</td></tr>
<tr><td>Claude 3.5 Sonnet</td><td><code>anthropic.claude-3-5-sonnet-20240620-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>Claude 3.5 Sonnet v2</td><td><code>anthropic.claude-3-5-sonnet-20241022-v2:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>Claude 3.7 Sonnet</td><td><code>anthropic.claude-3-7-sonnet-20250219-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>2</td></tr>
<tr><td>Claude 3 Haiku (legacy)</td><td><code>anthropic.claude-3-haiku-20240307-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>15</td></tr>
<tr><td>Claude 3 Sonnet</td><td><code>anthropic.claude-3-sonnet-20240229-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>5</td></tr>
<tr><td>Claude Fable 5</td><td><code>anthropic.claude-fable-5</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>claude-haiku-4-5</td><td><code>anthropic.claude-haiku-4-5</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>7</td></tr>
<tr><td>Claude Haiku 4.5</td><td><code>anthropic.claude-haiku-4-5-20251001-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Opus 4.1 (legacy)</td><td><code>anthropic.claude-opus-4-1-20250805-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>Claude Opus 4.5</td><td><code>anthropic.claude-opus-4-5-20251101-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Opus 4.6</td><td><code>anthropic.claude-opus-4-6-v1</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Opus 4.7</td><td><code>anthropic.claude-opus-4-7</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Opus 4.8</td><td><code>anthropic.claude-opus-4-8</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Opus 5</td><td><code>anthropic.claude-opus-5</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Sonnet 4 (legacy)</td><td><code>anthropic.claude-sonnet-4-20250514-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>19</td></tr>
<tr><td>Claude Sonnet 4.5</td><td><code>anthropic.claude-sonnet-4-5-20250929-v1:0</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Sonnet 4.6</td><td><code>anthropic.claude-sonnet-4-6</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Claude Sonnet 5</td><td><code>anthropic.claude-sonnet-5</code></td><td>Anthropic</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>Embed English</td><td><code>cohere.embed-english-v3</code></td><td>Cohere</td><td>TEXT</td><td>EMBEDDING</td><td>12</td></tr>
<tr><td>Embed Multilingual</td><td><code>cohere.embed-multilingual-v3</code></td><td>Cohere</td><td>TEXT</td><td>EMBEDDING</td><td>12</td></tr>
<tr><td>Embed v4</td><td><code>cohere.embed-v4:0</code></td><td>Cohere</td><td>TEXT, IMAGE</td><td>EMBEDDING</td><td>23</td></tr>
<tr><td>Rerank 3.5</td><td><code>cohere.rerank-v3-5:0</code></td><td>Cohere</td><td>TEXT</td><td>RERANKING</td><td>5</td></tr>
<tr><td>DeepSeek-R1</td><td><code>deepseek.r1-v1:0</code></td><td>DeepSeek</td><td>TEXT</td><td>TEXT</td><td>3</td></tr>
<tr><td>DeepSeek-V3.1</td><td><code>deepseek.v3-v1:0</code></td><td>DeepSeek</td><td>TEXT</td><td>TEXT</td><td>8</td></tr>
<tr><td>v3.1</td><td><code>deepseek.v3.1</code></td><td>DeepSeek</td><td>TEXT</td><td>TEXT</td><td>12</td></tr>
<tr><td>DeepSeek V3.2</td><td><code>deepseek.v3.2</code></td><td>DeepSeek</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>Gemma 3 12B IT</td><td><code>google.gemma-3-12b-it</code></td><td>Google</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Gemma 3 27B PT</td><td><code>google.gemma-3-27b-it</code></td><td>Google</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Gemma 3 4B IT</td><td><code>google.gemma-3-4b-it</code></td><td>Google</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>gemma-4-26b-a4b</td><td><code>google.gemma-4-26b-a4b</code></td><td>Google</td><td>TEXT, IMAGE</td><td>TEXT</td><td>4</td></tr>
<tr><td>gemma-4-31b</td><td><code>google.gemma-4-31b</code></td><td>Google</td><td>TEXT, IMAGE</td><td>TEXT</td><td>4</td></tr>
<tr><td>gemma-4-e2b</td><td><code>google.gemma-4-e2b</code></td><td>Google</td><td>TEXT, IMAGE</td><td>TEXT</td><td>4</td></tr>
<tr><td>Ray v2</td><td><code>luma.ray-v2:0</code></td><td>Luma AI</td><td>TEXT</td><td>VIDEO</td><td>1</td></tr>
<tr><td>Llama 3.1 70B Instruct</td><td><code>meta.llama3-1-70b-instruct-v1:0</code></td><td>Meta</td><td>TEXT</td><td>TEXT</td><td>3</td></tr>
<tr><td>Llama 3.1 8B Instruct</td><td><code>meta.llama3-1-8b-instruct-v1:0</code></td><td>Meta</td><td>TEXT</td><td>TEXT</td><td>3</td></tr>
<tr><td>Llama 3.3 70B Instruct</td><td><code>meta.llama3-3-70b-instruct-v1:0</code></td><td>Meta</td><td>TEXT</td><td>TEXT</td><td>3</td></tr>
<tr><td>Llama 3 70B Instruct</td><td><code>meta.llama3-70b-instruct-v1:0</code></td><td>Meta</td><td>TEXT</td><td>TEXT</td><td>5</td></tr>
<tr><td>Llama 3 8B Instruct</td><td><code>meta.llama3-8b-instruct-v1:0</code></td><td>Meta</td><td>TEXT</td><td>TEXT</td><td>5</td></tr>
<tr><td>Llama 4 Maverick 17B Instruct</td><td><code>meta.llama4-maverick-17b-instruct-v1:0</code></td><td>Meta</td><td>TEXT, IMAGE</td><td>TEXT</td><td>4</td></tr>
<tr><td>Llama 4 Scout 17B Instruct</td><td><code>meta.llama4-scout-17b-instruct-v1:0</code></td><td>Meta</td><td>TEXT, IMAGE</td><td>TEXT</td><td>4</td></tr>
<tr><td>MiniMax M2</td><td><code>minimax.minimax-m2</code></td><td>MiniMax</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>MiniMax M2.1</td><td><code>minimax.minimax-m2.1</code></td><td>MiniMax</td><td>TEXT</td><td>TEXT</td><td>13</td></tr>
<tr><td>MiniMax M2.5</td><td><code>minimax.minimax-m2.5</code></td><td>MiniMax</td><td>TEXT</td><td>TEXT</td><td>13</td></tr>
<tr><td>Devstral 2 123B</td><td><code>mistral.devstral-2-123b</code></td><td>Mistral AI</td><td>TEXT</td><td>TEXT</td><td>13</td></tr>
<tr><td>Magistral Small 2509</td><td><code>mistral.magistral-small-2509</code></td><td>Mistral AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Ministral 14B 3.0</td><td><code>mistral.ministral-3-14b-instruct</code></td><td>Mistral AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Ministral 3B</td><td><code>mistral.ministral-3-3b-instruct</code></td><td>Mistral AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Ministral 3 8B</td><td><code>mistral.ministral-3-8b-instruct</code></td><td>Mistral AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Mistral 7B Instruct</td><td><code>mistral.mistral-7b-instruct-v0:2</code></td><td>Mistral AI</td><td>TEXT</td><td>TEXT</td><td>9</td></tr>
<tr><td>Mistral Large (24.02)</td><td><code>mistral.mistral-large-2402-v1:0</code></td><td>Mistral AI</td><td>TEXT</td><td>TEXT</td><td>9</td></tr>
<tr><td>Mistral Large (24.07)</td><td><code>mistral.mistral-large-2407-v1:0</code></td><td>Mistral AI</td><td>TEXT</td><td>TEXT</td><td>1</td></tr>
<tr><td>Mistral Large 3</td><td><code>mistral.mistral-large-3-675b-instruct</code></td><td>Mistral AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>7</td></tr>
<tr><td>Mistral Small (24.02)</td><td><code>mistral.mistral-small-2402-v1:0</code></td><td>Mistral AI</td><td>TEXT</td><td>TEXT</td><td>1</td></tr>
<tr><td>Mixtral 8x7B Instruct</td><td><code>mistral.mixtral-8x7b-instruct-v0:1</code></td><td>Mistral AI</td><td>TEXT</td><td>TEXT</td><td>9</td></tr>
<tr><td>Pixtral Large (25.02)</td><td><code>mistral.pixtral-large-2502-v1:0</code></td><td>Mistral AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>7</td></tr>
<tr><td>Voxtral Mini 3B 2507</td><td><code>mistral.voxtral-mini-3b-2507</code></td><td>Mistral AI</td><td>SPEECH, TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>Voxtral Small 24B 2507</td><td><code>mistral.voxtral-small-24b-2507</code></td><td>Mistral AI</td><td>SPEECH, TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>Kimi K2 Thinking</td><td><code>moonshot.kimi-k2-thinking</code></td><td>Moonshot AI</td><td>TEXT</td><td>TEXT</td><td>7</td></tr>
<tr><td>kimi-k2-thinking</td><td><code>moonshotai.kimi-k2-thinking</code></td><td>Moonshot AI</td><td>TEXT</td><td>TEXT</td><td>12</td></tr>
<tr><td>Kimi K2.5</td><td><code>moonshotai.kimi-k2.5</code></td><td>Moonshot AI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>NVIDIA Nemotron Nano 12B v2 VL BF16</td><td><code>nvidia.nemotron-nano-12b-v2</code></td><td>NVIDIA</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>Nemotron Nano 3 30B</td><td><code>nvidia.nemotron-nano-3-30b</code></td><td>NVIDIA</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>NVIDIA Nemotron Nano 9B v2</td><td><code>nvidia.nemotron-nano-9b-v2</code></td><td>NVIDIA</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>NVIDIA Nemotron 3 Super 120B A12B</td><td><code>nvidia.nemotron-super-3-120b</code></td><td>NVIDIA</td><td>TEXT</td><td>TEXT</td><td>13</td></tr>
<tr><td>gpt-5.4</td><td><code>openai.gpt-5.4</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>gpt-5.4-2026-03-05</td><td><code>openai.gpt-5.4-2026-03-05</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>gpt-5.5</td><td><code>openai.gpt-5.5</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>2</td></tr>
<tr><td>gpt-5.5-2026-04-23</td><td><code>openai.gpt-5.5-2026-04-23</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>2</td></tr>
<tr><td>GPT-5.6 Luna</td><td><code>openai.gpt-5.6-luna</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>GPT-5.6 Sol</td><td><code>openai.gpt-5.6-sol</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>GPT-5.6 Terra</td><td><code>openai.gpt-5.6-terra</code></td><td>OpenAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>gpt-oss-120b</td><td><code>openai.gpt-oss-120b</code></td><td>OpenAI</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>gpt-oss-20b</td><td><code>openai.gpt-oss-20b</code></td><td>OpenAI</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>GPT OSS Safeguard 120B</td><td><code>openai.gpt-oss-safeguard-120b</code></td><td>OpenAI</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>GPT OSS Safeguard 20B</td><td><code>openai.gpt-oss-safeguard-20b</code></td><td>OpenAI</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>Qwen3 235B A22B 2507</td><td><code>qwen.qwen3-235b-a22b-2507</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>Qwen3 32B (dense)</td><td><code>qwen.qwen3-32b</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>qwen3-coder-30b-a3b-instruct</td><td><code>qwen.qwen3-coder-30b-a3b-instruct</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>Qwen3-Coder-30B-A3B-Instruct</td><td><code>qwen.qwen3-coder-30b-a3b-v1:0</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>13</td></tr>
<tr><td>qwen3-coder-480b-a35b-instruct</td><td><code>qwen.qwen3-coder-480b-a35b-instruct</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>12</td></tr>
<tr><td>Qwen3 Coder 480B A35B Instruct</td><td><code>qwen.qwen3-coder-480b-a35b-v1:0</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>8</td></tr>
<tr><td>Qwen3 Coder Next</td><td><code>qwen.qwen3-coder-next</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>3</td></tr>
<tr><td>Qwen3 Next 80B A3B</td><td><code>qwen.qwen3-next-80b-a3b</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>qwen3-next-80b-a3b-instruct</td><td><code>qwen.qwen3-next-80b-a3b-instruct</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>Qwen3 VL 235B A22B</td><td><code>qwen.qwen3-vl-235b-a22b</code></td><td>Qwen</td><td>TEXT, IMAGE</td><td>TEXT</td><td>10</td></tr>
<tr><td>qwen3-vl-235b-a22b-instruct</td><td><code>qwen.qwen3-vl-235b-a22b-instruct</code></td><td>Qwen</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>Stable Diffusion 3.5 Large</td><td><code>stability.sd3-5-large-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>1</td></tr>
<tr><td>Stable Image Conservative Upscale</td><td><code>stability.stable-conservative-upscale-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Creative Upscale</td><td><code>stability.stable-creative-upscale-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Fast Upscale</td><td><code>stability.stable-fast-upscale-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Control Sketch</td><td><code>stability.stable-image-control-sketch-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Control Structure</td><td><code>stability.stable-image-control-structure-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Core 1.0</td><td><code>stability.stable-image-core-v1:1</code></td><td>Stability AI</td><td>TEXT</td><td>IMAGE</td><td>1</td></tr>
<tr><td>Stable Image Erase Object</td><td><code>stability.stable-image-erase-object-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Inpaint</td><td><code>stability.stable-image-inpaint-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Remove Background</td><td><code>stability.stable-image-remove-background-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Search and Recolor</td><td><code>stability.stable-image-search-recolor-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Search and Replace</td><td><code>stability.stable-image-search-replace-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Style Guide</td><td><code>stability.stable-image-style-guide-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Ultra 1.0</td><td><code>stability.stable-image-ultra-v1:1</code></td><td>Stability AI</td><td>TEXT</td><td>IMAGE</td><td>1</td></tr>
<tr><td>Stable Image Outpaint</td><td><code>stability.stable-outpaint-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Stable Image Style Transfer</td><td><code>stability.stable-style-transfer-v1:0</code></td><td>Stability AI</td><td>TEXT, IMAGE</td><td>IMAGE</td><td>3</td></tr>
<tr><td>Marengo Embed v2.7 (legacy)</td><td><code>twelvelabs.marengo-embed-2-7-v1:0</code></td><td>TwelveLabs</td><td>TEXT, IMAGE, SPEECH, VIDEO</td><td>EMBEDDING</td><td>3</td></tr>
<tr><td>Marengo Embed 3.0</td><td><code>twelvelabs.marengo-embed-3-0-v1:0</code></td><td>TwelveLabs</td><td>TEXT, IMAGE, SPEECH, VIDEO</td><td>EMBEDDING</td><td>3</td></tr>
<tr><td>Pegasus v1.2</td><td><code>twelvelabs.pegasus-1-2-v1:0</code></td><td>TwelveLabs</td><td>TEXT, VIDEO</td><td>TEXT</td><td>23</td></tr>
<tr><td>Writer Palmyra Vision 7B</td><td><code>writer.palmyra-vision-7b</code></td><td>Writer</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>Palmyra X4</td><td><code>writer.palmyra-x4-v1:0</code></td><td>Writer</td><td>TEXT</td><td>TEXT</td><td>4</td></tr>
<tr><td>Palmyra X5</td><td><code>writer.palmyra-x5-v1:0</code></td><td>Writer</td><td>TEXT</td><td>TEXT</td><td>4</td></tr>
<tr><td>grok-4.3</td><td><code>xai.grok-4.3</code></td><td>xAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>3</td></tr>
<tr><td>Grok 4.6</td><td><code>xai.grok-4.6</code></td><td>xAI</td><td>TEXT, IMAGE</td><td>TEXT</td><td>23</td></tr>
<tr><td>glm-4.6</td><td><code>zai.glm-4.6</code></td><td>Zhipu AI</td><td>TEXT</td><td>TEXT</td><td>15</td></tr>
<tr><td>GLM 4.7</td><td><code>zai.glm-4.7</code></td><td>Z.AI</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
<tr><td>GLM 4.7 Flash</td><td><code>zai.glm-4.7-flash</code></td><td>Z.AI</td><td>TEXT</td><td>TEXT</td><td>13</td></tr>
<tr><td>GLM 5</td><td><code>zai.glm-5</code></td><td>Z.AI</td><td>TEXT</td><td>TEXT</td><td>10</td></tr>
</tbody></table>
<!-- /catalog:noscript -->
</noscript>

<div class="models-howto" markdown>

:material-map-marker-radius: **“Available in EU” has two answers.** *Where I can
call it* is the AWS regions you can reach the model from. *Where it runs* is
where inference executes — a model reachable from Frankfurt through a **global**
inference profile is not a model running in Frankfurt. For sovereignty, read the
second.

:material-cash: **The prices are AWS's.** Published AWS rates in USD per billed
unit. You pay AWS directly on your own account; stdapi.ai adds no margin and
sends no invoice. The authority is the
[Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/).

:material-check-decagram: **A blank score means blank.** A benchmark published
against the wrong model is worse than none, so an entry is attached only when
the evidence is unambiguous. Open a model to see which leaderboard entry each
score came from.

:material-history: **Legacy models stay listed.** Models AWS has marked legacy
are tagged and kept, so one you already run stays findable. Availability
reflects the AWS account this snapshot came from — your own
[`search_models`](api_search_models.md) is the authority for what *you* can call.

</div>

---

<div class="models-sources" markdown>

## :material-database-outline: Sources, licences and caveats

<!-- catalog:generated -->
Snapshot taken on **2026-08-26** from a stdapi.ai instance, covering 138 models across 33 AWS regions. Prices and availability move — before you commit to a number, confirm it against the [Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) and your own [`search_models`](api_search_models.md).
<!-- /catalog:generated -->

<!-- catalog:sources -->
Every number on this page comes from one of the sources below, reproduced unmodified. Mapping each entry onto an Amazon Bedrock model ID is our own work, and any error in that mapping is ours, not the source's.

| Source | Licence | Read on | Used here |
| --- | --- | --- | --- |
| The gateway's own [`search_models`](api_search_models.md) and [`model_pricing`](api_model_pricing.md) | — | 2026-08-26 | 138 |
| [Amazon Bedrock `ListFoundationModels`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html), read raw so its undocumented fields survive | — | 2026-08-26 | capabilities, APIs, media types, limits |
| [LMArena Leaderboard](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 2026-08-24 | 79 of 793 entries |
| [MTEB — Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/results) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 2026-08-26 | 5 of 12 entries |
| [Epoch AI — AI Benchmarking Hub](https://epoch.ai/benchmarks/use-this-data) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 2026-08-18 | 59 of 655 entries |
| [Amazon Bedrock model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) | [AWS documentation](https://aws.amazon.com/terms/) | 2026-08-26 | 112 of 124 model cards |
| [models.dev](https://models.dev/) | [MIT](https://github.com/anomalyco/models.dev/blob/dev/LICENSE) | 2026-08-26 | 65 of 120 models |
| [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 2026-08-26 | 2 of 65 entries |

- **LMArena Leaderboard** — Arena Elo ratings by LMArena (Arena Intelligence Inc.), reproduced unmodified under CC BY 4.0. The mapping to Amazon Bedrock model IDs is ours.
- **MTEB — Massive Text Embedding Benchmark** — Benchmark results from the MTEB results repository, dedicated to the public domain under CC0 1.0. The mapping to Amazon Bedrock model IDs is ours.
- **Epoch AI — AI Benchmarking Hub** — Benchmark results by Epoch AI, reproduced unmodified under CC BY 4.0. Rows derived from the Aider Polyglot and Terminal-Bench leaderboards keep their Apache-2.0 licence. The mapping to Amazon Bedrock model IDs is ours.
- **Amazon Bedrock model cards** — Context windows, output limits, knowledge cutoffs and lifecycle dates as stated on each model's Amazon Bedrock model card. Only those facts are taken; AWS's own descriptive copy is not reproduced.
- **models.dev** — Context windows, knowledge cutoffs and capability flags from models.dev, an open database of AI models, used under the MIT licence. Its Amazon Bedrock entries are keyed by Bedrock model ID, so the join is exact.
- **Open ASR Leaderboard** — Word error rates from the Open ASR Leaderboard, reproduced unmodified under Apache-2.0. The mapping to Amazon Bedrock model IDs is ours.

Some of what the table shows comes from parts of `ListFoundationModels` AWS does not document, so AWS may stop returning them at any time. A regeneration updates the published data set rather than replacing it: the last known value of such a field is kept, and a model AWS stops listing stays in the table, tagged `delisted`, with the date it was last seen.
<!-- /catalog:sources -->

<!-- catalog:providers -->
The model names and brand logos in the table above are trademarks of their respective owners: AI21 Labs, Jamba; AWS, Amazon and Amazon product names and logos; Alibaba Cloud, Qwen; Anthropic, Claude; Cohere; DeepSeek; Google, Gemini, Gemma; Luma AI; Meta, Llama; MiniMax; Mistral AI; Moonshot AI, Kimi; NVIDIA, Nemotron; OpenAI, ChatGPT, GPT, Codex; Stability AI, Stable Diffusion; TwelveLabs, Marengo, Pegasus; Writer, Palmyra; Z.ai, Zhipu AI, GLM; xAI, Grok.
<!-- /catalog:providers -->

</div>
