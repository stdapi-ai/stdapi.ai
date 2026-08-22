/*
 * Models page: renders the committed catalogue as a sortable, filterable table,
 * a scatter chart, and a side-by-side comparison of up to three models.
 *
 * Everything it reads is a static artefact under docs/models/, written by
 * `python -m docs_gen.model_catalog`. Nothing is fetched from a third party at
 * render time, and nothing here needs a build step.
 */

(function () {
  "use strict";

  /* Number of models that can be compared at once. */
  var COMPARE_LIMIT = 3;

  /*
   * Providers highlighted in the chart at once. Three is not a layout choice:
   * it is the largest set of categorical hues that clears colour-vision and
   * contrast separation on every pair, which is what a scatter needs — every
   * dot can sit next to every other. Past three, the rest stay neutral.
   */
  var HIGHLIGHT_LIMIT = 3;

  /* Geography buckets offered as one-click filters, in display order. Each
     names the region its prices fall back to, which is the region the
     compliance page's own recipe for that geography starts from. */
  var GEO_BUCKETS = [
    { key: "americas", label: "US", hint: "AWS regions in the Americas", region: "us-east-1" },
    { key: "europe", label: "EU", hint: "AWS regions in Europe", region: "eu-west-1" },
    {
      key: "asia_pacific",
      label: "Asia",
      hint: "AWS regions in Asia Pacific",
      region: "ap-northeast-1",
    },
  ];

  /* Not a place — an inference-profile type — so it is its own toggle, not a
     member of the geography chip group. */
  var GLOBAL_BUCKET = {
    key: "global",
    label: "Global routing",
    hint: "Served through a global cross-region inference profile",
  };

  /* Serving markers that name a geography rather than a single region. */
  var SERVING_BUCKETS = {
    global: "global",
    us: "americas",
    eu: "europe",
    ap: "asia_pacific",
    apac: "asia_pacific",
    ca: "americas",
  };

  /*
   * Where inference may run. This is a compliance decision before it is a
   * price one — a geography-scoped profile never leaves its geography, while
   * a global profile may route anywhere AWS serves the model — and the two are
   * different products at different prices, so the choice moves the number in
   * the Price column.
   */
  var ROUTING_CHOICES = [
    {
      value: "any",
      label: "as this gateway routes it",
      kinds: null,
      hint: "The product this gateway would actually bill you for from the selected region.",
    },
    {
      value: "region",
      label: "in the calling region only",
      kinds: ["region", ""],
      hint: "No cross-region inference: the request is served by the region you send it to.",
    },
    {
      value: "geography",
      label: "within the geography",
      kinds: ["region", "geography", ""],
      hint: "In-region, or a geography-scoped profile whose destinations never leave that geography. Chosen for you when you pick a geography.",
    },
    {
      value: "global",
      label: "anywhere, including a global profile",
      kinds: ["region", "geography", "global", ""],
      prefer: "global",
      hint: "No residency constraint: a global profile may run the request in any region AWS serves it from, and is usually the cheapest.",
    },
  ];

  /* Preference when the reader has not restricted routing: the product a call
     from that region actually reaches, cheapest last. */
  var ROUTING_ORDER = ["region", "geography", "", "global"];

  /* Billed dimensions shown as price columns, with their display label. */
  var PRICE_COLUMNS = [
    {
      key: "input_tokens", label: "Input $/1M", unit: "/1M in", scale: 1e6,
      dimensionHelp: "Tokens in the prompt and any context you send.",
    },
    {
      key: "output_tokens", label: "Output $/1M", unit: "/1M out", scale: 1e6,
      dimensionHelp: "Tokens the model generates in its response — usually the larger share of the bill.",
    },
    {
      key: "cache_read_tokens", label: "Cache read $/1M", unit: "/1M cached", scale: 1e6,
      dimensionHelp: "A cached prompt prefix reused from an earlier request, at a lower rate than a fresh input token.",
    },
    {
      key: "cache_write_tokens", label: "Cache write $/1M", unit: "/1M cache write", scale: 1e6,
      dimensionHelp: "Writing a new prompt prefix into the cache so a later request can reuse it.",
    },
    {
      key: "output_images", label: "$/image", unit: "/image", scale: 1,
      dimensionHelp: "Each image the model generates.",
    },
    {
      key: "input_seconds", label: "$/input second", unit: "/s in", scale: 1,
      dimensionHelp: "Seconds of audio or video you send the model.",
    },
    {
      key: "output_seconds", label: "$/output second", unit: "/s out", scale: 1,
      dimensionHelp: "Seconds of audio the model generates — Polly and other speech models.",
    },
    {
      key: "input_characters", label: "$/1M characters", unit: "/1M chars", scale: 1e6,
      dimensionHelp: "Text you send the model, billed per character rather than per token.",
    },
    {
      key: "input_images", label: "$/input image", unit: "/image in", scale: 1,
      dimensionHelp: "Each image you send as input.",
    },
    {
      key: "search_units", label: "$/1K search units", unit: "/1K searches", scale: 1e3,
      dimensionHelp: "Each batch of up to 100 documents a rerank request scores.",
    },
    {
      key: "comprehend_units", label: "$/1K units", unit: "/1K units", scale: 1e3,
      dimensionHelp: "Amazon Comprehend's own billing unit for the text analysed, not a token.",
    },
    {
      key: "text_units", label: "$/1K text units", unit: "/1K units", scale: 1e3,
      dimensionHelp: "Amazon Bedrock Guardrails' own billing unit — 1,000 characters of text checked.",
    },
    {
      key: "grounding_requests", label: "$/1K checks", unit: "/1K checks", scale: 1e3,
      dimensionHelp: "Each call the model's built-in grounding tool makes to a source.",
    },
  ];

  /* The same scales, for the per-unit rates in the model card. */
  var PRICE_SCALES = PRICE_COLUMNS.reduce(function (all, price) {
    all[price.key] = price.scale;
    return all;
  }, {});

  /* Minimal hand-drawn icons, so the page adds no icon font or sprite. */
  var ICONS = {
    search: '<circle cx="10" cy="10" r="6"/><path d="M14.5 14.5 L20 20"/>',
    close: '<path d="M6 6 L18 18 M18 6 L6 18"/>',
    table: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M9 9v11"/>',
    chart:
      '<path d="M4 4v16h16"/><circle cx="9" cy="14" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="18" cy="13" r="1.8"/>',
    columns: '<path d="M5 5v14M12 5v14M19 5v14"/>',
    compare: '<rect x="3" y="5" width="7" height="14" rx="1"/><rect x="14" y="5" width="7" height="14" rx="1"/>',
    reset: '<path d="M4 11a8 8 0 1 1 2.3 5.7"/><path d="M4 5v6h6"/>',
    best: '<path d="M4 13l5 5L20 7"/>',
    worst: '<path d="M5 12h14"/>',
    link: '<path d="M14 4h6v6"/><path d="M20 4L10 14"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    region: '<circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4a14 14 0 0 1 0 16a14 14 0 0 1 0-16"/>',
  };

  /* One glyph per modality, so a row can be scanned without reading it. */
  var MODALITY_ICONS = {
    TEXT: '<path d="M5 6h14M5 12h14M5 18h9"/>',
    IMAGE:
      '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="10" r="1.5"/><path d="M4 17l5-5 4 4 3-3 4 4"/>',
    VIDEO: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/>',
    AUDIO: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>',
    SPEECH:
      '<path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M17 8a5 5 0 0 1 0 8"/>',
    EMBEDDING:
      '<circle cx="6" cy="7" r="2"/><circle cx="18" cy="10" r="2"/><circle cx="9" cy="18" r="2"/><path d="M7.7 8.5l8.6 0.9M7.6 8.9l1.1 7.2"/>',
    RERANKING: '<path d="M4 7h13M4 12h9M4 17h5"/><path d="M19 6v12M16 15l3 3 3-3"/>',
    MODERATION: '<path d="M12 3l8 3v6c0 4.4-3.2 7.6-8 9-4.8-1.4-8-4.6-8-9V6l8-3z"/>',
  };

  /* One glyph per geography, so the region columns read at a glance. */
  var GEO_ICONS = {
    global: '<circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4a14 14 0 0 1 0 16a14 14 0 0 1 0-16"/>',
    region: '<path d="M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/>',
  };

  /* Which modality a route or an MCP tool name belongs to, longest match first. */
  var ENDPOINT_MODALITIES = [
    ["audio/speech", "SPEECH"],
    ["audio_speech", "SPEECH"],
    ["audio/transcription", "AUDIO"],
    ["audio/translation", "AUDIO"],
    ["audio_transcription", "AUDIO"],
    ["audio_translation", "AUDIO"],
    ["realtime", "SPEECH"],
    ["image", "IMAGE"],
    ["video", "VIDEO"],
    ["embed", "EMBEDDING"],
    ["rerank", "RERANKING"],
    ["moderation", "MODERATION"],
    ["chat", "TEXT"],
    ["completion", "TEXT"],
    ["message", "TEXT"],
    ["response", "TEXT"],
    ["conversation", "TEXT"],
  ];

  function modalityOf(value) {
    var text = String(value).toLowerCase();
    for (var i = 0; i < ENDPOINT_MODALITIES.length; i += 1) {
      if (text.indexOf(ENDPOINT_MODALITIES[i][0]) !== -1) {
        return ENDPOINT_MODALITIES[i][1];
      }
    }
    return null;
  }

  function endpointTag(value) {
    var modality = modalityOf(value);
    var tag = el("span", { class: "models-tag", title: modality ? modality + " endpoint" : value });
    if (modality && MODALITY_ICONS[modality]) {
      tag.appendChild(glyph(MODALITY_ICONS[modality]));
    }
    tag.appendChild(el("code", { class: "models-endpoint", text: value }));
    return tag;
  }

  /* Which section of the column chooser a column belongs to, in display order. */
  var COLUMN_GROUPS = ["Identity", "Capability", "Availability", "Price", "Benchmarks", "How to call", "Lifecycle"];

  var COLUMN_GROUP = {
    name: "Identity",
    provider: "Identity",
    service: "Identity",
    family: "Identity",
    aliases: "Identity",
    input_modalities: "Capability",
    output_modalities: "Capability",
    context_window: "Capability",
    knowledge_cutoff: "Capability",
    reasoning: "Capability",
    tool_call: "Capability",
    open_weights: "Identity",
    licence: "Identity",
    parameters: "Identity",
    active_parameters: "Identity",
    max_output_tokens: "Capability",
    streaming: "Capability",
    prompt_caching: "Capability",
    guardrails: "Capability",
    batch: "Capability",
    batch_in_region: "Capability",
    batch_cross_region: "Capability",
    latency_optimized: "Capability",
    provisioned: "Capability",
    count_tokens: "Capability",
    prompt_routing: "Capability",
    customizations: "Capability",
    image_types: "Capability",
    document_types: "Capability",
    video_types: "Capability",
    regions: "Availability",
    serving: "Availability",
    buckets: "Availability",
    inference_types: "Availability",
    routes: "How to call",
    mcp_tools: "How to call",
    apis: "How to call",
    legacy: "Lifecycle",
    retired: "Lifecycle",
    start_of_life: "Lifecycle",
    end_of_life: "Lifecycle",
    first_seen: "Lifecycle",
    last_seen: "Lifecycle",
  };

  /* What each column means, shown on its heading and in the column chooser. */
  var COLUMN_HELP = {
    name: "The value you pass as `model` — the Amazon Bedrock model ID, or the gateway's own ID for a Polly, Transcribe or Comprehend model. Click to open the model.",
    provider: "The vendor that built the model.",
    service: "The AWS service that serves it: Bedrock Runtime, Bedrock Mantle, Polly, Transcribe or Comprehend. All of them answer on the same host under the same API key, so which one a model comes from changes nothing about how you call it.",
    input_modalities: "What you can send it.",
    output_modalities: "What it produces.",
    context_window: "How much input the model accepts in one request.",
    knowledge_cutoff: "The date the model's training data ends.",
    reasoning: "Whether the model can be asked to reason explicitly before answering. Ask for it the way your own SDK does and the gateway passes it on in the form this model expects.",
    tool_call: "Whether the model can call tools. Describe them in OpenAI or Anthropic form and the gateway takes care of the rest.",
    open_weights: "Whether the model's weights are publicly released.",
    licence: "The licence the weights are released under. Open weights under a research-only licence are not the same proposition as Apache-2.0.",
    parameters: "Total parameters, for models that publish a count. More is not automatically better, but it is the rough measure of capacity.",
    active_parameters: "Parameters actually used per token. For a mixture-of-experts model this is what drives cost and latency; the total does not.",
    max_output_tokens: "The largest response the model is documented to produce, as stated on its AWS model card or by its vendor.",
    regions: "How many AWS regions it can be called from, counting only those a selected geography and routing leave open. The gateway fails over between your regions and their quotas add up, so a wider spread means more headroom, not just more choice. Hover a value for the list; the card and the comparison show it in full.",
    serving: "Where inference can actually run: a global or geography-scoped cross-region inference profile, or a single region. Pair it with the routing control to see what that choice costs.",
    buckets: "Which geographies it can be called from.",
    streaming: "Whether it can stream its response token by token, in whichever API you called. The gateway places the request in a region with capacity for it, then streams the answer straight back.",
    prompt_caching: "Whether Amazon Bedrock prompt caching applies, which cuts the cost of a repeated prompt prefix. Your API's own caching field is enough — the gateway places the cache points and reports the cached tokens back to you.",
    guardrails: "Whether Amazon Bedrock Guardrails can be applied through the model's own API. A guardrail set once on the gateway also screens the routes that cannot carry one themselves — embeddings, rerank, image, video, speech, transcription and moderation.",
    batch: "Whether it can run as a batch job, at roughly half the on-demand token price. Send it to the standard OpenAI or Anthropic batch endpoint and the gateway runs it, reporting the usage at the batch rate.",
    batch_in_region: "Whether AWS supports this model for a batch job run in a single region. The gateway chooses which region that is.",
    batch_cross_region: "Whether a batch job can be spread across a geography's regions, which raises the throughput a single region's quota allows.",
    count_tokens: "Whether you can ask what a prompt will cost in tokens before sending it. The count covers the request exactly as the gateway will send it.",
    prompt_routing: "Whether Amazon Bedrock intelligent prompt routing can send easy prompts to a cheaper model in the same family.",
    image_types: "Image formats it accepts.",
    document_types: "Document formats it accepts.",
    video_types: "Video formats it accepts.",
    latency_optimized: "Whether AWS publishes a latency-optimised variant.",
    provisioned: "Whether AWS lets provisioned throughput be reserved for it. Where your account has throughput reserved, the model stays listed and callable by name.",
    legacy: "Whether AWS has marked the model deprecated. The gateway leaves these out of the list it offers, and where a replacement is known it reroutes a request naming one rather than failing.",
    retired: "Whether AWS has stopped listing the model. The row is kept so a retired model stays findable, but its facts are the last ones seen.",
    first_seen: "When this model first appeared in a snapshot of this page.",
    last_seen: "When a snapshot last saw this model on AWS.",
    family: "The family AWS groups it under.",
    apis: "The Amazon Bedrock APIs it answers underneath; you never name one, the gateway picks it from the model. Blank means AWS publishes none for it.",
    inference_types: "How it can be invoked: on demand, through an inference profile, or on provisioned throughput.",
    customizations: "Whether it can be fine-tuned or distilled.",
    routes: "The API routes this gateway accepts it on. Every text model answers on all four text APIs under one model ID — a Nova model on Anthropic's messages route, a Claude model on OpenAI's /v1/responses — so changing model never means changing SDK.",
    mcp_tools: "The MCP tool names an agent can call it through, so an agent reaches this model by name with no HTTP client code.",
    aliases: "Other names the `model` parameter accepts for it. Name the plain model and the gateway picks the cross-region form that is valid where your call is served, never one that would leave its geography.",
    start_of_life: "When AWS made it generally available.",
    end_of_life: "When AWS will stop serving it.",
  };

  /* What each kind of value means, appended to the matching column's heading. */
  var PRICE_HELP =
    "Published AWS rate for the selected region, tier and routing. AWS bills it "
    + "directly at 0% markup: the gateway never resells model usage and adds no "
    + "margin of its own.";
  var SCORE_HELP =
    "Independent public leaderboard result, reproduced unmodified. Blank means no entry could be matched to this model with confidence — it is not a zero.";

  /* What each leaderboard actually measures, keyed by "source:board" — the
     same key scoreColumns() already builds per column. */
  var SCORE_BOARD_HELP = {
    "lmarena:text": "LMArena's human-voted head-to-head arena for general chat.",
    "lmarena:vision": "LMArena's human-voted head-to-head arena for vision and multimodal chat.",
    "lmarena:search": "LMArena's human-voted head-to-head arena for search-grounded chat.",
    "lmarena:text_to_image": "LMArena's human-voted head-to-head arena for text-to-image generation.",
    "epoch:gpqa_diamond": "Epoch AI's run of GPQA Diamond, graduate-level multiple-choice science questions.",
    "epoch:frontiermath": "Epoch AI's run of FrontierMath, unpublished research-level mathematics problems.",
    "epoch:math_level_5": "Epoch AI's run of the hardest tier of the MATH benchmark.",
    "epoch:aider_polyglot_external": "Epoch AI's run of Aider's Polyglot benchmark: real code-editing tasks across languages.",
    "epoch:swe_bench_verified": "Epoch AI's run of SWE-bench Verified: real GitHub issues the model has to fix.",
    "open_asr:english_short": "Hugging Face's Open ASR Leaderboard, English short-form speech.",
  };

  /*
   * The first screen of an alphabetical list of 142 models answers nobody's
   * question. Each of these applies a filter and a sort in one click; they are
   * shortcuts into the same table, not a separate ranking.
   */
  var STARTING_POINTS = [
    {
      key: "chat",
      label: "General chat",
      hint: "Text in, text out, ranked by the text arena",
      apply: function () {
        state.modalities.output_modalities.add("TEXT");
        state.sort = { key: "score:lmarena:text", direction: "descending" };
      },
    },
    {
      key: "value",
      label: "Cheapest capable",
      hint: "Text models with a published benchmark result, cheapest first",
      apply: function () {
        state.modalities.output_modalities.add("TEXT");
        state.onlyScored = true;
        state.sort = { key: "price:primary", direction: "ascending" };
      },
    },
    {
      key: "open",
      label: "Open weights",
      hint: "Models whose weights are published",
      apply: function () {
        state.filters.open_weights = "yes";
        state.sort = { key: "score:lmarena:text", direction: "descending" };
      },
    },
    {
      key: "tools",
      label: "Agents & tools",
      hint: "Models that can call tools",
      apply: function () {
        state.filters.tool_call = "yes";
        state.sort = { key: "score:lmarena:text", direction: "descending" };
      },
    },
    {
      key: "eu",
      label: "Runs in the EU",
      hint: "Inference executes in an EU region",
      apply: function () {
        state.sense = "runs";
        state.buckets = new Set(["europe"]);
      },
    },
    {
      key: "speech",
      label: "Speech output",
      hint: "Models that produce spoken audio",
      apply: function () {
        state.modalities.output_modalities.add("SPEECH");
      },
    },
    {
      key: "embedding",
      label: "Embeddings",
      hint: "Vector embeddings for search and retrieval",
      apply: function () {
        state.modalities.output_modalities.add("EMBEDDING");
      },
    },
  ];

  var app = null;
  var state = null;

  /* -- small helpers ----------------------------------------------------- */

  /*
   * The stacked layout below 38em sets display:block on the table elements,
   * and an engine is entitled to drop their implicit roles when it does. The
   * roles are therefore stated, so the table stays a table to a screen reader
   * at every width. Restated rather than relied on: nothing here is on the
   * wire, since the table is built from the catalogue at runtime.
   */
  var TABLE_ROLES = {
    table: "table",
    thead: "rowgroup",
    tbody: "rowgroup",
    tr: "row",
    td: "cell",
  };

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (TABLE_ROLES[tag]) {
      node.setAttribute("role", TABLE_ROLES[tag]);
    } else if (tag === "th") {
      node.setAttribute(
        "role",
        attrs && attrs.scope === "row" ? "rowheader" : "columnheader"
      );
    }
    apply(node, attrs);
    (children || []).forEach(function (child) {
      if (child) {
        node.appendChild(child);
      }
    });
    return node;
  }

  function svg(tag, attrs, children) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    apply(node, attrs);
    (children || []).forEach(function (child) {
      if (child) {
        node.appendChild(child);
      }
    });
    return node;
  }

  function apply(node, attrs) {
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") {
        node.textContent = attrs[key];
      } else if (key === "html") {
        node.innerHTML = attrs[key];
      } else if (attrs[key] !== null && attrs[key] !== undefined && attrs[key] !== false) {
        node.setAttribute(key, attrs[key]);
      }
    });
  }

  function icon(name, extra) {
    return svg(
      "svg",
      {
        class: "models-icon" + (extra ? " " + extra : ""),
        viewBox: "0 0 24 24",
        "aria-hidden": "true",
        focusable: "false",
        html: ICONS[name],
      },
      []
    );
  }

  function glyph(paths, extra) {
    return svg(
      "svg",
      {
        class: "models-glyph" + (extra ? " " + extra : ""),
        viewBox: "0 0 24 24",
        "aria-hidden": "true",
        focusable: "false",
        html: paths,
      },
      []
    );
  }

  /* The country an AWS region sits in. */
  function countryOf(region) {
    return (state.manifest.region_countries || {})[region] || null;
  }

  /* A regional-indicator pair, which renders as that country's flag. */
  function flagFor(region) {
    var code = countryOf(region);
    if (!code) {
      return null;
    }
    return String.fromCodePoint.apply(
      String,
      code.split("").map(function (letter) {
        return 0x1f1e6 + letter.charCodeAt(0) - 65;
      })
    );
  }

  /* Only a real http(s) URL is ever written to an href — a bare label or a
     javascript: URI from the source data renders as text instead. */
  function safeHref(url) {
    return typeof url === "string" && /^https?:/i.test(url) ? url : null;
  }

  /* "200K" and "1M" are numbers a reader can compare; the code has to as well. */
  function parseTokens(value) {
    if (typeof value === "number") {
      return value || null;
    }
    if (typeof value !== "string" || !value) {
      return null;
    }
    var text = value.trim().replace(/,/g, "");
    var scale = 1;
    if (/k$/i.test(text)) {
      scale = 1000;
      text = text.slice(0, -1);
    } else if (/m$/i.test(text)) {
      scale = 1e6;
      text = text.slice(0, -1);
    }
    var number = parseFloat(text);
    return isNaN(number) ? null : number * scale;
  }

  function unique(values) {
    return Array.from(new Set(values)).sort();
  }

  function routingChoice() {
    for (var i = 0; i < ROUTING_CHOICES.length; i += 1) {
      if (ROUTING_CHOICES[i].value === state.routing) {
        return ROUTING_CHOICES[i];
      }
    }
    return ROUTING_CHOICES[0];
  }

  /*
   * Choosing a geography is a residency decision, so the two controls that
   * decide what a price means follow it: routing stops at that geography, and
   * the price is quoted from a region inside it. Either one the reader sets
   * themselves is theirs — this only moves what they have left alone.
   */
  function followGeography() {
    var geographies = GEO_BUCKETS.filter(function (bucket) {
      return state.buckets.has(bucket.key);
    });
    /* Global routing is a profile type, not a place: asked for alongside a
       geography, it is the reader saying they accept leaving it. */
    var residency = geographies.length > 0 && !state.buckets.has(GLOBAL_BUCKET.key);
    if (!state.pinned.routing) {
      state.routing = residency ? "geography" : "any";
    }
    if (!geographies.length) {
      /* Dropping the geography has to undo what selecting it did, or the
         prices stay somewhere the reader never asked for. */
      if (!state.pinned.region) {
        state.region = state.manifest.reference_region;
      }
    } else if (!state.buckets.has(state.manifest.region_buckets[state.region])) {
      /* Even a region the reader chose has to move: a geography stops
         offering it, so leaving it selected would quote a rate the control
         no longer lists. Their choice is gone rather than remembered, so the
         control goes back to following the geography. */
      state.region = geographyRegion(geographies);
      state.pinned.region = false;
    }
  }

  /* The regions a price can be quoted from: a geography rules out the ones
     outside it, since their rate is not one its caller could be charged. */
  function offeredRegions() {
    var inside = state.manifest.regions.filter(function (name) {
      return state.buckets.has(state.manifest.region_buckets[name]);
    });
    return inside.length ? inside : state.manifest.regions;
  }

  /* The geography's own region when the catalogue serves it, else any of its
     regions, so the price shown is one the chosen geography can charge. */
  function geographyRegion(geographies) {
    for (var i = 0; i < geographies.length; i += 1) {
      var bucket = geographies[i];
      if (state.regionIndex[bucket.region] !== undefined) {
        return bucket.region;
      }
      var inside = state.manifest.regions.filter(function (name) {
        return state.manifest.region_buckets[name] === bucket.key;
      });
      if (inside.length) {
        return inside[0];
      }
    }
    return state.region;
  }

  /*
   * The one price group the reader is being quoted: the region they call from,
   * the routing they are willing to use, and — for a model served through two
   * AWS services at different rates — the cheaper of the two.
   */
  function groupFor(model, region) {
    var index = state.regionIndex[region];
    if (index === undefined) {
      return null;
    }
    var choice = routingChoice();
    var allowed = choice.kinds;
    /* The gateway routes some models globally whatever region you call from,
       and bills the global rate for them. Quoting the in-region rate there
       would show a price the caller is never charged. */
    var order =
      choice.value === "any" && model.default_routing === "global"
        ? ["global"].concat(ROUTING_ORDER)
        : ROUTING_ORDER;
    var here = model.price_groups.filter(function (group) {
      return (
        group.regions.indexOf(index) !== -1
        && (!allowed || allowed.indexOf(group.routing || "") !== -1)
      );
    });
    if (!here.length) {
      return null;
    }
    var best = null;
    here.forEach(function (group) {
      if (!best) {
        best = group;
        return;
      }
      var wanted = choice.prefer;
      if (wanted) {
        if ((group.routing === wanted) !== (best.routing === wanted)) {
          best = group.routing === wanted ? group : best;
          return;
        }
      }
      var rank = order.indexOf(group.routing || "");
      var winning = order.indexOf(best.routing || "");
      if (rank !== winning) {
        /* Without a restriction, quote the product the call would really take;
           with one, every survivor is acceptable and the order still decides. */
        best = rank < winning ? group : best;
      } else if (blended(group) < blended(best)) {
        // Same product on two services: nobody pays the dearer one on purpose.
        best = group;
      }
    });
    return best;
  }

  /* A single comparable number for a group, for choosing between two services. */
  function blended(group) {
    var input = parseFloat(group.prices.input_tokens || "0");
    var output = parseFloat(group.prices.output_tokens || "0");
    return input * 3 + output;
  }

  /*
   * With the cheapest tier selected, a dimension the cheap tier does not sell
   * has no price on that tier — falling back to the standard rate would put a
   * batch input price beside an on-demand cache-read price and call the pair a
   * quote. The cheap tier's map lists every dimension it does sell, so absence
   * from it is the answer, not a gap.
   */
  function priceFor(model, region, dimension) {
    var group = groupFor(model, region);
    if (!group) {
      return null;
    }
    var cheap = state.tier === "cheapest" && group.cheapest
      && Object.keys(group.cheapest).length;
    var raw = cheap ? group.cheapest[dimension] : group.prices[dimension];
    return raw === undefined ? null : parseFloat(raw);
  }

  /* Names the tier that bills less than standard here, or "" when none does. */
  function cheaperTier(model, region) {
    var group = groupFor(model, region);
    if (group) {
      return group.cheapest && Object.keys(group.cheapest).length
        ? group.cheapest_tier
        : "";
    }
    return "";
  }

  function formatPrice(value, scale) {
    if (value === null) {
      return "—";
    }
    var scaled = value * scale;
    if (scaled === 0) {
      return "0";
    }
    if (scaled < 0.01) {
      return scaled.toPrecision(2).replace(/0+$/, "").replace(/\.$/, "");
    }
    return scaled.toFixed(scaled < 1 ? 4 : 2).replace(/0+$/, "").replace(/\.$/, "");
  }

  /* Recomputed only when a filter that narrows it moves: the list is read once
     per row to render and again on every comparison while sorting. */
  var countedCache = { key: null, byModel: {} };

  /*
   * With a geography selected, the honest count is how many of *those* regions
   * serve the model: "23 regions" beside an EU filter answers a question the
   * reader did not ask. A routing restriction narrows it the same way — a
   * region the reader cannot reach that way is not one of theirs.
   */
  function countedRegions(model) {
    var key = state.sense + "|" + state.routing + "|"
      + Array.from(state.buckets).sort().join(",");
    if (countedCache.key !== key) {
      countedCache = { key: key, byModel: {} };
    }
    if (countedCache.byModel[model.id]) {
      return countedCache.byModel[model.id];
    }
    var names = model.regions.map(function (index) {
      return state.manifest.regions[index];
    });
    if (state.buckets.size) {
      names = state.sense === "runs"
        ? runningRegionNames(model)
        : names.filter(function (region) {
          return state.buckets.has(state.manifest.region_buckets[region]);
        });
    }
    var out = names.filter(function (region) {
      return routableIn(model, region);
    });
    countedCache.byModel[model.id] = out;
    return out;
  }

  /*
   * The regions this model's inference actually executes in, inside the
   * selected geographies. A "global" marker, or a marker naming that same
   * geography (a cross-region profile), covers every one of the model's own
   * regions there; a marker naming a single region covers only itself.
   */
  function runningRegionNames(model) {
    var names = model.regions.map(function (index) {
      return state.manifest.regions[index];
    });
    var out = new Set();
    Array.from(state.buckets).forEach(function (bucket) {
      if (bucket === "global") {
        if (model.serving.indexOf("global") !== -1) {
          names.forEach(function (region) {
            out.add(region);
          });
        }
        return;
      }
      var broad = model.serving.indexOf("global") !== -1
        || model.serving.some(function (marker) {
          return SERVING_BUCKETS[marker] === bucket;
        });
      if (broad) {
        names.forEach(function (region) {
          if (state.manifest.region_buckets[region] === bucket) {
            out.add(region);
          }
        });
        return;
      }
      model.serving.forEach(function (marker) {
        if (state.manifest.region_buckets[marker] === bucket) {
          out.add(marker);
        }
      });
    });
    return Array.from(out);
  }

  /* Serving markers inside the selected geographies, or the full list when
     none is selected — narrows "Runs in" the way runningRegionNames()
     narrows "Regions". */
  function countedServing(model) {
    var markers = model.serving || [];
    if (!state.buckets.size) {
      return markers;
    }
    return markers.filter(function (marker) {
      /* A global profile may run the request inside the selected geography as
         readily as outside it, and that it can leave is the fact a reader
         filtering by geography most needs to see. */
      if (marker === "global") {
        return true;
      }
      var bucket = SERVING_BUCKETS[marker] || state.manifest.region_buckets[marker];
      return bucket && state.buckets.has(bucket);
    });
  }

  function runsBuckets(model) {
    var buckets = new Set();
    model.serving.forEach(function (marker) {
      if (SERVING_BUCKETS[marker]) {
        buckets.add(SERVING_BUCKETS[marker]);
      } else if (state.manifest.region_buckets[marker]) {
        buckets.add(state.manifest.region_buckets[marker]);
      }
    });
    return buckets;
  }

  /* The name, with the id appended only when another model shares it — the
     common case stays uncluttered, the rare collision stops being ambiguous. */
  function disambiguatedName(model) {
    return state.duplicateNames.has(model.name)
      ? model.name + " (" + model.id + ")"
      : model.name;
  }

  function logoFor(model) {
    return logoImage(
      model.logo,
      model.logo_backdrop ? "models-logo--on-" + model.logo_backdrop : ""
    );
  }

  function logoImage(stem, extra) {
    if (!stem) {
      return null;
    }
    return el("img", {
      class: "models-logo" + (extra ? " " + extra : ""),
      src: new URL("logo_" + stem + ".svg", state.assetBase).href,
      alt: "",
      width: "18",
      height: "18",
    });
  }

  /* -- columns ----------------------------------------------------------- */

  function buildColumns(catalog) {
    var columns = [
      {
        key: "name",
        label: "Model",
        filter: "text",
        visible: true,
        sticky: true,
        value: function (model) {
          return model.name + " " + model.id + " " + model.aliases.join(" ");
        },
        sort: function (model) {
          return model.name.toLowerCase();
        },
        cell: function (model) {
          var button = el("button", {
            type: "button",
            class: "models-link models-name",
            "data-detail": model.id,
            title: "Open " + model.name,
          });
          var logo = logoFor(model);
          if (logo) {
            button.appendChild(logo);
          }
          button.appendChild(el("span", { text: model.name }));
          var cell = el("th", { scope: "row", class: "models-cell-name" }, [button]);
          if (model.legacy) {
            cell.appendChild(
              el("span", {
                class: "models-badge models-badge--legacy",
                text: "legacy",
                title: "AWS has marked this model deprecated",
              })
            );
          }
          if (model.retired) {
            cell.appendChild(
              el("span", {
                class: "models-badge models-badge--retired",
                text: "delisted",
                title:
                  "AWS no longer lists this model. Kept from the snapshot of " +
                  (model.last_seen || "an earlier run"),
              })
            );
          }
          cell.appendChild(el("span", { class: "models-id", text: model.id }));
          return cell;
        },
        compareCell: function (model) {
          var wrap = el("span", { class: "models-compare-title" });
          var logo = logoFor(model);
          if (logo) {
            wrap.appendChild(logo);
          }
          wrap.appendChild(el("span", { text: disambiguatedName(model) }));
          return wrap;
        },
      },
      textColumn("provider", "Provider", false, function (m) {
        return m.provider;
      }),
      {
        key: "service",
        label: "Service",
        filter: "select",
        visible: false,
        multi: true,
        /* A folded row answers on more than one service, so filtering by
           either of them has to find it. */
        options: servicesOf,
        value: function (model) {
          return servicesOf(model).join(" ");
        },
        sort: function (model) {
          return model.service.toLowerCase();
        },
        cell: function (model) {
          if (!model.variants || !model.variants.length) {
            return el("td", {}, [serviceTag(model)]);
          }
          /* A folded row answers under every one of these, by a different
             `model` value on each. */
          var cell = el("td", { class: "models-tags" });
          var seen = new Set();
          [{ service: model.service, service_logo: model.service_logo }]
            .concat(model.variants)
            .forEach(function (entry) {
              if (!entry.service || seen.has(entry.service)) {
                return;
              }
              seen.add(entry.service);
              cell.appendChild(serviceTag(entry));
            });
          return cell;
        },
      },
      listColumn("input_modalities", "Input", true, modalityTag),
      listColumn("output_modalities", "Output", true, modalityTag),
      {
        key: "context_window",
        label: "Context",
        filter: "select",
        visible: true,
        numeric: true,
        verdict: ["largest", "smallest"],
        value: function (model) {
          return model.context_window || "";
        },
        sort: function (model) {
          return parseTokens(model.context_window);
        },
        cell: function (model) {
          return el("td", { class: "models-num", text: model.context_window || "—" });
        },
      },
      textColumn("knowledge_cutoff", "Knowledge", false, function (m) {
        return m.knowledge_cutoff || "";
      }),
      boolColumn("reasoning", "Reasoning", false),
      boolColumn("tool_call", "Tools", true),
      boolColumn("open_weights", "Open weights", false),
      sizeColumn("parameters", "Parameters"),
      sizeColumn("active_parameters", "Active params"),
      {
        key: "licence",
        label: "Licence",
        filter: "select",
        visible: false,
        numeric: true,
        betterIsLower: true,
        verdict: ["most permissive", "most restrictive"],
        /* Groups spellings of the same licence ("Apache 2.0" and "Apache-2.0")
           under one filter option without touching the value each model stores. */
        filterKey: licenceFilterKey,
        value: function (model) {
          return model.licence || "";
        },
        sort: function (model) {
          return licenceRank(model.licence);
        },
        cell: function (model) {
          return el("td", { text: model.licence || "—" });
        },
      },
      numberColumn("max_output_tokens", "Max output", false, function (m) {
        return m.max_output_tokens;
      }),
      {
        key: "regions",
        label: "Regions",
        filter: "text",
        visible: true,
        numeric: true,
        verdict: ["most", "fewest"],
        /* A count is all the table column has room for; the card and the
           comparison have room for the list, and the list is the answer to
           "where can I actually call this". */
        multi: true,
        decorate: geographyTag,
        value: function (model) {
          return model.regions
            .map(function (index) {
              return state.manifest.regions[index];
            })
            .join(" ");
        },
        options: function (model) {
          return model.regions.map(function (index) {
            return state.manifest.regions[index];
          });
        },
        narrowedOptions: countedRegions,
        sort: function (model) {
          return countedRegions(model).length;
        },
        cell: function (model) {
          return el("td", {
            class: "models-num",
            text: String(countedRegions(model).length),
            title: regionsNote(model),
          });
        },
        detail: function (model) {
          var counted = countedRegions(model);
          if (!counted.length) {
            return el("td", { text: "—", title: regionsNote(model) });
          }
          var cell = el("td", {
            class: "models-tags models-tags--wrap",
            title: regionsNote(model),
          });
          counted.forEach(function (region) {
            cell.appendChild(geographyTag(region));
          });
          return cell;
        },
      },
      collapsedListColumn("serving", "Runs in", false),
      listColumn("buckets", "Callable from", false, geographyTag),
      boolColumn("streaming", "Streaming", false),
      boolColumn("prompt_caching", "Caching", false),
      boolColumn("guardrails", "Guardrails", false),
      boolColumn("batch", "Batch", false),
      boolColumn("batch_in_region", "Batch in region", false),
      boolColumn("batch_cross_region", "Batch cross-region", false),
      boolColumn("latency_optimized", "Latency opt.", false),
      boolColumn("provisioned", "Provisioned", false),
      boolColumn("count_tokens", "Token counting", false),
      boolColumn("prompt_routing", "Prompt routing", false),
      listColumn("image_types", "Image formats", false, plainTag, true),
      listColumn("document_types", "Document formats", false, plainTag, true),
      listColumn("video_types", "Video formats", false, plainTag, true),
      boolColumn("legacy", "Legacy", false),
      boolColumn("retired", "Retired", false),
      textColumn("first_seen", "First seen", false, function (m) {
        return m.first_seen || "";
      }),
      textColumn("last_seen", "Last seen", false, function (m) {
        return m.last_seen || "";
      }),
      textColumn("family", "Family", false, function (m) {
        return m.family || "";
      }),
      listColumn("apis", "Bedrock APIs", false, plainTag, true),
      listColumn("inference_types", "Inference types", false),
      listColumn("customizations", "Customization", false),
      listColumn("routes", "Routes", false, endpointTag, true),
      listColumn("mcp_tools", "MCP tools", false, endpointTag, true),
      listColumn("aliases", "Aliases", false, plainTag, true),
      textColumn("start_of_life", "Released", false, function (m) {
        return m.start_of_life || "";
      }),
      textColumn("end_of_life", "End of life", false, function (m) {
        return m.end_of_life || "";
      }),
    ];

    PRICE_COLUMNS.forEach(function (price) {
      var used = catalog.models.some(function (model) {
        return model.price_groups.some(function (group) {
          return group.prices[price.key] !== undefined;
        });
      });
      if (!used) {
        return;
      }
      columns.push({
        key: "price:" + price.key,
        label: price.label,
        filter: "none",
        visible: false,
        numeric: true,
        plottable: true,
        betterIsLower: true,
        verdict: ["cheapest", "most expensive"],
        help: price.dimensionHelp + " " + PRICE_HELP,
        value: function () {
          return "";
        },
        sort: function (model) {
          return priceFor(model, state.region, price.key);
        },
        raw: function (model) {
          var value = priceFor(model, state.region, price.key);
          return value === null ? null : value * price.scale;
        },
        format: function (value) {
          return value === null ? "—" : formatPrice(value / price.scale, price.scale);
        },
        cell: function (model) {
          var value = priceFor(model, state.region, price.key);
          return el("td", { class: "models-num", text: formatPrice(value, price.scale) });
        },
      });
    });

    columns.splice(indexOfKey(columns, "regions") + 1, 0, primaryPriceColumn());

    scoreColumns(catalog).forEach(function (column) {
      columns.push(column);
    });
    columns = columns.filter(function (column) {
      if (column.filter === "none") {
        return true;
      }
      return catalog.models.some(function (model) {
        var value = model[column.key];
        return value !== null && value !== undefined && value !== "" && value !== false
          ? !(Array.isArray(value) && !value.length)
          : false;
      });
    });
    columns.forEach(function (column) {
      if (column.help) {
        return;
      }
      if (column.key.indexOf("price:") === 0) {
        column.help = PRICE_HELP;
      } else if (column.key.indexOf("score:") === 0) {
        column.help = SCORE_HELP;
      } else {
        column.help = COLUMN_HELP[column.key] || "";
      }
    });
    columns.forEach(function (column) {
      column.group =
        column.key.indexOf("price:") === 0
          ? "Price"
          : column.key.indexOf("score:") === 0
            ? "Benchmarks"
            : COLUMN_GROUP[column.key] || "Capability";
    });
    /* A select column's option set is a pure function of the catalogue, so it
       is built once here rather than by scanning all 142 models on every
       filter-row rebuild. */
    columns.forEach(function (column) {
      if (column.filter === "select") {
        column.filterOptions = filterOptionsFor(column, catalog);
      }
    });
    /* Whether column.value(model) is empty exactly when column.cell(model)'s
       text would be — true for a plain text/size/licence column, false for a
       boolean ("unknown" is never empty text but renders as "—") or anything
       whose value() is a stub for a raw()/format() pipeline (price, score).
       Checked once against the real catalogue, so renderCompare can trust it
       instead of building every cell just to read its text back out. */
    columns.forEach(function (column) {
      if (column.key === "name" || column.multi || column.filter === "none") {
        column.compareByValue = false;
        return;
      }
      column.compareByValue = catalog.models.every(function (model) {
        var text = column.cell(model).textContent.trim();
        var cellPresent = text !== "" && text !== "—";
        return cellPresent === Boolean(column.value(model));
      });
    });
    return columns;
  }

  /* Groups a select column's values by column.filterKey (default: a case-fold),
     so spellings that should be treated the same share one option. */
  function filterOptionsFor(column, catalog) {
    var keyFn = column.filterKey || function (value) {
      return String(value).toLowerCase();
    };
    var groups = new Map();
    catalog.models.forEach(function (model) {
      (column.multi ? column.options(model) : [column.value(model)]).forEach(function (value) {
        if (!value) {
          return;
        }
        var key = keyFn(value);
        var counts = groups.get(key);
        if (!counts) {
          counts = new Map();
          groups.set(key, counts);
        }
        counts.set(value, (counts.get(value) || 0) + 1);
      });
    });
    var options = Array.from(groups.entries()).map(function (entry) {
      return { value: entry[0], label: mostCommonSpelling(entry[1]) };
    });
    options.sort(function (a, b) {
      return a.label.localeCompare(b.label);
    });
    return options;
  }

  /* The spelling used most across the catalogue is the one shown, so
     "Apache 2.0" (17 models) outranks "Apache-2.0" (1) as the option label. */
  function mostCommonSpelling(counts) {
    var best = null;
    counts.forEach(function (count, label) {
      if (!best || count > best.count || (count === best.count && label < best.label)) {
        best = { label: label, count: count };
      }
    });
    return best.label;
  }

  function scoreColumns(catalog) {
    var seen = new Map();
    catalog.models.forEach(function (model) {
      model.scores.forEach(function (score) {
        var key = score.source + ":" + score.board;
        if (!seen.has(key)) {
          seen.set(key, score);
        }
      });
    });
    return Array.from(seen.entries())
      .sort(function (a, b) {
        return a[1].label.localeCompare(b[1].label);
      })
      .map(function (entry) {
        var key = entry[0];
        var sample = entry[1];
        return {
          key: "score:" + key,
          label: sample.metric === "elo" ? sample.label + " Elo" : sample.label,
          filter: "none",
          visible: key === "lmarena:text",
          numeric: true,
          plottable: true,
          betterIsLower: !sample.higher_is_better,
          verdict: ["best", "lowest"],
          bestFirst: sample.higher_is_better ? "descending" : "ascending",
          help:
            (SCORE_BOARD_HELP[key] ? SCORE_BOARD_HELP[key] + " " : "") +
            (sample.higher_is_better ? "Higher is better. " : "Lower is better. ") +
            SCORE_HELP,
          value: function () {
            return "";
          },
          sort: function (model) {
            var score = findScore(model, key);
            return score ? score.value : null;
          },
          raw: function (model) {
            var score = findScore(model, key);
            return score ? score.value : null;
          },
          format: function (value) {
            if (value === null) {
              return "—";
            }
            return sample.unit === "%" ? value.toFixed(1) + "%" : String(Math.round(value));
          },
          cell: function (model) {
            var score = findScore(model, key);
            if (!score) {
              return el("td", { class: "models-num", text: "—" });
            }
            var text = score.unit === "%" ? score.value.toFixed(1) + "%" : Math.round(score.value);
            return el("td", {
              class: "models-num",
              text: String(text),
              title: score.matched_name + " · " + score.as_of + " · matched by " + score.match_method,
            });
          },
        };
      });
  }

  function indexOfKey(columns, key) {
    for (var i = 0; i < columns.length; i += 1) {
      if (columns[i].key === key) {
        return i;
      }
    }
    return columns.length - 1;
  }

  /* How the quoted product reads in a sentence, and what it costs to choose
     it over routing globally. */
  function routingNote(model) {
    var group = groupFor(model, state.region);
    if (!group) {
      return "";
    }
    var named = {
      region: "served in " + state.region + " itself",
      geography: "served through a geography profile that does not leave it",
      global: "served through the global profile, which may run anywhere",
    }[group.routing || ""] || "";
    var parts = named ? [named] : [];
    if (group.service) {
      parts.push("priced by " + group.service);
    }
    var index = state.regionIndex[state.region];
    var anywhere = model.price_groups.filter(function (other) {
      return other.routing === "global" && other.regions.indexOf(index) !== -1;
    })[0];
    if (anywhere && group.routing && group.routing !== "global") {
      var here = blended(group);
      var there = blended(anywhere);
      if (here && there && here !== there) {
        var delta = Math.round((Math.abs(here - there) / here) * 100);
        parts.push(
          delta
            ? "routing globally instead costs " + delta + "% "
              + (there < here ? "less" : "more")
            : "routing globally costs the same"
        );
      }
    }
    return parts.length ? ". " + parts.join("; ") + "." : "";
  }

  /*
   * A third of the catalogue is billed per image, per second or per character,
   * so a token-price column alone reads as "stdapi.ai has no price for this".
   * This one always shows whatever the model is actually billed on.
   */
  function primaryPrice(model) {
    var input = priceFor(model, state.region, "input_tokens");
    var output = priceFor(model, state.region, "output_tokens");
    if (input !== null && output !== null) {
      /* Sorted on a blended rate, not on input alone: output tokens usually
         dominate the bill, so ranking by input would put exactly the models
         people reach for in the wrong order. Three input to one output is the
         common convention for a mixed workload. */
      return {
        unit: "tokens",
        value: ((input * 3 + output) / 4) * 1e6,
        text: "$" + formatPrice(input, 1e6) + " → $" + formatPrice(output, 1e6) + " /1M",
        title:
          "Input and output tokens per million in " +
          state.region +
          ", sorted on a 3:1 input-to-output blend" +
          routingNote(model),
      };
    }
    if (input !== null || output !== null) {
      var only = input === null ? output : input;
      var side = input === null ? " /1M out" : " /1M in";
      return {
        unit: "tokens",
        value: only * 1e6,
        text: "$" + formatPrice(only, 1e6) + side,
        title: "Per million tokens in " + state.region + routingNote(model),
      };
    }
    for (var i = 0; i < PRICE_COLUMNS.length; i += 1) {
      var price = PRICE_COLUMNS[i];
      var value = priceFor(model, state.region, price.key);
      if (value !== null) {
        return {
          unit: price.key,
          value: value * price.scale,
          text: "$" + formatPrice(value, price.scale) + price.unit,
          title: price.label + " in " + state.region + routingNote(model),
        };
      }
    }
    return null;
  }

  /* Every region named across a model's own price groups — the regions AWS
     actually publishes a rate in, regardless of which one is selected. */
  function pricedRegions(model) {
    var names = [];
    model.price_groups.forEach(function (group) {
      group.regions.forEach(function (index) {
        var name = state.manifest.regions[index];
        if (name && names.indexOf(name) === -1) {
          names.push(name);
        }
      });
    });
    return names;
  }

  /*
   * A routing restriction only speaks about a region where AWS offers the
   * model: where it does not, the Regions filters decide whether the row
   * belongs and the blank price cell says why it is blank. Where it does, the
   * price groups name the routing being billed, and for the two models AWS
   * publishes no price for, the serving markers are the only statement left.
   */
  /* An empty list is a service the idea does not apply to — Polly, Translate,
     Transcribe — not a model that cannot be called without a profile. */
  function onDemand(model) {
    var types = model.inference_types || [];
    return !types.length || types.indexOf("ON_DEMAND") !== -1;
  }

  function routableIn(model, region) {
    var allowed = routingChoice().kinds;
    var index = state.regionIndex[region];
    if (!allowed || index === undefined || model.regions.indexOf(index) === -1) {
      return true;
    }
    /* Ruling out both cross-region kinds means no inference profile at all,
       and a model AWS publishes only through one cannot be reached that way:
       its in-region rate is what a profile bills when it lands there, not an
       on-demand product. */
    if (
      allowed.indexOf("geography") === -1
      && allowed.indexOf("global") === -1
      && !onDemand(model)
    ) {
      return false;
    }
    if (groupFor(model, region)) {
      return true;
    }
    if (pricedRegions(model).indexOf(region) !== -1) {
      return false;
    }
    return (model.serving || []).some(function (marker) {
      if (marker === "global") {
        return allowed.indexOf("global") !== -1;
      }
      if (SERVING_BUCKETS[marker]) {
        return (
          allowed.indexOf("geography") !== -1
          && SERVING_BUCKETS[marker] === state.manifest.region_buckets[region]
        );
      }
      return marker === region && allowed.indexOf("region") !== -1;
    });
  }

  function routableHere(model) {
    return routableIn(model, state.region);
  }

  /* A blank Price cell only means "not in this region" — the model can still
     be priced elsewhere, and the title says where. */
  function unpricedHere(model) {
    var priced = pricedRegions(model);
    if (!priced.length) {
      return "AWS publishes no price for this model.";
    }
    /* Two different reasons for a dash: AWS prices the model nowhere near
       here, or it prices it here but not for the routing the reader chose. */
    if (priced.indexOf(state.region) !== -1) {
      return (
        "Priced in " + state.region + ", but not "
        + routingChoice().label + ". Choose another routing to see its rate."
      );
    }
    return "Not priced in " + state.region + ". Priced in " + priced.sort().join(", ") + ".";
  }

  function primaryPriceColumn() {
    return {
      key: "price:primary",
      label: "Price",
      filter: "none",
      visible: true,
      numeric: true,
      plottable: true,
      betterIsLower: true,
      verdict: ["cheapest", "most expensive"],
      raw: function (model) {
        var found = primaryPrice(model);
        return found && found.unit === "tokens" ? found.value : null;
      },
      format: function (value) {
        return value === null ? "—" : "$" + formatPrice(value, 1);
      },
      help:
        "What AWS charges for the unit this model is billed on — tokens, images, "
        + "seconds or characters — in the selected region. Token models show input "
        + "then output per million and sort on a 3:1 blend of the two; models "
        + "billed on another unit sort among themselves.",
      value: function () {
        return "";
      },
      /* $0.0001 per second and $0.02 per million tokens are not comparable, so
         each unit sorts as its own block rather than interleaving by magnitude
         and putting a per-second service ahead of every token model. */
      sort: function (model) {
        var found = primaryPrice(model);
        if (!found) {
          return null;
        }
        return {
          block: found.unit === "tokens" ? "" : found.unit,
          value: found.value,
        };
      },
      cell: function (model) {
        var found = primaryPrice(model);
        if (!found) {
          return el("td", { text: "—", title: unpricedHere(model) });
        }
        var cell = el("td", { class: "models-price", title: found.title });
        cell.appendChild(el("span", { text: found.text }));
        var cheaper = state.tier === "cheapest" && cheaperTier(model, state.region);
        if (cheaper) {
          /* The CSS gap is visual only — a literal space keeps "in" and
             "batch" from reading as one word to a screen reader or a paste. */
          cell.appendChild(document.createTextNode(" "));
          cell.appendChild(
            el("span", {
              class: "models-badge",
              text: cheaper,
              title: "Cheaper on the " + cheaper + " tier",
            })
          );
        }
        if (model.prompt_caching) {
          var cacheRate = priceFor(model, state.region, "cache_read_tokens");
          cell.appendChild(document.createTextNode(" "));
          cell.appendChild(
            el("span", {
              class: "models-badge models-badge--cache",
              text: "cache",
              title:
                cacheRate === null
                  ? "Prompt caching cuts the cost of a repeated prompt prefix."
                  : "Prompt caching cuts the cost of a repeated prompt prefix — cached " +
                    "tokens are $" + formatPrice(cacheRate, 1e6) + " /1M here.",
            })
          );
        }
        return cell;
      },
    };
  }

  /* Permissive open, then any other open licence, then proprietary. Ranking a
     licence is a judgement, so it is one rank and one place, not a score. */
  var PERMISSIVE = /apache|^mit$|bsd|^cc-?by|openrail|^gemma$/i;

  /* A licence that forbids commercial use or derivatives is not permissive,
     however it is spelled — cc-by-nc-4.0 matches the CC pattern above. A
     research licence is restricted whether or not it spells out "only". */
  var RESTRICTED = /-nc\b|-nc-|-nd\b|-nd-|non-?commercial|research/i;

  /* "Apache 2.0" and "Apache-2.0" are the same licence; grouping the filter
     option on this key, not on the raw text, keeps them one entry. */
  function licenceFilterKey(text) {
    return String(text).toLowerCase().replace(/[\s-]+/g, " ").trim();
  }

  function licenceRank(licence) {
    if (!licence) {
      return null;
    }
    if (/proprietary|closed/i.test(licence)) {
      return 3;
    }
    if (RESTRICTED.test(licence)) {
      return 2;
    }
    return PERMISSIVE.test(licence) ? 1 : 2;
  }

  function findScore(model, key) {
    for (var i = 0; i < model.scores.length; i += 1) {
      if (model.scores[i].source + ":" + model.scores[i].board === key) {
        return model.scores[i];
      }
    }
    return null;
  }

  function textColumn(key, label, visible, read) {
    return {
      key: key,
      label: label,
      filter: "select",
      visible: visible,
      value: read,
      sort: function (model) {
        /* null, not "", or the missing-value branch in sortRows never fires
           and every blank cell lands at the top of an ascending sort. */
        var text = read(model);
        return text ? String(text).toLowerCase() : null;
      },
      cell: function (model) {
        return el("td", { text: read(model) || "—" });
      },
    };
  }

  function numberColumn(key, label, visible, read) {
    return {
      key: key,
      label: label,
      filter: "none",
      visible: visible,
      numeric: true,
      plottable: true,
      bestFirst: "descending",
      verdict: ["most", "fewest"],
      value: function () {
        return "";
      },
      sort: read,
      raw: read,
      format: function (value) {
        return value === null ? "—" : value.toLocaleString();
      },
      cell: function (model) {
        var value = read(model);
        return el("td", { class: "models-num", text: value === null || value === undefined ? "—" : value.toLocaleString() });
      },
    };
  }

  function listColumn(key, label, visible, decorate, wrap) {
    return {
      key: key,
      label: label,
      filter: "select",
      visible: visible,
      multi: true,
      decorate: decorate,
      value: function (model) {
        return (model[key] || []).join(" ");
      },
      options: function (model) {
        return model[key] || [];
      },
      sort: function (model) {
        return (model[key] || []).join(",").toLowerCase();
      },
      cell: function (model) {
        var values = model[key] || [];
        if (!values.length) {
          return el("td", { text: "—" });
        }
        if (!decorate) {
          return el("td", { text: values.join(", ") });
        }
        var cell = el("td", { class: "models-tags" + (wrap ? " models-tags--wrap" : "") });
        values.forEach(function (value) {
          cell.appendChild(decorate(value));
        });
        return cell;
      },
    };
  }

  function modalityTag(value) {
    var tag = el("span", { class: "models-tag", title: value });
    if (MODALITY_ICONS[value]) {
      tag.appendChild(glyph(MODALITY_ICONS[value]));
    }
    tag.appendChild(el("span", { text: value.charAt(0) + value.slice(1).toLowerCase() }));
    return tag;
  }

  /* Without a cross-region profile a model "runs in" every region it is served
     from, which is a wall of thirty codes. Past a handful, name the geographies. */
  function collapsedListColumn(key, label, visible) {
    var column = listColumn(key, label, visible, geographyTag);
    /* Only the comparison reads this — the filter dropdown and passesFilters()
       keep matching every marker regardless of geography, so narrowing here
       never removes a value the filter can still select. */
    column.narrowedOptions = countedServing;
    column.cell = function (model) {
      var all = model[key] || [];
      if (!all.length) {
        return el("td", { text: "—" });
      }
      var values = countedServing(model);
      var narrowed = state.buckets.size && values.length !== all.length;
      if (!values.length) {
        return el("td", {
          class: "models-tags",
          text: "—",
          title: "None of where it runs falls inside the selected geographies.",
        });
      }
      var cell = el("td", {
        class: "models-tags",
        title: (narrowed ? "In the selected geographies: " : "") + values.join(", "),
      });
      if (values.length <= 4) {
        values.forEach(function (value) {
          cell.appendChild(geographyTag(value));
        });
        return cell;
      }
      var buckets = {};
      values.forEach(function (value) {
        var bucket = SERVING_BUCKETS[value] || state.manifest.region_buckets[value] || value;
        buckets[bucket] = true;
      });
      Object.keys(buckets)
        .sort()
        .forEach(function (bucket) {
          cell.appendChild(geographyTag(bucket));
        });
      cell.appendChild(el("span", { class: "models-tag-more", text: values.length + " regions" }));
      return cell;
    };
    return column;
  }

  /* Says which of the two filters narrowed the list, so a count smaller than
     the model's own is never left unexplained. */
  function regionsNote(model) {
    var counted = countedRegions(model);
    var narrowed = [];
    if (state.buckets.size) {
      narrowed.push("in the selected geographies");
    }
    if (state.routing !== "any") {
      narrowed.push("with routing " + routingChoice().label);
    }
    var lead = narrowed.length ? "Callable " + narrowed.join(", ") : "Callable from";
    if (!counted.length) {
      return lead + ": nowhere.";
    }
    return (
      lead + ": "
      + counted
        .map(function (region) {
          var flag = flagFor(region);
          return (flag ? flag + " " : "") + region;
        })
        .join(", ")
    );
  }

  function geographyTag(value) {
    var country = countryOf(value);
    var flag = flagFor(value);
    var tag = el("span", {
      class: "models-tag",
      title: country ? value + " — " + country : value,
    });
    if (flag) {
      tag.appendChild(el("span", { class: "models-flag", text: flag }));
    } else {
      tag.appendChild(glyph(value === "global" ? GEO_ICONS.global : GEO_ICONS.region));
    }
    tag.appendChild(el("span", { text: value }));
    return tag;
  }

  /* Capabilities where "yes" is the better answer, and the two where it is not. */
  var WORSE_WHEN_TRUE = { legacy: true, retired: true };

  /* Parameter counts are written 235B, 8x7B, 671B: comparable once parsed. */
  function sizeColumn(key, label) {
    return {
      key: key,
      label: label,
      filter: "select",
      visible: false,
      numeric: true,
      verdict: ["largest", "smallest"],
      value: function (model) {
        return model[key] || "";
      },
      sort: function (model) {
        return parseParameters(model[key]);
      },
      cell: function (model) {
        return el("td", { class: "models-num", text: model[key] || "—" });
      },
    };
  }

  /* "235B", "1T", "30B-A3B" — the first count is the one being ranked, and any
     figure after it qualifies that count rather than multiplying it. */
  function parseParameters(value) {
    if (typeof value !== "string" || !value) {
      return null;
    }
    var found = /([\d.]+)\s*([KMBT])/i.exec(value);
    if (!found) {
      return null;
    }
    var number = parseFloat(found[1]);
    var scale = { k: 1e3, m: 1e6, b: 1e9, t: 1e12 }[found[2].toLowerCase()];
    return isNaN(number) || !number ? null : number * scale;
  }

  function boolColumn(key, label, visible) {
    var worseWhenTrue = Boolean(WORSE_WHEN_TRUE[key]);
    return {
      key: key,
      label: label,
      filter: "select",
      visible: visible,
      numeric: true,
      betterIsLower: !worseWhenTrue,
      verdict: worseWhenTrue ? ["no", "yes"] : ["yes", "no"],
      /* The verdict word would just repeat the cell's own "Yes"/"No" — the
         comparison already suppresses that label for this column. */
      booleanVerdict: true,
      value: function (model) {
        if (model[key] === true) {
          return "yes";
        }
        return model[key] === false ? "no" : "unknown";
      },
      sort: function (model) {
        if (model[key] === true) {
          return 0;
        }
        return model[key] === false ? 1 : null;
      },
      cell: function (model) {
        var text = model[key] === true ? "Yes" : model[key] === false ? "No" : "—";
        return el("td", { text: text });
      },
    };
  }

  /* -- filtering and sorting --------------------------------------------- */

  function visibleModels() {
    return state.catalog.models.filter(function (model) {
      return passesFilters(model);
    });
  }

  function passesFilters(model, ignoreLegacy) {
    if (!ignoreLegacy && !state.showLegacy && (model.legacy || model.retired)) {
      return false;
    }
    if (state.onlyScored && !model.scores.length) {
      return false;
    }
    /* Restricting where inference may run is a requirement, not a preference:
       a model AWS does not offer that way in this region is not a candidate,
       and listing it with a dash buries the ones that are. */
    if (!routableHere(model)) {
      return false;
    }
    var modalities = state.modalities;
    for (var key in modalities) {
      if (!Object.prototype.hasOwnProperty.call(modalities, key) || !modalities[key].size) {
        continue;
      }
      var owned = new Set(model[key] || []);
      var missing = Array.from(modalities[key]).some(function (value) {
        return !owned.has(value);
      });
      if (missing) {
        return false;
      }
    }
    if (state.buckets.size) {
      var buckets = state.sense === "runs" ? runsBuckets(model) : new Set(model.buckets);
      var hit = Array.from(state.buckets).some(function (bucket) {
        return bucket === "global" ? model.serving.indexOf("global") !== -1 : buckets.has(bucket);
      });
      if (!hit) {
        return false;
      }
    }
    /* Every column's value() is a pure function of the model, so the haystack
       is built once per model rather than on every keystroke across 142 rows. */
    if (state.search) {
      if (state.haystacks[model.id] === undefined) {
        state.haystacks[model.id] = state.columns
          .map(function (column) {
            return column.value(model);
          })
          .join(" ")
          .toLowerCase();
      }
      if (state.haystacks[model.id].indexOf(state.search) === -1) {
        return false;
      }
    }
    var keys = Object.keys(state.filters);
    for (var i = 0; i < keys.length; i += 1) {
      var wanted = state.filters[keys[i]];
      if (!wanted) {
        continue;
      }
      var column = columnByKey(keys[i]);
      if (!column) {
        continue;
      }
      if (column.filter === "select") {
        var keyFn = column.filterKey || function (value) {
          return String(value).toLowerCase();
        };
        var parts = column.multi ? column.options(model) : [column.value(model)];
        var matched = parts.some(function (part) {
          return keyFn(part) === wanted;
        });
        if (!matched) {
          return false;
        }
      } else if (String(column.value(model)).toLowerCase().indexOf(wanted) === -1) {
        return false;
      }
    }
    return true;
  }

  function sortRows(rows) {
    if (!state.sort) {
      return rows;
    }
    var column = columnByKey(state.sort.key);
    if (!column) {
      return rows;
    }
    var sign = state.sort.direction === "ascending" ? 1 : -1;
    /* One sort() call per row, not two per comparison: the price and region
       keys each walk the model's price groups, and a comparison sort asks for
       them ~2,000 times over 142 rows. */
    var keyed = rows.map(function (model) {
      return { model: model, key: column.sort(model) };
    });
    keyed.sort(function (a, b) {
      /* A model with no value stays last either way, so reversing the sort
         never fills the top of the table with empty cells. */
      var leftMissing = valueOf(a.key) === null || valueOf(a.key) === undefined;
      var rightMissing = valueOf(b.key) === null || valueOf(b.key) === undefined;
      if (leftMissing || rightMissing) {
        if (leftMissing === rightMissing) {
          return a.model.id.localeCompare(b.model.id);
        }
        return leftMissing ? 1 : -1;
      }
      /* The block never reverses: "most expensive first" has no meaning across
         billed units, so per-second models stay together below the token
         models whichever way the column is sorted. */
      var block = compare(a.key.block, b.key.block);
      if (block) {
        return block;
      }
      var order = compare(valueOf(a.key), valueOf(b.key));
      return order ? order * sign : a.model.id.localeCompare(b.model.id);
    });
    return keyed.map(function (entry) {
      return entry.model;
    });
  }

  function valueOf(key) {
    return key && key.block !== undefined ? key.value : key;
  }

  function compare(left, right) {
    if (left < right) {
      return -1;
    }
    return left > right ? 1 : 0;
  }

  function columnByKey(key) {
    return state.columnIndex[key];
  }

  function visibleColumns() {
    return state.columns.filter(function (column) {
      return column.visible;
    });
  }

  /* -- rendering --------------------------------------------------------- */

  /*
   * Re-rendering replaces the node the reader is standing on, which drops focus
   * to the document body with no announcement. Every rebuilt control carries a
   * stable data attribute, so focus is restored to the same one afterwards.
   */
  function keepFocus(work) {
    var active = document.activeElement;
    var selector = null;
    if (active && app.contains(active)) {
      ["data-sort", "data-highlight", "data-view", "data-bucket", "data-column", "data-filter"].some(
        function (name) {
          var value = active.getAttribute(name);
          if (value) {
            selector = "[" + name + '="' + value.replace(/"/g, '\\"') + '"]';
          }
          return Boolean(value);
        }
      );
    }
    work();
    if (!selector || app.contains(document.activeElement)) {
      return;
    }
    var restored = app.querySelector(selector);
    if (restored) {
      restored.focus({ preventScroll: true });
    }
  }

  function render() {
    keepFocus(paint);
  }

  function paint() {
    var rows = sortRows(visibleModels());
    renderStats(rows);
    state.nodes.chooser.hidden = state.view === "chart";
    if (state.view === "chart") {
      state.nodes.tableWrap.hidden = true;
      state.nodes.chartWrap.hidden = false;
      renderChart(rows);
    } else {
      state.nodes.chartWrap.hidden = true;
      state.nodes.tableWrap.hidden = false;
      renderHead();
      renderBody(rows);
      updateScrollHints();
    }
    syncControls();
    var active = activeRefinements();
    state.nodes.refineCount.textContent = active ? String(active) : "";
    state.nodes.refineCount.hidden = !active;
    var summary = rows.length + " of " + state.catalog.models.length + " models";
    if (state.nodes.count.textContent !== summary) {
      state.nodes.count.textContent = summary;
      announce(rows.length ? summary : "No model matches these filters.");
    }
    renderCompare();
    syncUrl();
  }

  function renderStats(rows) {
    /* An empty result set is not "0 providers, 0 regions" — that is just the
       last real answer going stale. Leave the tiles as they were. */
    if (!rows.length) {
      return;
    }
    var providers = new Set();
    var regions = new Set();
    var scored = 0;
    rows.forEach(function (model) {
      providers.add(model.provider);
      /* The same narrowing the Regions column applies: with a geography
         selected, count only the regions that match it. */
      countedRegions(model).forEach(function (region) {
        regions.add(region);
      });
      if (model.scores.length) {
        scored += 1;
      }
    });
    var cheapest = rows.reduce(function (best, model) {
      var input = priceFor(model, state.region, "input_tokens");
      return input !== null && (best === null || input < best) ? input : best;
    }, null);
    var tiles = [
      { value: providers.size, label: providers.size === 1 ? "provider" : "providers" },
      { value: regions.size, label: regions.size === 1 ? "AWS region" : "AWS regions" },
      {
        value: cheapest === null ? "—" : "$" + formatPrice(cheapest, 1e6),
        label: "cheapest input, per 1M",
      },
      { value: scored, label: "with a public benchmark score" },
    ];
    var host = state.nodes.stats;
    host.textContent = "";
    tiles.forEach(function (tile) {
      host.appendChild(
        el("div", { class: "models-stat" }, [
          el("span", { class: "models-stat-value", text: String(tile.value) }),
          el("span", { class: "models-stat-label", text: tile.label }),
        ])
      );
    });
  }

  /* Rebuilding the header on every keystroke would destroy the filter input the
     reader is typing into, so it is only rebuilt when its shape changes. */
  function headSignature() {
    return (
      visibleColumns()
        .map(function (column) {
          return column.key;
        })
        .join("|") +
      "#" +
      (state.sort ? state.sort.key + state.sort.direction : "")
    );
  }

  function renderHead() {
    var signature = headSignature();
    if (state.headSignature === signature) {
      return;
    }
    state.headSignature = signature;
    var head = state.nodes.head;
    head.textContent = "";
    var headings = el("tr", {}, [
      el("th", { class: "models-pick", scope: "col" }, [
        el("span", { class: "models-sr", text: "Select" }),
      ]),
    ]);
    /* The filter row is controls, not headers: as <th> every data cell in the
       column would be announced with its filter's text as well as its name. */
    var filters = el("tr", { class: "models-filters" }, [el("td", { class: "models-pick" })]);
    visibleColumns().forEach(function (column) {
      var sorted = state.sort && state.sort.key === column.key ? state.sort.direction : "none";
      var button = el("button", {
        type: "button",
        class: "models-sort",
        "data-sort": column.key,
        title: column.help
          ? column.label + " — " + column.help + "\n\nActivate to sort."
          : "Sort by " + column.label,
      });
      button.appendChild(el("span", { text: column.label }));
      if (sorted !== "none") {
        button.appendChild(
          el("span", {
            class: "models-sort-arrow",
            "aria-hidden": "true",
            text: sorted === "ascending" ? " \u25b2" : " \u25bc",
          })
        );
      }
      headings.appendChild(
        el(
          "th",
          {
            scope: "col",
            "aria-sort": sorted === "none" ? null : sorted,
            class: column.sticky ? "models-sticky" : null,
          },
          [button]
        )
      );
      filters.appendChild(
        el("td", { class: column.sticky ? "models-sticky" : null }, [filterControl(column, "in the table header")])
      );
    });
    head.appendChild(headings);
    head.appendChild(filters);
  }

  /* This same control also gets built for the Filters panel, so the label
     needs a location to stay distinct from the in-table one — two controls
     with the identical name and no way to tell them apart is its own bug. */
  function filterControl(column, where) {
    var label = "Filter by " + column.label + (where ? ", " + where : "");
    if (column.filter === "none") {
      return el("span", { class: "models-filter-none", text: "" });
    }
    if (column.filter === "select") {
      var select = el("select", {
        class: "models-filter",
        "data-filter": column.key,
        "aria-label": label,
      });
      select.appendChild(el("option", { value: "", text: "all" }));
      column.filterOptions.forEach(function (option) {
        var item = el("option", { value: option.value, text: option.label });
        if (state.filters[column.key] === option.value) {
          item.setAttribute("selected", "selected");
        }
        select.appendChild(item);
      });
      return select;
    }
    return el("input", {
      type: "search",
      class: "models-filter",
      placeholder: "filter",
      value: state.filters[column.key] || "",
      "data-filter": column.key,
      "aria-label": label,
    });
  }

  /* How many hidden legacy/retired models would match everything else. */
  function legacyMatchCount() {
    if (state.showLegacy) {
      return 0;
    }
    return state.catalog.models.filter(function (model) {
      return (model.legacy || model.retired) && passesFilters(model, true);
    }).length;
  }

  /* The page's own how-to promises a legacy model stays findable, so the
     empty state has to say when hiding it is the only reason nothing shows,
     and let the reader undo that in one click. It also names the actual
     cause — a search term or a filter — instead of a generic "these filters". */
  function emptyStateNode() {
    var wrap = el("span", {});
    wrap.appendChild(
      document.createTextNode(state.search ? 'No model matches "' + state.search + '".' : "No model matches these filters.")
    );
    var hidden = legacyMatchCount();
    if (hidden) {
      wrap.appendChild(
        document.createTextNode(
          " " + hidden + (hidden === 1 ? " legacy model matches" : " legacy models match") + " — "
        )
      );
      var button = el("button", { type: "button", class: "models-link", text: "Show legacy" });
      button.addEventListener("click", function () {
        state.showLegacy = true;
        render();
      });
      wrap.appendChild(button);
    }
    return wrap;
  }

  function renderBody(rows) {
    var body = state.nodes.body;
    body.textContent = "";
    if (!rows.length) {
      body.appendChild(
        el("tr", {}, [
          el(
            "td",
            { class: "models-empty", colspan: String(visibleColumns().length + 1) },
            [emptyStateNode()]
          ),
        ])
      );
      return;
    }
    var columns = visibleColumns();
    var fragment = document.createDocumentFragment();
    rows.forEach(function (model) {
      var checkbox = el("input", {
        type: "checkbox",
        "data-compare": model.id,
        /* One tick opens this model's card; two or three compare them — the
           name has to cover both, not just the second one. */
        "aria-label": "Select " + model.name + " to open or compare",
      });
      if (state.compare.indexOf(model.id) !== -1) {
        checkbox.checked = true;
      }
      var row = el("tr", { class: model.legacy ? "is-legacy" : null }, [
        el("td", { class: "models-pick", "data-label": "Select" }, [checkbox]),
      ]);
      columns.forEach(function (column) {
        var cell = column.cell(model);
        if (column.sticky) {
          cell.classList.add("models-sticky");
        }
        /* Read by the stacked-card layout below 37.5em, where a column
           heading is not there to say what the value is. The name is the
           card's own heading, so it needs no label. */
        if (column.key !== "name") {
          cell.setAttribute("data-label", column.label);
        }
        row.appendChild(cell);
      });
      fragment.appendChild(row);
    });
    body.appendChild(fragment);
  }

  function updateScrollHints() {
    /* Interleaving style writes with layout reads forces a synchronous reflow
       on every scroll event; all the reads happen first. */
    if (state.scrollFrame) {
      return;
    }
    state.scrollFrame = window.requestAnimationFrame(function () {
      state.scrollFrame = 0;
      measureScroll();
    });
  }

  function measureScroll() {
    var wrap = state.nodes.scroll;
    var frame = state.nodes.tableWrap;
    var headings = state.nodes.head.firstChild;
    var height = headings && headings.getBoundingClientRect
      ? headings.getBoundingClientRect().height
      : 0;
    var top = wrap.offsetTop;
    /* Below the stacking breakpoint the rows carry their own labels and stop
       overflowing, but the heading strip keeps every sort and filter control
       on one scrolling line — so it is the strip's overflow that the reader
       has to be told about. */
    var strip = state.nodes.head;
    var stripOverflow = strip ? strip.scrollWidth - strip.clientWidth : 0;
    var overflow = Math.max(wrap.scrollWidth - wrap.clientWidth, stripOverflow);
    var right = overflow > 2 && wrap.scrollLeft < overflow - 2;
    var left = wrap.scrollLeft > 2;
    if (height) {
      wrap.style.setProperty("--models-head-height", height + "px");
    }
    frame.style.setProperty("--models-scroll-top", top + "px");
    [wrap, frame].forEach(function (node) {
      node.classList.toggle("can-scroll-right", right);
      node.classList.toggle("can-scroll-left", left);
    });
    /* Only the element that actually scrolls is announced and reachable as a
       scrollable region: above the breakpoint that is the table, below it the
       heading strip, and claiming both would send a reader to a dead end. */
    var bodyScrolls = wrap.scrollWidth - wrap.clientWidth > 2;
    wrap.setAttribute(
      "aria-label",
      bodyScrolls ? "Model table, scrollable" : "Model table"
    );
    /* Only the tabindex: the strip is the table's own rowgroup, and giving it
       any other role would leave its rows outside one. Every heading cell
       holds a focusable control, so tabbing already scrolls it. */
    if (strip && stripOverflow > 2) {
      strip.setAttribute("tabindex", "0");
    } else if (strip) {
      strip.removeAttribute("tabindex");
    }
    state.nodes.scrollHint.hidden = overflow <= 2;
    state.nodes.scrollHint.textContent =
      wrap.scrollWidth - wrap.clientWidth > 2
        ? "Scroll sideways for more columns, or choose them above."
        : "Scroll the heading row sideways to sort or filter on another field.";
  }

  /* -- chart ------------------------------------------------------------- */

  function plottableColumns() {
    return state.columns.filter(function (column) {
      return column.plottable;
    });
  }

  function renderChart(rows) {
    var host = state.nodes.chart;
    host.textContent = "";
    var xColumn = columnByKey(state.chart.x) || plottableColumns()[0];
    var yColumn = columnByKey(state.chart.y) || plottableColumns()[1];
    if (!xColumn || !yColumn) {
      host.appendChild(el("p", { class: "models-empty", text: "Nothing to plot." }));
      return;
    }

    var points = rows
      .map(function (model) {
        return { model: model, x: xColumn.raw(model), y: yColumn.raw(model) };
      })
      .filter(function (point) {
        return point.x !== null && point.x !== undefined && point.y !== null && point.y !== undefined;
      });

    state.nodes.chartNote.textContent =
      "The dashed line joins the models nothing else beats on both measures at " +
      "once — the best available trade-off. Every other model is beaten by " +
      "something on the line. " +
      points.length +
      " of " +
      rows.length +
      " models have both values; the rest cannot be plotted.";

    /* Computed before either empty check, so state.chart.known and .highlight
       are never left describing the previous filter's providers — including
       while this render shows no points at all. */
    var counts = new Map();
    points.forEach(function (point) {
      counts.set(point.model.provider, (counts.get(point.model.provider) || 0) + 1);
    });
    state.chart.known = Array.from(counts.keys());
    var byCount = Array.from(counts.entries()).sort(function (a, b) {
      return b[1] - a[1] || a[0].localeCompare(b[0]);
    });
    state.chart.highlight = byCount
      .map(function (entry) {
        return entry[0];
      })
      .filter(shownProvider)
      .slice(0, HIGHLIGHT_LIMIT);

    if (!points.length) {
      host.appendChild(chartLegend(counts));
      host.appendChild(
        el("p", { class: "models-empty", text: "No model has both of these values." })
      );
      return;
    }

    points = points.filter(function (point) {
      return shownProvider(point.model.provider);
    });
    if (!points.length) {
      host.appendChild(chartLegend(counts));
      host.appendChild(
        el("p", { class: "models-empty", text: "No provider is selected." })
      );
      return;
    }

    host.appendChild(chartLegend(counts));
    host.appendChild(state.nodes.chartNote);
    host.appendChild(plot(points, xColumn, yColumn));
    host.appendChild(state.nodes.tip);
  }

  /*
   * The models no other model beats on both axes at once. On price against a
   * benchmark this is the whole question — everything above the line is paying
   * more for less — so the frontier is drawn, labelled, and left visible even
   * when its provider is not one of the highlighted three.
   */
  /* The frontier already honours betterIsLower; the axis title says so too,
     since "Open ASR (WER)" alone does not say which end is good. */
  function directionLabel(column) {
    if (column.betterIsLower === true || column.bestFirst === "ascending") {
      return " (lower is better)";
    }
    if (column.betterIsLower === false || column.bestFirst === "descending") {
      return " (higher is better)";
    }
    return "";
  }

  function frontier(points, xColumn, yColumn) {
    var xLower = Boolean(xColumn.betterIsLower);
    var yLower = Boolean(yColumn.betterIsLower);
    var better = function (a, b, lower) {
      return lower ? a < b : a > b;
    };
    var atLeast = function (a, b, lower) {
      return lower ? a <= b : a >= b;
    };
    return points.filter(function (candidate) {
      return !points.some(function (other) {
        if (other === candidate) {
          return false;
        }
        var dominatesX = atLeast(other.x, candidate.x, xLower);
        var dominatesY = atLeast(other.y, candidate.y, yLower);
        var strictly =
          better(other.x, candidate.x, xLower) || better(other.y, candidate.y, yLower);
        return dominatesX && dominatesY && strictly;
      });
    });
  }

  function chartLegend(counts) {
    var legend = el("div", { class: "models-legend" });
    legend.appendChild(el("span", { class: "models-legend-title", text: "Providers" }));
    legend.appendChild(
      el("button", {
        type: "button",
        class: "models-legend-chip",
        text: "All",
        "data-providers": "all",
      })
    );
    legend.appendChild(
      el("button", {
        type: "button",
        class: "models-legend-chip",
        text: "None",
        "data-providers": "none",
      })
    );
    Array.from(counts.entries())
      .sort(function (a, b) {
        return b[1] - a[1] || a[0].localeCompare(b[0]);
      })
      .forEach(function (entry) {
        var provider = entry[0];
        var shown = shownProvider(provider);
        var slot = state.chart.highlight.indexOf(provider);
        var chip = el("button", {
          type: "button",
          class:
            "models-legend-chip" +
            (slot === -1 ? "" : " series-" + (slot + 1)) +
            (shown ? "" : " is-off"),
          "aria-pressed": shown ? "true" : "false",
          "data-highlight": provider,
          /* The name and count sit in adjacent spans with no separator between
             them, so the accessible name needs one the visual layout does not. */
          "aria-label": provider + ", " + entry[1] + " models plotted",
          title:
            provider +
            " — " +
            entry[1] +
            " models plotted. Click to " +
            (shown ? "hide" : "show") +
            ".",
        });
        chip.appendChild(
          svg("svg", { class: "models-swatch", viewBox: "0 0 12 12", "aria-hidden": "true" }, [
            marker(slot, 6, 6, 5),
          ])
        );
        chip.appendChild(el("span", { text: provider }));
        chip.appendChild(el("span", { class: "models-legend-count", text: String(entry[1]) }));
        legend.appendChild(chip);
      });
    /* Only the first three get a colour and shape — silent otherwise, the
       plot would look like it forgot the other providers instead of grouping
       them on purpose. */
    if (counts.size > HIGHLIGHT_LIMIT) {
      legend.appendChild(
        el("span", {
          class: "models-legend-note",
          text:
            "Only the first " + HIGHLIGHT_LIMIT + " stay distinguishable by colour and shape — " +
            "the rest plot as grey diamonds, named on hover or focus.",
        })
      );
    }
    return legend;
  }

  /* Shape doubles for colour, so a highlighted provider is never colour-alone. */
  function marker(slot, cx, cy, size) {
    var cls = slot === -1 ? "series-other" : "series-" + (slot + 1);
    /* Four shapes for four groups, so the plot still separates without colour
       — in forced-colours mode, in print, and for a reader who sees no hue. */
    if (slot === -1) {
      var d = size * 0.95;
      return svg("polygon", {
        class: cls,
        points: [cx, cy - d, cx + d, cy, cx, cy + d, cx - d, cy].join(" "),
      });
    }
    if (slot === 1) {
      return svg("rect", {
        class: cls,
        x: cx - size * 0.8,
        y: cy - size * 0.8,
        width: size * 1.6,
        height: size * 1.6,
        rx: 1,
      });
    }
    if (slot === 2) {
      var s = size * 1.05;
      return svg("polygon", {
        class: cls,
        points: [cx, cy - s, cx + s, cy + s * 0.8, cx - s, cy + s * 0.8].join(" "),
      });
    }
    return svg("circle", { class: cls, cx: cx, cy: cy, r: size * 0.9 });
  }

  /* A log axis is warranted by the spread of the data, not by what it measures:
     prices span four orders of magnitude, Elo ratings span a third of one. */
  function spansOrders(values) {
    var positive = values.filter(function (value) {
      return value > 0;
    });
    if (positive.length < values.length || !positive.length) {
      return false;
    }
    return Math.max.apply(null, positive) / Math.min.apply(null, positive) > 100;
  }

  function scaleFor(values, logarithmic) {
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    if (logarithmic) {
      min = Math.max(min, 1e-9);
      max = Math.max(max, min * 10);
      return { min: Math.log10(min), max: Math.log10(max), log: true };
    }
    if (min === max) {
      min -= 1;
      max += 1;
    }
    var pad = (max - min) * 0.08;
    return { min: min - pad, max: max + pad, log: false };
  }

  function project(scale, value) {
    var raw = scale.log ? Math.log10(Math.max(value, 1e-9)) : value;
    return (raw - scale.min) / (scale.max - scale.min);
  }

  /* Readable axis labels: 0.1, 0.5, 1, 5 rather than 0.1592, 0.7246, 3.3. */
  function ticks(scale, format) {
    var values = scale.log ? logTicks(scale) : linearTicks(scale);
    return values.map(function (value) {
      return { fraction: project(scale, value), label: format(value) };
    });
  }

  function logTicks(scale) {
    var low = Math.pow(10, scale.min);
    var high = Math.pow(10, scale.max);
    var values = [];
    for (var exponent = Math.floor(scale.min); exponent <= Math.ceil(scale.max); exponent += 1) {
      [1, 2, 5].forEach(function (multiple) {
        var value = multiple * Math.pow(10, exponent);
        if (value >= low * 0.999 && value <= high * 1.001) {
          values.push(value);
        }
      });
    }
    while (values.length > 6) {
      values = values.filter(function (_, index) {
        return index % 2 === 0;
      });
    }
    return values;
  }

  function linearTicks(scale) {
    var span = scale.max - scale.min;
    var rough = span / 4;
    var magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
    var normalised = rough / magnitude;
    var step = (normalised <= 1 ? 1 : normalised <= 2 ? 2 : normalised <= 5 ? 5 : 10) * magnitude;
    var values = [];
    for (
      var value = Math.ceil(scale.min / step) * step;
      value <= scale.max + step * 1e-6;
      value += step
    ) {
      values.push(value);
    }
    return values;
  }

  /* A per-character multiplier underestimates a capital-heavy name enough to
     escape the viewBox; canvas measureText reads the real glyph widths for
     the font .models-point-label actually renders in, no DOM attachment
     required. Falls back to the old estimate if canvas is unavailable. */
  var labelMeasureContext = null;
  function measureLabelWidth(text) {
    if (labelMeasureContext === null) {
      var canvas = document.createElement("canvas");
      labelMeasureContext = canvas.getContext && canvas.getContext("2d");
    }
    if (!labelMeasureContext) {
      return text.length * 5.6;
    }
    var family = state.nodes.chart
      ? window.getComputedStyle(state.nodes.chart).fontFamily
      : "sans-serif";
    labelMeasureContext.font = "600 10.5px " + family;
    return labelMeasureContext.measureText(text).width;
  }

  function plot(points, xColumn, yColumn) {
    /* The viewBox scales with the container, so the type has to scale the other
       way or 11px becomes 4px on a phone. */
    var available = state.nodes.chart.clientWidth || 900;
    var width = Math.max(420, Math.min(900, available));
    var height = width < 620 ? 380 : 460;
    var pad = { top: 16, right: 20, bottom: 54, left: 76 };
    var innerWidth = width - pad.left - pad.right;
    var innerHeight = height - pad.top - pad.bottom;

    var xValues = points.map(function (point) {
      return point.x;
    });
    var yValues = points.map(function (point) {
      return point.y;
    });
    var xScale = scaleFor(xValues, spansOrders(xValues));
    var yScale = scaleFor(yValues, spansOrders(yValues));

    var root = svg("svg", {
      class: "models-plot",
      viewBox: "0 0 " + width + " " + height,
      role: "group",
      "aria-label":
        yColumn.label + " against " + xColumn.label + " for " + points.length + " models",
      "aria-describedby": "models-chart-note",
    });

    ticks(yScale, yColumn.format).forEach(function (tick) {
      var y = pad.top + innerHeight * (1 - tick.fraction);
      root.appendChild(
        svg("line", { class: "models-grid", x1: pad.left, x2: pad.left + innerWidth, y1: y, y2: y })
      );
      root.appendChild(
        svg("text", {
          class: "models-axis-label",
          x: pad.left - 10,
          y: y + 4,
          "text-anchor": "end",
          text: tick.label,
        })
      );
    });
    ticks(xScale, xColumn.format).forEach(function (tick) {
      var x = pad.left + innerWidth * tick.fraction;
      root.appendChild(
        svg("line", {
          class: "models-grid",
          x1: x,
          x2: x,
          y1: pad.top,
          y2: pad.top + innerHeight,
        })
      );
      root.appendChild(
        svg("text", {
          class: "models-axis-label",
          x: x,
          y: pad.top + innerHeight + 20,
          "text-anchor": "middle",
          text: tick.label,
        })
      );
    });

    root.appendChild(
      svg("text", {
        class: "models-axis-title",
        x: pad.left + innerWidth / 2,
        y: height - 14,
        "text-anchor": "middle",
        text: xColumn.label + (xScale.log ? " (log scale)" : "") + directionLabel(xColumn),
      })
    );
    root.appendChild(
      svg("text", {
        class: "models-axis-title",
        x: 16,
        y: pad.top + innerHeight / 2,
        "text-anchor": "middle",
        transform: "rotate(-90 16 " + (pad.top + innerHeight / 2) + ")",
        text: yColumn.label + directionLabel(yColumn),
      })
    );

    var best = frontier(points, xColumn, yColumn);
    var bestSet = new Set(
      best.map(function (point) {
        return point.model.id;
      })
    );
    var place = function (point) {
      return {
        x: pad.left + innerWidth * project(xScale, point.x),
        y: pad.top + innerHeight * (1 - project(yScale, point.y)),
      };
    };

    if (best.length > 1) {
      var path = best
        .slice()
        .sort(function (a, b) {
          return a.x - b.x;
        })
        .map(function (point) {
          var at = place(point);
          return at.x + "," + at.y;
        })
        .join(" ");
      root.appendChild(svg("polyline", { class: "models-frontier", points: path }));
    }

    /* Neutral dots first, so a highlighted provider is never hidden behind one. */
    points
      .slice()
      .sort(function (a, b) {
        return (
          state.chart.highlight.indexOf(a.model.provider) -
          state.chart.highlight.indexOf(b.model.provider)
        );
      })
      .forEach(function (point) {
        var slot = state.chart.highlight.indexOf(point.model.provider);
        var cx = pad.left + innerWidth * project(xScale, point.x);
        var cy = pad.top + innerHeight * (1 - project(yScale, point.y));
        var onFrontier = bestSet.has(point.model.id);
        var dot = marker(slot, cx, cy, slot === -1 ? (onFrontier ? 5.5 : 4.5) : 6);
        dot.classList.add("models-dot");
        if (onFrontier) {
          dot.classList.add("is-frontier");
        }
        dot.setAttribute("data-plot", point.model.id);
        dot.setAttribute("tabindex", "0");
        dot.setAttribute("role", "button");
        dot.setAttribute(
          "aria-label",
          disambiguatedName(point.model) + ", " + xColumn.label + " " + xColumn.format(point.x) +
            ", " + yColumn.label + " " + yColumn.format(point.y)
        );
        /* Separate attributes, not a delimited string — a model name can
           itself contain "|", which a split("|") would misparse. */
        dot.dataset.tipName = disambiguatedName(point.model);
        dot.dataset.tipProvider = point.model.provider;
        dot.dataset.tipX = xColumn.label + ": " + xColumn.format(point.x);
        dot.dataset.tipY = yColumn.label + ": " + yColumn.format(point.y);
        root.appendChild(dot);
      });

    /* Direct labels for the frontier: the reader should not have to hover to
       learn which models are the ones worth looking at. */
    var placed = [];
    best
      .slice()
      .sort(function (a, b) {
        return a.x - b.x;
      })
      .filter(function (_, index, all) {
        /* Spread the labels along the whole line: labelling the first ten by
           price leaves the expensive end, which a budget holder is reading,
           unnamed. */
        var step = Math.max(1, Math.ceil(all.length / 10));
        return index % step === 0 || index === all.length - 1;
      })
      .forEach(function (point) {
        var at = place(point);
        var label = { x: at.x + 9, y: at.y - 8, width: measureLabelWidth(point.model.name) };
        /* Clamped inside the plot's own edges, or a name near the right side
           runs off the viewBox instead of staying on the canvas. */
        label.x = Math.min(label.x, width - pad.right - label.width);
        label.x = Math.max(label.x, pad.left);
        /* Nudge a label down until it clears the ones already placed, so the
           frontier reads as a list rather than as overlapping ink. */
        var collides = function (candidate) {
          return placed.some(function (other) {
            return (
              Math.abs(candidate.y - other.y) < 13 &&
              candidate.x < other.x + other.width &&
              other.x < candidate.x + candidate.width
            );
          });
        };
        for (var attempt = 0; attempt < 6 && collides(label); attempt += 1) {
          label.y += 14;
        }
        /* Clamped the same way as x: six nudges of 14px can in principle push
           a label below the axis, even though no current data reaches it. */
        label.y = Math.min(label.y, height - pad.bottom - 4);
        label.y = Math.max(label.y, pad.top + 10);
        /* Still colliding after every nudge — dropped rather than overprinted;
           the point itself is still on the frontier and still plotted. */
        if (collides(label)) {
          return;
        }
        placed.push(label);
        root.appendChild(
          svg("text", {
            class: "models-point-label",
            x: label.x,
            y: label.y,
            text: disambiguatedName(point.model),
          })
        );
      });

    return root;
  }

  /* -- comparison -------------------------------------------------------- */

  /*
   * The comparison sits below a table that is a screen and a half long, so
   * ticking a box at the top would otherwise change something the reader
   * cannot see. A tray pinned to the viewport says what is selected and takes
   * them to it.
   */
  function renderTray() {
    var tray = state.nodes.tray;
    tray.textContent = "";
    if (!state.compare.length) {
      tray.hidden = true;
      updateTrayReserve();
      return;
    }
    var models = state.compare
      .map(function (id) {
        return state.byId[id];
      })
      .filter(Boolean);

    var list = el("div", { class: "models-tray-list" });
    models.forEach(function (model) {
      var chip = el("span", { class: "models-tray-chip" });
      var logo = logoFor(model);
      if (logo) {
        chip.appendChild(logo);
      }
      chip.appendChild(el("span", { text: disambiguatedName(model) }));
      chip.appendChild(
        el(
          "button",
          {
            type: "button",
            class: "models-tray-drop",
            "data-drop": model.id,
            "aria-label": "Remove " + disambiguatedName(model) + " from the selection",
            title: "Remove " + disambiguatedName(model),
          },
          [el("span", { "aria-hidden": "true", text: "\u00d7" })]
        )
      );
      list.appendChild(chip);
    });
    /* Below ~420px the per-model chip list stacks tall enough to cover the
       plot and the panel under it — a narrow screen gets this one-line
       summary instead, toggled by CSS, not this list. */
    tray.appendChild(
      el("span", { class: "models-tray-summary", text: models.length + " selected" })
    );
    tray.appendChild(list);

    var action = el("button", { type: "button", class: "models-chip models-tray-open" });
    action.appendChild(icon(models.length > 1 ? "compare" : "table"));
    action.appendChild(
      el("span", {
        text:
          models.length > 1
            ? "Compare " + models.length + " models"
            : "Open " + models[0].name,
      })
    );
    action.addEventListener("click", revealSelection);
    tray.appendChild(action);
    tray.appendChild(
      el("button", { type: "button", class: "models-close", text: "Clear", "data-clear": "compare" })
    );
    tray.hidden = false;
    watchTarget();
    updateTrayReserve();
  }

  /* The tray floats over the foot of the page, so the reserve tracks its real
     rendered height — which changes with viewport width — and goes to zero
     once it has faded out (is-arrived) or there is nothing selected, rather
     than a fixed guess that is wrong at most widths. Set on the document, not
     the app root: the app is not the last element on the page, so its own
     padding cannot protect the site footer past the end of the article. */
  function updateTrayReserve() {
    var tray = state.nodes.tray;
    var active = Boolean(state.compare.length) && !tray.classList.contains("is-arrived");
    app.classList.toggle("has-tray", active);
    var height = active ? tray.getBoundingClientRect().height : 0;
    document.documentElement.style.setProperty(
      "--models-tray-reserve",
      height ? Math.ceil(height) + 24 + "px" : "0px"
    );
  }

  /* Dropping a chip removes the button focus was on, so focus moves to the
     chip that took its place, or to the tray's other controls once the last
     chip is gone. */
  function focusTrayAfterDrop(index) {
    var drops = state.nodes.tray.querySelectorAll(".models-tray-drop");
    var target =
      drops[index] ||
      drops[drops.length - 1] ||
      state.nodes.tray.querySelector(".models-tray-open") ||
      state.nodes.tray.querySelector(".models-close");
    if (target) {
      target.focus({ preventScroll: true });
    }
  }

  /* Hide the tray once the panel it points at is on screen. */
  function watchTarget() {
    if (typeof IntersectionObserver === "undefined") {
      return;
    }
    if (!state.trayWatcher) {
      state.trayWatcher = new IntersectionObserver(
        function (entries) {
          var arrived = entries.some(function (entry) {
            return entry.isIntersecting;
          });
          state.nodes.tray.classList.toggle("is-arrived", arrived);
          /* opacity:0 leaves the buttons in the tab order; inert removes them. */
          state.nodes.tray.inert = arrived;
          /* Arrival happens on scroll, outside any render() cycle, so the
             reserve has to be updated from here too or it outlives the tray
             that earned it. */
          updateTrayReserve();
        },
        { threshold: 0.15 }
      );
    }
    state.trayWatcher.disconnect();
    var target = state.compare.length > 1 ? state.nodes.compare : state.nodes.detail;
    if (!target.hidden) {
      state.trayWatcher.observe(target);
    }
  }

  function revealSelection() {
    var panel = state.compare.length > 1 ? state.nodes.compare : state.nodes.detail;
    if (!panel.scrollIntoView) {
      return;
    }
    var still =
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    panel.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "center" });
  }

  /* renderTray() (and the observer it arms in watchTarget()) has to run after
     the compare/detail panel's own hidden flag is finalised below, or the
     observer starts watching a target that is still hidden on the very
     render that makes it visible — the observer then never fires and the
     tray never learns to get out of the way. */
  function renderCompare() {
    paintCompare();
    renderTray();
  }

  function paintCompare() {
    var panel = state.nodes.compare;
    /* One selected model is not a comparison — it is that model's card. Opening
       it is toggleCompare's and the tray drop's job, at the moment the
       selection actually drops to one; doing it here too would fight a model
       opened directly by name or by chart dot on every re-render. */
    if (state.compare.length === 1) {
      panel.hidden = true;
      panel.textContent = "";
      return;
    }
    if (!state.compare.length) {
      panel.hidden = true;
      panel.textContent = "";
      return;
    }
    var models = state.compare
      .map(function (id) {
        return state.byId[id];
      })
      .filter(Boolean);

    /* Comparison is not the table: it shows every field the models can actually
       be compared on, in section order, and drops any field one of them is
       missing — a row with a dash in it compares nothing. */
    var fields = state.columns
      .slice()
      .sort(function (a, b) {
        return COLUMN_GROUPS.indexOf(a.group) - COLUMN_GROUPS.indexOf(b.group);
      })
      .map(function (column) {
        return column.key;
      });

    var grid = el("dl", { class: "models-compare-grid" });
    grid.style.gridTemplateColumns =
      "minmax(9rem, 1fr) repeat(" + models.length + ", minmax(10rem, 1fr))";

    grid.appendChild(el("dt", { class: "models-compare-head", text: "" }));
    models.forEach(function (model) {
      var head = el("dd", { class: "models-compare-head" });
      head.appendChild(columnByKey("name").compareCell(model));
      grid.appendChild(head);
    });

    /* A scalar cell is built once and kept: building it to test the text and
       then again to use it doubles the DOM work, images included. A multi-value
       column is rendered from the union below, so testing it only needs its
       list, never a node. */
    var built = {};
    var comparable = fields.filter(function (key) {
      if (key === "name") {
        return false;
      }
      var column = columnByKey(key);
      if (column.multi) {
        return models.every(function (model) {
          return column.options(model).length;
        });
      }
      if (column.compareByValue) {
        return models.every(function (model) {
          return Boolean(column.value(model));
        });
      }
      var cells = models.map(function (model) {
        return column.cell(model);
      });
      built[key] = cells;
      return cells.every(function (cell) {
        var text = cell.textContent.trim();
        return text !== "" && text !== "—";
      });
    });
    var skipped = fields.length - 1 - comparable.length;

    comparable.forEach(function (key) {
        var column = columnByKey(key);
        grid.appendChild(
          el("dt", { text: column.label, title: column.help || column.label })
        );
        var ranking = rank(column, models);
        /* A column can narrow its own multi-value list for the current view
           (Runs in, by the selected geography) without that narrowing
           changing what the column can be filtered or ranked on. */
        var optionsFor = column.narrowedOptions || column.options;
        var shared = column.multi ? sharedValues(column, models, optionsFor) : null;
        models.forEach(function (model, position) {
          var cell = el("dd", { "data-model": disambiguatedName(model) });
          if (shared) {
            /* Show the union, not each model's own list: what one model has and
               another does not is the comparison, and a shorter list next to a
               longer one does not say which entries are missing. */
            fillFromUnion(cell, column, model, shared, optionsFor);
          } else {
            var source = built[key] ? built[key][position] : column.cell(model);
            while (source.firstChild) {
              cell.appendChild(source.firstChild);
            }
          }
          var words = column.verdict || ["best", "lowest"];
          if (ranking.best.indexOf(position) !== -1) {
            cell.classList.add("is-best");
            cell.appendChild(icon("best", "models-verdict"));
            if (!column.booleanVerdict) {
              cell.appendChild(el("span", { class: "models-verdict-label", text: words[0] }));
            }
          } else if (ranking.worst.indexOf(position) !== -1) {
            cell.classList.add("is-worst");
            cell.appendChild(icon("worst", "models-verdict"));
            if (!column.booleanVerdict) {
              cell.appendChild(el("span", { class: "models-verdict-label", text: words[1] }));
            }
          }
          grid.appendChild(cell);
        });
      });

    panel.textContent = "";
    panel.setAttribute("role", "region");
    panel.setAttribute("aria-label", "Comparing " + models.length + " models");
    panel.appendChild(
      el("div", { class: "models-panel-head" }, [
        icon("compare"),
        el("h2", { text: "Comparing " + models.length + " models" }),
        el("button", {
          type: "button",
          class: "models-close",
          text: "Clear",
          "data-clear": "compare",
        }),
      ])
    );
    /* Not just .models-scroll: that class caps height for the vertical table
       scroller, which clips the comparison — a grid with no vertical overflow
       of its own — into a nested vertical scroller instead of the horizontal
       one it actually needs. */
    var gridWrap = el(
      "div",
      {
        class: "models-scroll models-compare-scroll",
        tabindex: "0",
        role: "group",
        "aria-label": "Comparison, scrollable",
      },
      [grid]
    );
    panel.appendChild(gridWrap);
    panel.appendChild(
      el("p", {
        class: "models-note",
        text:
          "Struck through: that model does not have it. A dot: only some of the compared " +
          "models do. The check or cross with a word names the best or worst value in the row.",
      })
    );
    if (skipped > 0) {
      panel.appendChild(
        el("p", {
          class: "models-note",
          text:
            skipped +
            " more fields are hidden: at least one of these models has no value for them.",
        })
      );
    }
    panel.hidden = false;
  }

  /* Every value any compared model has, and which of them all of them have.
     read defaults to column.options, but a column that narrows its list for
     the current view (Runs in) passes its narrowed reader instead. */
  function sharedValues(column, models, read) {
    var pick = read || column.options;
    var counts = {};
    models.forEach(function (model) {
      pick(model).forEach(function (value) {
        counts[value] = (counts[value] || 0) + 1;
      });
    });
    return {
      all: Object.keys(counts).sort(),
      common: Object.keys(counts).filter(function (value) {
        return counts[value] === models.length;
      }),
    };
  }

  function fillFromUnion(cell, column, model, shared, read) {
    cell.classList.add("models-tags", "models-tags--wrap");
    var owned = new Set((read || column.options)(model));
    var decorate = column.decorate || plainTag;
    shared.all.forEach(function (value) {
      var tag = decorate(value);
      if (!owned.has(value)) {
        tag.classList.add("is-absent");
        tag.setAttribute("title", model.name + " does not have " + value);
      } else if (shared.common.indexOf(value) === -1) {
        tag.classList.add("is-distinct");
        tag.setAttribute("title", "Only some of these models have " + value);
      }
      cell.appendChild(tag);
    });
    if (!shared.all.length) {
      cell.appendChild(el("span", { text: "—" }));
    }
  }

  /*
   * Ranks a comparison row. Only numeric rows can have a best and a worst, and
   * a row where every model ties has neither — highlighting a tie would invent
   * a difference that is not there.
   */
  function rank(column, models) {
    if (!column.numeric || models.length < 2) {
      return { best: [], worst: [] };
    }
    /* $0.14 per image is not cheaper than $5 per million tokens. A mixed-unit
       column has no cheapest, so it gets no verdict. */
    if (column.key === "price:primary") {
      var units = new Set(
        models.map(function (model) {
          var found = primaryPrice(model);
          return found ? found.unit : null;
        })
      );
      if (units.size > 1) {
        return { best: [], worst: [] };
      }
    }
    var values = models.map(function (model) {
      var value = valueOf(column.sort(model));
      return value === null || value === undefined ? null : value;
    });
    var present = values.filter(function (value) {
      return value !== null;
    });
    if (present.length < 2) {
      return { best: [], worst: [] };
    }
    var low = Math.min.apply(null, present);
    var high = Math.max.apply(null, present);
    if (low === high) {
      return { best: [], worst: [] };
    }
    var bestValue = column.betterIsLower ? low : high;
    var worstValue = column.betterIsLower ? high : low;
    return {
      best: indicesOf(values, bestValue),
      worst: indicesOf(values, worstValue),
    };
  }

  function indicesOf(values, target) {
    var out = [];
    values.forEach(function (value, index) {
      if (value === target) {
        out.push(index);
      }
    });
    return out;
  }

  /* -- detail ------------------------------------------------------------ */

  function openDetail(model, options) {
    if (!model) {
      return;
    }
    var scroll = !options || options.scroll !== false;
    state.openModel = model.id;
    var panel = state.nodes.detail;
    panel.textContent = "";

    panel.setAttribute("tabindex", "-1");
    panel.setAttribute("role", "region");
    panel.setAttribute("aria-label", model.name);
    var head = el("div", { class: "models-panel-head" });
    var logo = logoFor(model);
    if (logo) {
      head.appendChild(logo);
    }
    head.appendChild(el("h2", { text: model.name }));
    if (model.legacy) {
      head.appendChild(el("span", { class: "models-badge models-badge--legacy", text: "legacy" }));
    }
    head.appendChild(
      el("button", { type: "button", class: "models-close", text: "Close", "data-clear": "detail" })
    );
    panel.appendChild(head);
    panel.appendChild(el("p", { class: "models-id", text: model.id }));

    if (model.variants && model.variants.length) {
      /* This id is only one of the callable strings — a reader who only saw
         the row's own id would not know the other service takes a different
         one for the same model at the same price. */
      panel.appendChild(
        el("p", {
          class: "models-note",
          text: "Reached through more than one AWS service, by a different model value on each:",
        })
      );
      var variantList = el("ul", { class: "models-variants" });
      model.variants.forEach(function (variant) {
        var item = el("li", {});
        var variantLogo = logoImage(variant.service_logo, "models-logo--service");
        if (variantLogo) {
          item.appendChild(variantLogo);
        }
        item.appendChild(el("code", { class: "models-endpoint", text: variant.id }));
        item.appendChild(document.createTextNode(" via " + variant.service));
        variantList.appendChild(item);
      });
      panel.appendChild(variantList);
    }

    var prose = el("div", { class: "models-prose" });
    panel.appendChild(prose);
    panel.appendChild(factGrid(model));

    if (model.scores.length) {
      panel.appendChild(el("h3", { text: "Independent scores" }));
      panel.appendChild(scoreTable(model));
    }
    if (model.references.length) {
      var list = el("ul", {});
      model.references.forEach(function (reference) {
        var href = safeHref(reference.url);
        var item = el("li", {}, [
          href
            ? el("a", { href: href, rel: "nofollow noopener", target: "_blank", text: reference.label })
            : el("span", { text: reference.label }),
        ]);
        item.appendChild(document.createTextNode(" — " + reference.detail));
        list.appendChild(item);
      });
      panel.appendChild(el("h3", { text: "Further evaluations" }));
      panel.appendChild(list);
    }

    panel.appendChild(el("h3", { text: "Published AWS prices" }));
    var host = el("div", { "aria-live": "polite", text: "Loading…" });
    panel.appendChild(host);
    panel.hidden = false;

    loadDetail(model)
      .then(function (detail) {
        renderProse(prose, detail);
        host.textContent = "";
        var published = (detail.prices && detail.prices.prices) || [];
        var here = published.filter(function (row) {
          return row.region === state.region;
        });
        if (here.length) {
          host.appendChild(priceTable(detail.prices));
        } else if (published.length) {
          host.textContent =
            "AWS publishes no price for this model in " + state.region
            + ". Pick another region above to see where it is priced.";
        } else {
          host.textContent = "AWS publishes no prices for this model.";
        }
      })
      .catch(function () {
        host.textContent = "The model detail could not be loaded.";
      });

    if (scroll) {
      if (panel.scrollIntoView) {
        panel.scrollIntoView({ block: "nearest" });
      }
      panel.focus({ preventScroll: true });
    }
    syncUrl();
  }

  /* Closing returns focus to the row that opened it, not to the document top. */
  function closeDetail() {
    var opened = state.openModel;
    state.nodes.detail.hidden = true;
    state.openModel = null;
    /* A single tick is what opens the card, so leaving it ticked would re-open
       the card on the next render and make dismissing it look broken. */
    if (state.compare.length === 1 && state.compare[0] === opened) {
      state.compare = [];
      renderTray();
      var box = app.querySelector('[data-compare="' + opened.replace(/"/g, '\\"') + '"]');
      if (box) {
        box.checked = false;
      }
    }
    syncUrl();
    /* Opened from the table row or from a chart dot — both carry the id, and
       the table's stays in the DOM (just hidden) while the chart is showing,
       so the visible one has to be picked explicitly rather than by DOM order. */
    var escaped = opened ? opened.replace(/"/g, '\\"') : null;
    var trigger = null;
    if (escaped) {
      var candidates = [
        app.querySelector('[data-detail="' + escaped + '"]'),
        app.querySelector('[data-plot="' + escaped + '"]'),
      ];
      trigger = candidates.filter(function (node) {
        return node && node.getClientRects().length > 0;
      })[0];
    }
    if (trigger) {
      trigger.focus({ preventScroll: false });
    }
  }

  /*
   * Built from the column definitions rather than a hand-kept list, so the card
   * can never drift from the table: every column with a value for this model
   * appears, grouped the same way the column chooser groups them.
   */
  /* One request per model per visit: the documents are tens of kilobytes and
     the panel is rebuilt far more often than the model changes. */
  function loadDetail(model) {
    if (state.details[model.slug]) {
      return Promise.resolve(state.details[model.slug]);
    }
    return fetch(new URL("detail/" + model.slug + ".json", state.dataBase).href)
      .then(function (response) {
        return response.json();
      })
      .then(function (detail) {
        state.details[model.slug] = detail;
        return detail;
      });
  }

  function factGrid(model) {
    var host = el("div", { class: "models-facts-groups" });
    COLUMN_GROUPS.forEach(function (group) {
      var members = [];
      state.columns.forEach(function (column) {
        if (column.group !== group || column.key === "name") {
          return;
        }
        /* The card has room the table column does not, so a column that can
           say more than its cell does says it here. */
        var cell = (column.detail || column.cell)(model);
        var text = cell.textContent.trim();
        if (text !== "" && text !== "—") {
          members.push({ column: column, cell: cell });
        }
      });
      if (!members.length) {
        return;
      }
      host.appendChild(el("h3", { class: "models-facts-group", text: group }));
      var grid = el("dl", { class: "models-facts" });
      members.forEach(function (member) {
        var column = member.column;
        grid.appendChild(
          el("dt", { text: column.label, title: column.help || column.label })
        );
        var source = member.cell;
        var cell = el("dd", { class: source.className, title: source.getAttribute("title") });
        while (source.firstChild) {
          cell.appendChild(source.firstChild);
        }
        grid.appendChild(cell);
      });
      host.appendChild(grid);
    });
    return host;
  }

  /* Every AWS service this row answers on, the row's own first. */
  function servicesOf(model) {
    var names = [model.service];
    (model.variants || []).forEach(function (variant) {
      if (names.indexOf(variant.service) === -1) {
        names.push(variant.service);
      }
    });
    return names;
  }

  function serviceTag(model) {
    var tag = el("span", { class: "models-tag", title: model.service });
    var logo = logoImage(model.service_logo, "models-logo--service");
    if (logo) {
      tag.appendChild(logo);
    }
    tag.appendChild(el("span", { text: model.service || "—" }));
    return tag;
  }

  function plainTag(value) {
    return el("span", { class: "models-tag" }, [el("code", { class: "models-endpoint", text: value })]);
  }

  function renderProse(host, detail) {
    host.textContent = "";
    var blocks = [
      ["About this model", detail.description || detail.summary],
      ["What it is good at", detail.attributes],
      ["Use cases the vendor names", detail.use_cases],
      ["Languages", detail.languages],
    ].filter(function (pair) {
      return typeof pair[1] === "string" && pair[1].trim();
    });
    blocks.forEach(function (pair) {
      host.appendChild(el("h3", { text: pair[0] }));
      host.appendChild(el("p", { class: "models-prose-text", text: pair[1] }));
    });
    var policyHref = safeHref(detail.policy_url);
    if (policyHref) {
      host.appendChild(
        el("p", { class: "models-note" }, [
          el("a", {
            href: policyHref,
            rel: "nofollow noopener",
            target: "_blank",
            text: "Vendor model policy",
          }),
        ])
      );
    } else if (detail.policy_url) {
      /* AWS sometimes puts prose here instead of a URL — show it as text
         rather than a link that resolves to a nonsense path. */
      host.appendChild(el("p", { class: "models-note", text: detail.policy_url }));
    }
  }

  function scoreTable(model) {
    var body = el("tbody", {});
    model.scores.forEach(function (score) {
      var value = score.unit === "%" ? score.value.toFixed(1) + "%" : Math.round(score.value);
      var interval =
        score.ci_low !== null && score.ci_low !== undefined
          ? Math.round(score.ci_low) + "–" + Math.round(score.ci_high)
          : "—";
      body.appendChild(
        el("tr", {}, [
          el("td", { text: score.label }),
          el("td", { class: "models-num", text: String(value) }),
          el("td", { class: "models-num", text: interval }),
          el("td", { class: "models-num", text: score.samples ? String(score.samples) : "—" }),
          el("td", { text: score.matched_name }),
          el("td", { text: score.match_method }),
          el("td", { text: score.as_of }),
        ])
      );
    });
    return el("table", { class: "models-detail-table" }, [
      el("thead", {}, [
        el("tr", {}, [
          el("th", { scope: "col", text: "Leaderboard" }),
          el("th", { scope: "col", text: "Score" }),
          el("th", { scope: "col", text: "95% CI" }),
          el("th", { scope: "col", text: "Samples" }),
          el("th", { scope: "col", text: "Matched entry" }),
          el("th", { scope: "col", text: "Matched by" }),
          el("th", { scope: "col", text: "As of" }),
        ]),
      ]),
      body,
    ]);
  }

  /* A rate of 0.000000135 reads as a glitch; the table shows $/1M, so does this. */
  function perUnit(row) {
    var value = parseFloat(row.unit_price);
    var symbol = row.currency === "USD" ? "$" : row.currency + " ";
    var scaled = PRICE_SCALES[row.dimension];
    if (scaled && scaled > 1) {
      var suffix = scaled === 1e6 ? " / 1M" : " / 1K";
      return { text: symbol + formatPrice(value, scaled) + suffix };
    }
    return { text: symbol + formatPrice(value, 1) };
  }

  function priceTable(card) {
    var body = el("tbody", {});
    var rows = card.prices.filter(function (row) {
      return row.region === state.region;
    });
    rows.forEach(function (row) {
      var shown = perUnit(row);
      body.appendChild(
        el("tr", {}, [
          el("td", { text: row.dimension }),
          el("td", { text: row.tier }),
          el("td", { text: row.routing || "—" }),
          el("td", { text: row.spec || row.cache_ttl || row.context || "—" }),
          el("td", {
            class: "models-num",
            text: shown.text,
            title: row.unit_price + " " + row.currency + " per unit",
          }),
        ])
      );
    });
    var table = el("table", { class: "models-detail-table" }, [
      el("thead", {}, [
        el("tr", {}, [
          el("th", { scope: "col", text: "Dimension" }),
          el("th", { scope: "col", text: "Tier" }),
          el("th", { scope: "col", text: "Routing" }),
          el("th", { scope: "col", text: "Variant" }),
          el("th", { scope: "col", text: "Price" }),
        ]),
      ]),
      body,
    ]);
    var wrapper = el("div", {
      class: "models-scroll",
      tabindex: "0",
      role: "region",
      "aria-label": "Published prices, scrollable",
    }, [table]);
    var note = el("p", { class: "models-note" });
    note.textContent =
      rows.length +
      " published prices in " +
      state.region +
      ", per one billed unit. AWS publishes and bills them, not stdapi.ai.";
    var host = el("div", {}, [wrapper, note]);
    return host;
  }

  /* -- shareable state --------------------------------------------------- */

  function syncUrl() {
    if (!window.history || !window.history.replaceState) {
      return;
    }
    var params = new URLSearchParams();
    if (state.search) {
      params.set("q", state.search);
    }
    /* A geography moves these two on its own, so a link has to say outright
       when the reader set one of them instead — otherwise reopening the link
       hands the reader's own choice back to the geography. */
    if (state.pinned.region || state.region !== state.manifest.reference_region) {
      params.set("region", state.region);
    }
    if (state.sense !== "callable") {
      params.set("sense", state.sense);
    }
    if (state.tier !== "standard") {
      params.set("tier", state.tier);
    }
    if (state.pinned.routing || state.routing !== "any") {
      params.set("routing", state.routing);
    }
    if (state.buckets.size) {
      params.set("in", Array.from(state.buckets).join(","));
    }
    if (state.showLegacy) {
      params.set("legacy", "1");
    }
    if (state.onlyScored) {
      params.set("scored", "1");
    }
    if (state.start && presetSignature() !== state.startSignature) {
      state.start = null;
    }
    if (state.start) {
      params.set("start", state.start);
    }
    Object.keys(state.modalities).forEach(function (key) {
      if (state.modalities[key].size) {
        params.set(key === "input_modalities" ? "in_mod" : "out_mod",
          Array.from(state.modalities[key]).join(","));
      }
    });
    var picked = Object.keys(state.filters).filter(function (key) {
      return state.filters[key];
    });
    if (picked.length) {
      /* The Name and Regions filters are free text and can contain "," or
         ":" themselves — encoded per key/value so the structural "," between
         pairs and ":" between key and value stay unambiguous. */
      params.set(
        "where",
        picked
          .map(function (key) {
            return encodeURIComponent(key) + ":" + encodeURIComponent(state.filters[key]);
          })
          .join(",")
      );
    }
    /* Only the exact default — name, ascending — is worth leaving out; sorting
       by name descending is still a choice the URL has to carry. */
    if (state.sort && (state.sort.key !== "name" || state.sort.direction !== "ascending")) {
      params.set("sort", state.sort.key + (state.sort.direction === "descending" ? ":desc" : ""));
    }
    if (state.view !== "table") {
      params.set("view", state.view);
      if (state.chart.x && state.chart.y) {
        params.set("axes", state.chart.x + "|" + state.chart.y);
      }
    }
    /* Not scoped to chart view: the hidden-provider set is in-memory state
       that survives a switch back to the table, so a link copied from the
       table view has to carry it too or it silently reverts on load. */
    if (state.chart.hidden.size) {
      params.set("hide", Array.from(state.chart.hidden).join(","));
    }
    if (state.compare.length) {
      params.set("compare", state.compare.join(","));
    }
    if (state.openModel) {
      params.set("model", state.openModel);
    }
    var query = params.toString();
    window.history.replaceState(null, "", query ? "?" + query : window.location.pathname);
  }

  /*
   * A preset names a whole view, so the chip and the ?start= parameter only
   * still describe it while nothing has been refined since the click. Anything
   * the reader changes afterwards is theirs, and must survive being shared.
   */
  /*
   * Re-apply the parameters a shared link actually carried, over whatever a
   * preset in the same link set. Only keys the URL named are touched: an
   * absent key means "whatever the preset chose", not "the default".
   */
  function restoreExplicit(initial) {
    var byParam = {
      sort: function () {
        if (state.columnIndex[initial.sort.key]) {
          state.sort = initial.sort;
        }
      },
      where: function () {
        state.filters = initial.filters;
      },
      in_mod: function () {
        state.modalities.input_modalities = new Set(initial.inputModalities);
      },
      out_mod: function () {
        state.modalities.output_modalities = new Set(initial.outputModalities);
      },
      in: function () {
        state.buckets = new Set(initial.buckets);
      },
      sense: function () {
        state.sense = initial.sense;
      },
      tier: function () {
        state.tier = initial.tier;
      },
      routing: function () {
        state.routing = initial.routing;
      },
      legacy: function () {
        state.showLegacy = initial.showLegacy;
      },
      scored: function () {
        state.onlyScored = initial.onlyScored;
      },
    };
    Object.keys(byParam).forEach(function (name) {
      if (initial.present.has(name)) {
        byParam[name]();
      }
    });
  }

  function presetSignature() {
    return JSON.stringify([
      state.sort,
      state.filters,
      Array.from(state.modalities.input_modalities).sort(),
      Array.from(state.modalities.output_modalities).sort(),
      Array.from(state.buckets).sort(),
      state.sense,
      state.tier,
      state.routing,
      state.region,
      state.showLegacy,
      state.onlyScored,
    ]);
  }

  function readUrl() {
    var params = new URLSearchParams(window.location.search);
    return {
      search: (params.get("q") || "").toLowerCase(),
      region: params.get("region"),
      sense: params.get("sense") === "runs" ? "runs" : "callable",
      tier: params.get("tier") === "cheapest" ? "cheapest" : "standard",
      routing: ROUTING_CHOICES.some(function (choice) {
        return choice.value === params.get("routing");
      })
        ? String(params.get("routing"))
        : "any",
      buckets: (params.get("in") || "").split(",").filter(Boolean),
      showLegacy: params.get("legacy") === "1",
      onlyScored: params.get("scored") === "1",
      inputModalities: (params.get("in_mod") || "").split(",").filter(Boolean),
      outputModalities: (params.get("out_mod") || "").split(",").filter(Boolean),
      filters: readFilters(params.get("where")),
      sort: readSort(params.get("sort")),
      axes: (params.get("axes") || "").split("|"),
      hiddenProviders: (params.get("hide") || "").split(",").filter(Boolean),
      view: params.get("view") === "chart" ? "chart" : "table",
      compare: (params.get("compare") || "").split(",").filter(Boolean),
      model: params.get("model"),
      start: params.get("start"),
      present: new Set(Array.from(params.keys())),
    };
  }

  function readFilters(value) {
    var out = {};
    (value || "").split(",").forEach(function (pair) {
      var at = pair.indexOf(":");
      if (at > 0) {
        try {
          out[decodeURIComponent(pair.slice(0, at))] = decodeURIComponent(pair.slice(at + 1));
        } catch (error) {
          /* Malformed percent-encoding in a hand-edited URL — drop this one
             pair rather than throw and lose every filter after it. */
        }
      }
    });
    return out;
  }

  function readSort(value) {
    if (!value) {
      return null;
    }
    var descending = /:desc$/.test(value);
    return {
      key: value.replace(/:desc$/, ""),
      direction: descending ? "descending" : "ascending",
    };
  }

  /* -- toolbar ----------------------------------------------------------- */

  function buildStartingPoints() {
    var bar = el("div", { class: "models-starts" });
    bar.appendChild(el("span", { class: "models-starts-title", text: "Start with" }));
    STARTING_POINTS.forEach(function (preset) {
      bar.appendChild(
        el("button", {
          type: "button",
          class: "models-chip models-start",
          text: preset.label,
          title: preset.hint,
          "data-start": preset.key,
        })
      );
    });
    bar.addEventListener("click", function (event) {
      var button = event.target.closest ? event.target.closest("[data-start]") : null;
      if (!button) {
        return;
      }
      var key = button.getAttribute("data-start");
      /* A second click on the pressed preset is off, not a re-apply — it
         announces as a toggle, so it has to behave like one. */
      if (state.start === key) {
        resetFilters();
        announce(button.textContent + " cleared");
        return;
      }
      var preset = STARTING_POINTS.filter(function (item) {
        return item.key === key;
      })[0];
      if (!preset) {
        return;
      }
      resetFilters({ silent: true });
      state.start = key;
      preset.apply();
      followGeography();
      state.startSignature = presetSignature();
      state.headSignature = null;
      syncControls();
      render();
      announce(button.textContent + " selected");
    });
    return bar;
  }

  function buildToolbar() {
    var toolbar = el("div", { class: "models-toolbar" });

    var search = el("label", { class: "models-search-wrap" }, [icon("search")]);
    var input = el("input", {
      type: "search",
      class: "models-search",
      placeholder: "Search name, ID, provider, route…",
      "aria-label": "Search models",
      value: state.search,
    });
    /* Every keystroke otherwise rebuilds ~3,400 nodes, rewrites the URL and,
       with one model ticked, refetches its card. One render per pause. The
       timer lives on state, not in this closure, so init() can cancel it on
       the next page instant-navigates to before it fires against stale data. */
    input.addEventListener("input", function () {
      state.search = input.value.trim().toLowerCase();
      window.clearTimeout(state.searchTimer);
      state.searchTimer = window.setTimeout(render, 120);
    });
    search.appendChild(input);
    state.nodes.search = input;
    toolbar.appendChild(search);

    var refine = el("div", { class: "models-refine" });
    refine.appendChild(
      labelled(
        "Provider",
        select(
          [{ value: "", label: "any" }].concat(
            unique(
              state.catalog.models.map(function (model) {
                return model.provider;
              })
            ).map(function (name) {
              return { value: name.toLowerCase(), label: name };
            })
          ),
          state.filters.provider || "",
          function (value) {
            state.filters.provider = value;
            render();
          },
          "provider"
        )
      )
    );

    var openWeightsColumn = columnByKey("open_weights");
    if (openWeightsColumn) {
      refine.appendChild(labelled("Open weights", filterControl(openWeightsColumn, "in the Filters panel")));
    }
    var toolCallColumn = columnByKey("tool_call");
    if (toolCallColumn) {
      refine.appendChild(labelled("Tools", filterControl(toolCallColumn, "in the Filters panel")));
    }

    refine.appendChild(modalityPicker("input_modalities", "Accepts"));
    refine.appendChild(modalityPicker("output_modalities", "Produces"));

    /* Toggling a bucket chip, whether it lives in the geography group or
       stands alone as "Global routing". */
    function onBucketClick(event) {
      var chip = event.target.closest ? event.target.closest("[data-bucket]") : null;
      if (!chip) {
        return;
      }
      var bucket = chip.getAttribute("data-bucket");
      if (state.buckets.has(bucket)) {
        state.buckets.delete(bucket);
        chip.setAttribute("aria-pressed", "false");
      } else {
        state.buckets.add(bucket);
        chip.setAttribute("aria-pressed", "true");
      }
      followGeography();
      syncControls();
      render();
    }
    function bucketChip(bucket) {
      return el("button", {
        type: "button",
        class: "models-chip",
        text: bucket.label,
        title: bucket.hint,
        "aria-pressed": state.buckets.has(bucket.key) ? "true" : "false",
        "data-bucket": bucket.key,
      });
    }

    var globalChip = bucketChip(GLOBAL_BUCKET);
    globalChip.addEventListener("click", onBucketClick);
    refine.appendChild(globalChip);

    var chips = el("div", { class: "models-chips", role: "group", "aria-label": "Geography" });
    chips.appendChild(icon("region", "models-chips-icon"));
    chips.appendChild(el("span", { class: "models-modalities-title", text: "Geography" }));
    GEO_BUCKETS.forEach(function (bucket) {
      chips.appendChild(bucketChip(bucket));
    });
    chips.addEventListener("click", onBucketClick);
    refine.appendChild(chips);

    /* Right beside the chips it narrows, since it decides what "EU" means:
       callable from, or actually running in. */
    refine.appendChild(
      labelled(
        "Regions",
        select(
          [
            { value: "callable", label: "where I can call it" },
            { value: "runs", label: "where it runs" },
          ],
          state.sense,
          function (value) {
            state.sense = value;
            render();
          },
          "sense"
        )
      )
    );

    refine.appendChild(
      labelled(
        "Price shown",
        select(
          [
            { value: "standard", label: "on-demand" },
            { value: "cheapest", label: "cheapest tier reachable by request" },
          ],
          state.tier,
          function (value) {
            state.tier = value;
            render();
          },
          "tier"
        )
      )
    );

    var routing = labelled(
      "Routing",
      select(
        ROUTING_CHOICES.map(function (choice) {
          return { value: choice.value, label: choice.label };
        }),
        state.routing,
        function (value) {
          state.routing = value;
          state.pinned.routing = true;
          render();
        },
        "routing"
      )
    );
    routing.setAttribute("title", routingChoice().hint);
    refine.appendChild(routing);

    /* Routing is a data-residency decision before it is a price one, and the
       two answers differ, so the control says where the rules are written. */
    var residency = el("p", { class: "models-note models-residency" });
    residency.appendChild(
      document.createTextNode(
        "Routing decides where your request is processed, and the gateway can "
        + "hold every request to whichever of these you pick. "
      )
    );
    residency.appendChild(
      el("a", {
        href: "../operations_compliance/#cross-region-inference-profiles-and-data-geography",
        text: "Data sovereignty & compliance",
      })
    );
    residency.appendChild(document.createTextNode("."));
    refine.appendChild(residency);

    refine.appendChild(
      labelled(
        "Prices in",
        select(
          offeredRegions().map(function (name) {
            return { value: name, label: name };
          }),
          state.region,
          function (value) {
            state.region = value;
            state.pinned.region = true;
            render();
          },
          "region"
        )
      )
    );

    var legacyCount = state.catalog.models.filter(function (model) {
      return model.legacy;
    }).length;
    var legacy = el("label", {
      class: "models-toggle",
      title:
        "Models AWS has marked legacy, or has stopped listing. Still callable "
        + "while AWS serves them, and kept here so one you already run stays findable.",
    });
    var legacyBox = el("input", { type: "checkbox" });
    legacyBox.checked = state.showLegacy;
    legacyBox.addEventListener("change", function () {
      state.showLegacy = legacyBox.checked;
      render();
    });
    legacy.appendChild(legacyBox);
    legacy.appendChild(document.createTextNode("Show legacy (" + legacyCount + ")"));
    refine.appendChild(legacy);
    state.nodes.legacy = legacyBox;

    /* Matches the "with a public benchmark score" stat tile, which counts
       the default (non-legacy) view rather than the whole catalogue. */
    var scoredCount = state.catalog.models.filter(function (model) {
      return model.scores.length && !model.legacy && !model.retired;
    }).length;
    var scored = el("label", {
      class: "models-toggle",
      title:
        "Hide models no public leaderboard has an entry for. A blank benchmark "
        + "is a missing measurement, not a bad one, so this narrows the table "
        + "rather than ranking it.",
    });
    var scoredBox = el("input", { type: "checkbox" });
    scoredBox.checked = state.onlyScored;
    scoredBox.addEventListener("change", function () {
      state.onlyScored = scoredBox.checked;
      render();
    });
    scored.appendChild(scoredBox);
    scored.appendChild(document.createTextNode("Benchmarked only (" + scoredCount + ")"));
    refine.appendChild(scored);
    state.nodes.scored = scoredBox;
    state.nodes.refine = refine;

    var views = el("div", { class: "models-views", role: "group", "aria-label": "View" });
    [
      { key: "table", label: "Table", glyph: "table" },
      { key: "chart", label: "Chart", glyph: "chart" },
    ].forEach(function (view) {
      var button = el("button", {
        type: "button",
        class: "models-chip",
        "aria-pressed": state.view === view.key ? "true" : "false",
        "data-view": view.key,
        title: "Show the " + view.label.toLowerCase() + " view",
      });
      button.appendChild(icon(view.glyph));
      button.appendChild(el("span", { text: view.label }));
      views.appendChild(button);
    });
    views.addEventListener("click", function (event) {
      var button = event.target.closest ? event.target.closest("[data-view]") : null;
      if (!button) {
        return;
      }
      state.view = button.getAttribute("data-view");
      Array.prototype.forEach.call(views.children, function (child) {
        child.setAttribute("aria-pressed", child === button ? "true" : "false");
      });
      render();
    });
    toolbar.appendChild(views);

    var reset = el("button", { type: "button", class: "models-chip", title: "Clear every filter" });
    reset.appendChild(icon("reset"));
    reset.appendChild(el("span", { text: "Reset" }));
    reset.addEventListener("click", function () {
      resetFilters();
    });
    toolbar.appendChild(reset);

    toolbar.appendChild(el("span", { class: "models-count", text: "" }));
    state.nodes.count = toolbar.querySelector(".models-count");
    return toolbar;
  }

  /* Ten controls in a row is a wall. The two everyone uses stay out; the rest
     fold away behind a count of how many are doing something. */
  function buildRefinements() {
    /* A shared URL with a filter applied should not also require the reader
       to discover the disclosure hiding it before they can see why. */
    var details = el("details", {
      class: "models-columns models-refine-panel",
      open: activeRefinements() ? "open" : null,
    });
    var summary = el("summary", {});
    summary.appendChild(icon("region"));
    summary.appendChild(el("span", { text: "Filters" }));
    summary.appendChild(el("span", { class: "models-refine-count", text: "" }));
    details.appendChild(summary);
    details.appendChild(state.nodes.refine);
    state.nodes.refineCount = summary.querySelector(".models-refine-count");
    return details;
  }

  function activeRefinements() {
    var active = state.buckets.size + state.modalities.input_modalities.size;
    active += state.modalities.output_modalities.size;
    /* An unknown key from a hand-edited URL (?where=constructor:x) matches no
       column and no control can clear it — it should not count as an active
       filter or force the panel open for a filter nobody can see or act on. */
    active += Object.keys(state.filters).filter(function (key) {
      return state.filters[key] && columnByKey(key);
    }).length;
    if (state.showLegacy) {
      active += 1;
    }
    if (state.onlyScored) {
      active += 1;
    }
    if (state.region !== state.manifest.reference_region) {
      active += 1;
    }
    if (state.tier !== "standard") {
      active += 1;
    }
    if (state.routing !== "any") {
      active += 1;
    }
    if (state.sense !== "callable") {
      active += 1;
    }
    return active;
  }

  function resetFilters(options) {
    state.start = null;
    state.onlyScored = false;
    state.modalities = { input_modalities: new Set(), output_modalities: new Set() };
    state.search = "";
    state.filters = {};
    state.buckets = new Set();
    state.showLegacy = false;
    state.compare = [];
    /* A reset that leaves a card open, or the chart showing, is not a reset. */
    state.openModel = null;
    state.nodes.detail.hidden = true;
    state.nodes.detail.textContent = "";
    state.view = "table";
    state.sense = "callable";
    state.tier = "standard";
    state.routing = "any";
    state.region = state.manifest.reference_region;
    state.pinned = { routing: false, region: false };
    state.sort = { key: "name", direction: "ascending" };
    state.chart.hidden = new Set();
    state.columns.forEach(function (column) {
      column.visible = state.defaultColumns.indexOf(column.key) !== -1;
    });
    state.headSignature = null;
    syncControls();
    if (!options || !options.silent) {
      render();
    }
  }

  /*
   * Every control reads its value back out of the state it sets, so a preset or
   * a reset that changes state cannot leave a chip unpressed or a dropdown
   * naming a region whose prices are not the ones on screen.
   */
  function syncControls() {
    var bound = {
      provider: state.filters.provider || "",
      sense: state.sense,
      tier: state.tier,
      routing: state.routing,
      region: state.region,
    };
    Array.prototype.forEach.call(app.querySelectorAll("[data-select]"), function (node) {
      var name = node.getAttribute("data-select");
      if (bound[name] !== undefined) {
        node.value = bound[name];
      }
      if (name === "routing" && node.parentNode) {
        // The hint explains the chosen option, so it moves with the choice.
        node.parentNode.setAttribute("title", routingChoice().hint);
      }
      if (name === "region") {
        // A geography narrows which regions can be quoted, and the control
        // has to stop offering the ones it rules out.
        var offered = offeredRegions();
        var listed = Array.prototype.map.call(node.options, function (option) {
          return option.value;
        });
        if (offered.join(",") !== listed.join(",")) {
          node.textContent = "";
          offered.forEach(function (region) {
            node.appendChild(el("option", { value: region, text: region }));
          });
        }
        node.value = state.region;
      }
    });
    /* The toolbar dropdown and the in-table column filter both write the same
       state.filters entry, so both have to read it back or one goes stale
       the moment the other one changes. */
    Array.prototype.forEach.call(app.querySelectorAll("[data-filter]"), function (node) {
      var value = state.filters[node.getAttribute("data-filter")] || "";
      if (node.value !== value) {
        node.value = value;
      }
    });
    Array.prototype.forEach.call(app.querySelectorAll("[data-column]"), function (box) {
      var column = columnByKey(box.getAttribute("data-column"));
      box.checked = Boolean(column && column.visible);
    });
    Array.prototype.forEach.call(app.querySelectorAll("[data-bucket]"), function (chip) {
      chip.setAttribute(
        "aria-pressed",
        state.buckets.has(chip.getAttribute("data-bucket")) ? "true" : "false"
      );
    });
    Array.prototype.forEach.call(app.querySelectorAll("[data-modality]"), function (chip) {
      var parts = chip.getAttribute("data-modality").split(":");
      chip.setAttribute(
        "aria-pressed",
        state.modalities[parts[0]] && state.modalities[parts[0]].has(parts[1])
          ? "true"
          : "false"
      );
    });
    Array.prototype.forEach.call(app.querySelectorAll("[data-view]"), function (button) {
      button.setAttribute(
        "aria-pressed",
        button.getAttribute("data-view") === state.view ? "true" : "false"
      );
    });
    Array.prototype.forEach.call(app.querySelectorAll("[data-start]"), function (chip) {
      chip.setAttribute(
        "aria-pressed",
        state.start === chip.getAttribute("data-start") ? "true" : "false"
      );
    });
    if (state.nodes.search) {
      state.nodes.search.value = state.search;
    }
    if (state.nodes.legacy) {
      state.nodes.legacy.checked = state.showLegacy;
    }
    if (state.nodes.scored) {
      state.nodes.scored.checked = state.onlyScored;
    }
  }

  /*
   * Modalities are a conjunction, not a choice: asking for image and text input
   * means both, so a text-only model is not an answer. Nothing selected means
   * no constraint.
   */
  function modalityPicker(key, label) {
    var values = unique(
      state.catalog.models.reduce(function (all, model) {
        return all.concat(model[key] || []);
      }, [])
    );
    var wrap = el("div", { class: "models-modalities", role: "group", "aria-label": label });
    wrap.appendChild(el("span", { class: "models-modalities-title", text: label }));
    values.forEach(function (value) {
      var chosen = state.modalities[key].has(value);
      var chip = el("button", {
        type: "button",
        class: "models-chip models-chip--tiny",
        "aria-pressed": chosen ? "true" : "false",
        "data-modality": key + ":" + value,
        title: label + " " + value.toLowerCase(),
      });
      if (MODALITY_ICONS[value]) {
        chip.appendChild(glyph(MODALITY_ICONS[value]));
      }
      chip.appendChild(el("span", { text: value.charAt(0) + value.slice(1).toLowerCase() }));
      wrap.appendChild(chip);
    });
    wrap.addEventListener("click", function (event) {
      var chip = event.target.closest ? event.target.closest("[data-modality]") : null;
      if (!chip) {
        return;
      }
      var parts = chip.getAttribute("data-modality").split(":");
      var chosen = state.modalities[parts[0]];
      if (chosen.has(parts[1])) {
        chosen.delete(parts[1]);
        chip.setAttribute("aria-pressed", "false");
      } else {
        chosen.add(parts[1]);
        chip.setAttribute("aria-pressed", "true");
      }
      render();
    });
    return wrap;
  }

  function labelled(text, control) {
    var wrap = el("label", { class: "models-labelled" });
    wrap.appendChild(el("span", { text: text }));
    wrap.appendChild(control);
    return wrap;
  }

  function select(options, current, onChange, bind) {
    var node = el("select", { class: "models-filter" });
    if (bind) {
      node.setAttribute("data-select", bind);
    }
    options.forEach(function (option) {
      var item = el("option", { value: option.value, text: option.label });
      if (option.value === current) {
        item.setAttribute("selected", "selected");
      }
      node.appendChild(item);
    });
    node.addEventListener("change", function () {
      onChange(node.value);
    });
    return node;
  }

  function buildColumnChooser() {
    var details = el("details", { class: "models-columns" });
    var summary = el("summary", {});
    summary.appendChild(icon("columns"));
    summary.appendChild(el("span", { text: "Columns" }));
    summary.appendChild(el("span", { class: "models-refine-count", hidden: "hidden", text: "" }));
    details.appendChild(summary);
    var body = el("div", { class: "models-columns-body" });
    COLUMN_GROUPS.forEach(function (group) {
      /* The name column has no checkbox, the same way the Compare column has
         none: unlike a hidden Price or Regions column, a hidden name column
         leaves every row anonymous with nothing else to identify it by. */
      var members = state.columns.filter(function (column) {
        return column.group === group && column.key !== "name";
      });
      if (!members.length) {
        return;
      }
      /* The first heading the app renders below the page's own H1, so it has
         to open at H2 — the model card's fact groups nest under the card's
         own H2 instead, which is why those are H3. */
      body.appendChild(el("h2", { class: "models-columns-group", text: group }));
      body.appendChild(columnGrid(members));
    });
    details.appendChild(body);
    return details;
  }

  function columnGrid(columns) {
    var grid = el("div", { class: "models-columns-grid" });
    columns.forEach(function (column) {
      var entry = el("div", { class: "models-columns-entry" });
      var label = el("label", {});
      var box = el("input", { type: "checkbox", "data-column": column.key });
      box.checked = column.visible;
      box.addEventListener("change", function () {
        column.visible = box.checked;
        render();
      });
      label.appendChild(box);
      label.appendChild(document.createTextNode(column.label));
      entry.appendChild(label);
      if (column.help) {
        entry.appendChild(el("span", { class: "models-columns-help", text: column.help }));
      }
      grid.appendChild(entry);
    });
    return grid;
  }

  function buildChartControls() {
    var bar = el("div", { class: "models-chart-controls" });
    var options = plottableColumns().map(function (column) {
      return { value: column.key, label: column.label };
    });
    bar.appendChild(
      labelled(
        "X axis",
        select(options, state.chart.x, function (value) {
          state.chart.x = value;
          render();
        })
      )
    );
    bar.appendChild(
      labelled(
        "Y axis",
        select(options, state.chart.y, function (value) {
          state.chart.y = value;
          render();
        })
      )
    );
    return bar;
  }

  /* -- wiring ------------------------------------------------------------ */

  function wireTable(table) {
    table.addEventListener("click", function (event) {
      var target = event.target.closest ? event.target.closest("[data-sort],[data-detail]") : null;
      if (!target) {
        return;
      }
      var sortKey = target.getAttribute("data-sort");
      if (sortKey) {
        var column = columnByKey(sortKey);
        var current = state.sort && state.sort.key === sortKey ? state.sort.direction : null;
        /* A first click sorts best-first: cheapest price, highest Elo, lowest
           word error rate; a text column sorts A–Z. */
        var direction =
          column.bestFirst || (column.numeric && !column.betterIsLower ? "descending" : "ascending");
        if (current === "ascending") {
          direction = "descending";
        } else if (current === "descending") {
          direction = "ascending";
        }
        state.sort = { key: sortKey, direction: direction };
        render();
        return;
      }
      openDetail(state.byId[target.getAttribute("data-detail")]);
    });

    /* Compare checkboxes only exist in the table body, so this stays scoped
       to it — [data-filter] is wired on the app in wireFilters() instead,
       since the same control also appears in the Filters panel. */
    table.addEventListener("change", function (event) {
      var target = event.target;
      var compareId = target.getAttribute && target.getAttribute("data-compare");
      if (compareId) {
        toggleCompare(compareId, target);
      }
    });
  }

  /* Delegated on the app, not the table, because filterControl() is reused
     to build the same [data-filter] control in the Filters panel — a select
     or text input bound to state.filters wherever it is rendered. */
  function wireFilters(root) {
    root.addEventListener("change", function (event) {
      var target = event.target;
      var filterKey = target.getAttribute && target.getAttribute("data-filter");
      if (filterKey) {
        state.filters[filterKey] = target.value.trim().toLowerCase();
        render();
      }
    });
    root.addEventListener("input", function (event) {
      var filterKey = event.target.getAttribute && event.target.getAttribute("data-filter");
      if (filterKey && event.target.tagName === "INPUT") {
        state.filters[filterKey] = event.target.value.trim().toLowerCase();
        window.clearTimeout(state.filterTimer);
        state.filterTimer = window.setTimeout(render, 120);
      }
    });
  }

  /* Assigning the same text twice is not re-announced, so the clear and the
     message are separated by a turn of the event loop. */
  function announce(message) {
    var live = state.nodes.live;
    live.textContent = "";
    window.setTimeout(function () {
      live.textContent = message;
    }, 50);
  }

  function warn(message) {
    announce(message);
    state.nodes.live.classList.add("is-warning");
    window.setTimeout(function () {
      state.nodes.live.classList.remove("is-warning");
    }, 6000);
  }

  function toggleCompare(id, checkbox) {
    var index = state.compare.indexOf(id);
    var hadNone = state.compare.length === 0;
    if (checkbox.checked && index === -1) {
      if (state.compare.length >= COMPARE_LIMIT) {
        checkbox.checked = false;
        warn("Only " + COMPARE_LIMIT + " models can be compared at once — remove one first.");
        return;
      }
      state.compare.push(id);
    } else if (!checkbox.checked && index !== -1) {
      state.compare.splice(index, 1);
    }
    if (state.compare.length === 1) {
      /* A tick that picks the first model scrolls its card into view — a tick
         has to do something visible. Dropping back down to one from a
         comparison does not scroll: the reader is already looking at it. */
      openDetail(state.byId[state.compare[0]], { scroll: hadNone });
    }
    renderCompare();
    syncUrl();
  }

  function mount(catalog) {
    var initial = readUrl();
    /* Object.create(null) so a URL parameter naming an Object.prototype member
       (?region=constructor) can never resolve to an inherited value instead
       of a real lookup miss. */
    var regionIndex = Object.create(null);
    catalog.manifest.regions.forEach(function (name, index) {
      regionIndex[name] = index;
    });
    var byId = Object.create(null);
    var nameCounts = Object.create(null);
    catalog.models.forEach(function (model) {
      byId[model.id] = model;
      nameCounts[model.name] = (nameCounts[model.name] || 0) + 1;
    });
    /* Two model versions can share a display name (Nova Reel v1:0/v1:1, both
       gpt-oss sizes under two ids) — anywhere only the name is shown for more
       than one model at once, that is not enough to tell them apart. */
    var duplicateNames = new Set(
      Object.keys(nameCounts).filter(function (name) {
        return nameCounts[name] > 1;
      })
    );

    state = {
      catalog: catalog,
      manifest: catalog.manifest,
      regionIndex: regionIndex,
      byId: byId,
      duplicateNames: duplicateNames,
      dataBase: state.dataBase,
      assetBase: state.assetBase,
      region:
        initial.region && regionIndex[initial.region] !== undefined
          ? initial.region
          : catalog.manifest.reference_region,
      sense: initial.sense,
      tier: initial.tier,
      routing: initial.routing,
      /* A link that names routing or a region carries a reader's own choice,
         which a geography selected in the same link must not overwrite. */
      pinned: {
        routing: initial.present.has("routing"),
        region: initial.present.has("region"),
      },
      search: initial.search,
      buckets: new Set(initial.buckets),
      filters: initial.filters,
      compare: initial.compare
        .filter(function (id) {
          return byId[id];
        })
        .slice(0, COMPARE_LIMIT),
      showLegacy: initial.showLegacy,
      onlyScored: initial.onlyScored,
      start: null,
      modalities: {
        input_modalities: new Set(initial.inputModalities),
        output_modalities: new Set(initial.outputModalities),
      },
      view: initial.view,
      openModel: null,
      details: {},
      haystacks: {},
      sort: { key: "name", direction: "ascending" },
      nodes: {},
    };
    state.columns = buildColumns(catalog);
    state.columnIndex = state.columns.reduce(function (all, column) {
      all[column.key] = column;
      return all;
    }, Object.create(null));
    if (initial.sort && state.columnIndex[initial.sort.key]) {
      state.sort = initial.sort;
    }
    state.defaultColumns = state.columns
      .filter(function (column) {
        return column.visible;
      })
      .map(function (column) {
        return column.key;
      });
    /* A shared preset link has to reproduce what the click does — the pressed
       chip and, since apply() can set its own sort or filters, the same
       final state a fresh click on it would leave. */
    if (initial.start) {
      var startPreset = STARTING_POINTS.filter(function (item) {
        return item.key === initial.start;
      })[0];
      if (startPreset) {
        state.start = initial.start;
        startPreset.apply();
        followGeography();
        state.startSignature = presetSignature();
        // Whatever the link states outright is the reader's own refinement of
        // the preset, so it is applied over the top rather than discarded.
        restoreExplicit(initial);
      }
    }
    /* Again, because the link's own geography lands after the preset's. */
    followGeography();
    var plottable = plottableColumns();
    /* The question the chart answers by default is "what do I get for the
       money", so it opens on price against the broadest benchmark. */
    var byKey = function (wanted) {
      return plottable.filter(function (column) {
        return column.key === wanted;
      })[0];
    };
    var firstScore = plottable.filter(function (column) {
      return column.key.indexOf("score:") === 0;
    })[0];
    state.chart = {
      hidden: new Set(initial.hiddenProviders),
      known: [],
      x:
        (byKey(initial.axes[0]) || byKey("price:primary") || byKey("price:input_tokens")
          || plottable[0] || {}).key,
      y: (byKey(initial.axes[1]) || byKey("score:lmarena:text") || firstScore
        || plottable[1] || plottable[0] || {}).key,
      highlight: [],
    };

    var head = el("thead", {});
    var body = el("tbody", {});
    var caption = el("caption", { class: "models-sr", text: "Model table" });
    var table = el("table", { class: "models-table" }, [caption, head, body]);
    state.nodes.head = head;
    state.nodes.body = body;
    state.nodes.stats = el("div", { class: "models-stats" });
    state.nodes.scroll = el("div", {
      class: "models-scroll",
      tabindex: "0",
      role: "region",
      "aria-label": "Model table",
    }, [table]);
    state.nodes.scrollHint = el("p", {
      class: "models-scroll-hint",
      text: "Scroll sideways for more columns, or choose them above.",
    });
    state.nodes.tableWrap = el("div", { class: "models-table-wrap" }, [
      state.nodes.scrollHint,
      state.nodes.scroll,
    ]);
    state.nodes.chart = el("div", { class: "models-chart" });
    state.nodes.tip = el("div", { class: "models-tip", hidden: "hidden", role: "tooltip" });
    state.nodes.chartNote = el("p", { class: "models-note", id: "models-chart-note" });
    state.nodes.chartWrap = el("div", { hidden: "hidden" }, [
      buildChartControls(),
      state.nodes.chart,
    ]);
    state.nodes.compare = el("div", { class: "models-panel", hidden: "hidden" });
    state.nodes.detail = el("div", { class: "models-panel", hidden: "hidden" });
    state.nodes.live = el("p", { class: "models-live", role: "status", "aria-live": "polite" });
    state.nodes.tray = el("div", {
      class: "models-tray",
      hidden: "hidden",
      role: "region",
      "aria-label": "Selected models",
    });

    app.textContent = "";
    app.appendChild(state.nodes.stats);
    app.appendChild(buildStartingPoints());
    app.appendChild(buildToolbar());
    var panels = el("div", { class: "models-panels" });
    panels.appendChild(buildRefinements());
    state.nodes.chooser = buildColumnChooser();
    panels.appendChild(state.nodes.chooser);
    app.appendChild(panels);
    app.appendChild(state.nodes.tableWrap);
    app.appendChild(state.nodes.chartWrap);
    app.appendChild(state.nodes.compare);
    app.appendChild(state.nodes.detail);
    app.appendChild(state.nodes.live);
    app.appendChild(state.nodes.tray);
    app.hidden = false;

    wireTable(table);
    wireFilters(app);
    state.nodes.chart.addEventListener("pointerover", showTip);
    state.nodes.chart.addEventListener("pointerout", hideTip);
    state.nodes.chart.addEventListener("focusin", showTip);
    state.nodes.chart.addEventListener("focusout", hideTip);
    state.nodes.chart.addEventListener("keydown", onDotKey);
    state.nodes.scroll.addEventListener("scroll", updateScrollHints);

    app.addEventListener("click", function (event) {
      var clear = event.target.closest ? event.target.closest("[data-clear]") : null;
      if (clear) {
        if (clear.getAttribute("data-clear") === "compare") {
          /* A single selection's card is what "Clear" is dismissing too, or
             it survives the clear and keeps ?model= in the URL. */
          if (state.compare.length === 1 && state.openModel === state.compare[0]) {
            closeDetail();
          }
          state.compare = [];
          render();
        } else {
          closeDetail();
        }
        return;
      }
      var drop = event.target.closest ? event.target.closest("[data-drop]") : null;
      if (drop) {
        var dropIndex = Array.prototype.indexOf.call(
          state.nodes.tray.querySelectorAll(".models-tray-drop"),
          drop
        );
        var dropped = state.compare.indexOf(drop.getAttribute("data-drop"));
        if (dropped !== -1) {
          state.compare.splice(dropped, 1);
        }
        if (state.compare.length === 1) {
          openDetail(state.byId[state.compare[0]], { scroll: false });
        }
        render();
        focusTrayAfterDrop(dropIndex);
        return;
      }
      var bulk = event.target.closest ? event.target.closest("[data-providers]") : null;
      if (bulk) {
        setProviders(bulk.getAttribute("data-providers") === "all");
        return;
      }
      var highlight = event.target.closest ? event.target.closest("[data-highlight]") : null;
      if (highlight) {
        toggleHighlight(highlight.getAttribute("data-highlight"));
        return;
      }
      var dot = event.target.closest ? event.target.closest("[data-plot]") : null;
      if (dot) {
        openDetail(state.byId[dot.getAttribute("data-plot")]);
      }
    });

    render();
    if (initial.model && byId[initial.model]) {
      openDetail(byId[initial.model]);
    } else if (state.compare.length === 1) {
      openDetail(state.byId[state.compare[0]], { scroll: false });
    }
  }

  /* Delegated on the chart host rather than attached per dot — up to 142
     listeners collapse into one, matching the pointer/focus handlers below. */
  function onDotKey(event) {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    var dot = event.target.closest ? event.target.closest("[data-plot]") : null;
    if (!dot) {
      return;
    }
    event.preventDefault();
    openDetail(state.byId[dot.getAttribute("data-plot")]);
  }

  function showTip(event) {
    var dot = event.target.closest ? event.target.closest("[data-tip-name]") : null;
    var tip = state.nodes.tip;
    if (!dot) {
      return;
    }
    tip.textContent = "";
    tip.appendChild(el("strong", { text: dot.dataset.tipName }));
    [dot.dataset.tipProvider, dot.dataset.tipX, dot.dataset.tipY].forEach(function (line) {
      tip.appendChild(el("span", { text: line }));
    });
    var box = dot.getBoundingClientRect();
    var host = state.nodes.chart.getBoundingClientRect();
    tip.style.left = Math.round(box.left - host.left + box.width / 2) + "px";
    tip.style.top = Math.round(box.top - host.top) + "px";
    tip.hidden = false;
  }

  function hideTip() {
    state.nodes.tip.hidden = true;
  }

  /*
   * Clicking a provider filters the plot. Colour is a separate matter: only the
   * first few shown providers can take a hue, because past three no set of
   * categorical colours stays distinguishable on a scatter where every point
   * can sit beside every other. The rest are drawn neutral and named on hover,
   * and each of the four groups gets its own shape so colour is never the only
   * thing telling them apart.
   */
  function toggleHighlight(provider) {
    if (state.chart.hidden.has(provider)) {
      state.chart.hidden.delete(provider);
    } else {
      state.chart.hidden.add(provider);
    }
    render();
  }

  /* Hidden providers are the ones tracked, so a provider that appears when a
     filter is cleared plots by default rather than staying invisibly off. */
  function shownProvider(provider) {
    return !state.chart.hidden.has(provider);
  }

  function setProviders(all) {
    state.chart.hidden = all ? new Set() : new Set(state.chart.known);
    render();
  }

  function onKey(event) {
    if (!state || !document.body.contains(app)) {
      return;
    }
    if (event.key !== "Escape") {
      return;
    }
    if (!state.nodes.tip.hidden) {
      hideTip();
      return;
    }
    if (!state.nodes.detail.hidden) {
      closeDetail();
    }
  }

  /*
   * Instant navigation runs init on every page change, so anything attached to
   * the document or the window is attached once here instead — otherwise the
   * handlers pile up one copy per visit for the life of the tab.
   */
  document.addEventListener("keydown", onKey);
  window.addEventListener("resize", function () {
    if (state && state.nodes && state.nodes.scroll) {
      updateScrollHints();
    }
    if (state && state.nodes && state.nodes.tray && !state.nodes.tray.hidden) {
      updateTrayReserve();
    }
  });

  function init() {
    app = document.querySelector("[data-models-app]");
    if (!app) {
      return;
    }
    if (state) {
      if (state.trayWatcher) {
        state.trayWatcher.disconnect();
      }
      /* A debounced render still pending from the page just left would run
         render() against the state object this line is about to replace. */
      window.clearTimeout(state.searchTimer);
      window.clearTimeout(state.filterTimer);
    }
    /* The build ships this file minified under another name, so the tag is
       found by stem rather than by the name of the source. */
    var script = document.querySelector('script[src*="models-table"]');
    var source = script ? script.src : window.location.href;
    state = {
      dataBase: new URL("../models/", source).href,
      assetBase: new URL("../styles/", source).href,
    };
    fetch(new URL("catalog.json", state.dataBase).href)
      .then(function (response) {
        return response.json();
      })
      .then(mount)
      .catch(function () {
        app.textContent = "";
        app.appendChild(
          el("p", { class: "models-empty", text: "The model catalogue could not be loaded." })
        );
        app.hidden = false;
      });
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
