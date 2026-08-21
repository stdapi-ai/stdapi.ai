---
title: Conversations API - OpenAI-Compatible Conversation State
description: Create conversations, add and list items, and attach them to Responses requests with the OpenAI-compatible Conversations API. Server-side multi-turn state with metadata, cursor pagination and per-item deletion.
keywords: Conversations API, OpenAI conversations, conversation state, multi-turn, conversation items, conversation metadata, responses conversation parameter, cursor pagination
---

# Conversations API

Keep multi-turn state on the server. A conversation holds the items of an
exchange — user messages, model output, reasoning and tool calls — so each new
turn only has to send the new message. Pass a conversation ID as `conversation`
on a [Responses](api_openai_responses.md) request and both the request input and
the response output are added to it automatically.

## Why Choose the Conversations API?

<div class="grid cards" markdown>

- :material-chat-processing: __Send Only the New Turn__
  <br>The conversation's items become the input prefix of the next request, so a long exchange stays a one-message request.

- :material-playlist-plus: __Explicit Item Control__
  <br>Add, list, retrieve and delete items yourself, independently of any model call.

- :material-tag-multiple: __Attached Metadata__
  <br>Up to 16 key-value pairs per conversation, merged on update and removable key by key.

- :material-format-list-bulleted: __Cursor Pagination__
  <br>List items newest- or oldest-first with `limit` and the `after` cursor.

- :material-server-network: __No Gateway State__
  <br>The thread lives in your AWS account, not in a gateway instance, so any instance behind a load balancer serves any conversation.

</div>

## Available Endpoints

| Endpoint                                          | Method   | What It Does                        | MCP Tool                            |
|---------------------------------------------------|----------|-------------------------------------|-------------------------------------|
| `/v1/conversations`                               | `POST`   | Create a conversation               | `openai_conversation`               |
| `/v1/conversations/{conversation_id}`             | `GET`    | Retrieve a conversation             | `openai_conversation_get`           |
| `/v1/conversations/{conversation_id}`             | `POST`   | Update the conversation's metadata  | `openai_conversation_update`        |
| `/v1/conversations/{conversation_id}`             | `DELETE` | Delete a conversation and its items | `openai_conversation_delete`        |
| `/v1/conversations/{conversation_id}/items`       | `POST`   | Add items to a conversation         | `openai_conversation_items`         |
| `/v1/conversations/{conversation_id}/items`       | `GET`    | List a conversation's items         | `openai_conversation_items_list`    |
| `/v1/conversations/{conversation_id}/items/{item_id}` | `GET`    | Retrieve one item               | `openai_conversation_item_get`      |
| `/v1/conversations/{conversation_id}/items/{item_id}` | `DELETE` | Delete one item                 | `openai_conversation_item_delete`   |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                                   |                  Status                  | Notes                                                                       |
|-------------------------------------------|:----------------------------------------:|-----------------------------------------------------------------------------|
| **Conversation**                          |                                          |                                                                             |
| `items`                                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Up to 20 initial items, prepared before the conversation is created, so a rejected item leaves no empty conversation behind |
| `metadata`                                |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | The limits in [Metadata](#metadata). A key given a `null` value is accepted and dropped rather than stored; on an update that same `null` removes the key |
| A request with no body                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Both fields are optional, and the body itself may be omitted                |
| Retrieve, update, delete                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `metadata` is the only field an update takes; a delete removes the conversation and every item it holds |
| **Items**                                 |                                          |                                                                             |
| `items` on an add                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | 1 to 20 per request; the response is a `list` envelope of the items added, not the whole conversation |
| Item shapes                               |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | The [Responses](api_openai_responses.md) `input` and `output` items: messages, reasoning items, tool calls and their outputs |
| A message `content` sent as a string      |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Expanded into an `input_text` or `output_text` part according to the message's `role` |
| An `id` sent on a new item                |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Accepted and ignored — the server mints the identifier, prefixed by the item's type |
| `item_reference` items                    |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Resolved against the conversation and then dropped rather than stored again, since the item it names is already there; one naming an item that is not answers `404` |
| Retrieve and delete one item              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | A delete returns the conversation, and the item leaves both the listing and the prefix of the next Responses turn |
| **Listing items**                         |                                          |                                                                             |
| `order`, `limit`, `after`                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Bounds and cursor semantics in [Listing](#listing)                          |
| `include`                                 |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Accepted on the listing, on a retrieval and on an add. Only `reasoning.encrypted_content` changes the response; every other value is accepted and ignored |
| `first_id` / `last_id` / `has_more`       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Populated on every page; the server never auto-paginates                    |
| **Lifecycle**                             |                                          |                                                                             |
| Conversation lifetime                     |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | 30 days from creation, after which every route on it answers `404`. Amazon Bedrock session storage sets the window and it cannot be extended |
| Adding and deleting items                 |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | 1,000 requests per conversation — see [Limits](#limits)                     |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations

</div>

!!! note "No model is involved"
    Every endpoint on this page is state management: items are stored and read
    back as they were sent. Nothing here embeds, summarises or re-runs an item,
    so no endpoint takes a model identifier and a conversation behaves the same
    whatever the deployment's catalog holds. Summarising a long exchange is
    [`POST /v1/responses/compact`](api_openai_responses.md#conversation-compaction),
    on the Responses API.

## Quick Start

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-gateway/v1", api_key="YOUR_API_KEY")

conversation = client.conversations.create(metadata={"topic": "travel"})

first = client.responses.create(
    model="amazon.nova-micro-v1:0",
    input="My favourite city is Lisbon.",
    conversation=conversation.id,
)

second = client.responses.create(
    model="amazon.nova-micro-v1:0",
    input="Which city did I name?",
    conversation=conversation.id,
)
print(second.output_text)
```

The second request carries no history: the conversation supplies it.

## Using a Conversation with the Responses API

| Request                                    | Effect                                                                 |
|--------------------------------------------|------------------------------------------------------------------------|
| `conversation="conv-..."`                  | The conversation's items are prepended to `input`; the request input and the response output are appended to the conversation. |
| `conversation={"id": "conv-..."}`          | Same; both forms are accepted.                                          |
| `conversation=...`, `store=false`          | The conversation is still used as the input prefix, but nothing is added to it. |
| `conversation=...`, `stream=true`          | Items are added once the stream has ended, and the terminal event carries the conversation. |
| `conversation=...` and `previous_response_id=...` | Rejected with `400` (`mutually_exclusive_parameters`) — pick one way of continuing the exchange. |

The response echoes the conversation it belongs to as `"conversation": {"id": "conv-..."}`.
A response that fails before generating anything adds nothing to the conversation.

`conversation` is also accepted on `/v1/responses/input_tokens`, where the
conversation's items are counted ahead of `input`.

## Items

Items use the same shapes as the Responses API `input` and `output`: messages,
reasoning items, function calls and their outputs.

- **Item IDs are assigned by the server.** An `id` sent on a new item is ignored.
- **Adding items returns the items that were added**, as a `list` envelope — not
  the whole conversation.
- **`item_reference` items** point at an item already in the conversation; a
  reference to an item that is not there returns `404`.
- **Deleting an item returns the conversation**, and the item disappears from
  the listing.
- `include=reasoning.encrypted_content` returns the encrypted content of
  reasoning items; other `include` values are accepted and ignored.

### Listing

| Parameter | Default  | Notes                                                        |
|-----------|----------|--------------------------------------------------------------|
| `order`   | `desc`   | `asc` is conversation order.                                 |
| `limit`   | `20`     | Up to 100 items per page; `0` returns an empty page.         |
| `after`   | —        | An item ID; only items strictly after it are returned. An ID that is not in the conversation returns `404`. |
| `include` | —        | Extra item fields to return.                                 |

Each page carries `first_id`, `last_id` and `has_more`. Pass the page's
`last_id` as `after` to read the next page; the server never auto-paginates.

## Metadata

| Limit                        | Value |
|------------------------------|-------|
| Key-value pairs              | 16    |
| Key length                   | 64    |
| Value length                 | 512   |

Updating **merges**: keys that are not sent keep their value, and a key sent as
`null` is removed. `metadata` is required on update — omitting it returns `400`
(`missing_required_parameter`), and sending `null` returns `400`
(`invalid_type`).

## Limits

| Limit                                             | Value                  |
|---------------------------------------------------|------------------------|
| Items per add request                             | 20                     |
| Invocation steps read when listing a conversation | 1,000 — a large item spans several, so a listing can stop early |
| Conversation lifetime                             | 30 days after creation |

A response bound to a conversation counts as one adding request, whatever its
number of output items. Start a new conversation to continue an exchange that
has reached the limit, and see
[Troubleshooting](operations_troubleshooting.md)
if a conversation stops accepting items earlier than expected.

## Errors

| Status | When                                                                 |
|--------|----------------------------------------------------------------------|
| `400`  | A malformed `conversation_id` or `item_id`, a metadata limit, an empty or oversized `items` list, or `conversation` combined with `previous_response_id`. |
| `404`  | A well-formed identifier that names no conversation or item, including one created on another provider. |

## Prerequisites

Conversations are stored in Amazon Bedrock session storage in your own account.
The gateway's IAM role needs the
[Bedrock Session Storage permissions](operations_iam_permissions.md#bedrock-session-storage-optional);
without them, conversation requests fail with `503`. Set
[`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-bedrock-session-encryption-key-arn)
to encrypt conversation content with your own AWS KMS key.

## See Also

- [Responses API](api_openai_responses.md) — the `conversation` parameter and stored responses.
- [Configuration](operations_configuration.md#bedrock-session-storage-optional) — session storage settings.
- [IAM Permissions](operations_iam_permissions.md#bedrock-session-storage-optional) — the policy statement.
