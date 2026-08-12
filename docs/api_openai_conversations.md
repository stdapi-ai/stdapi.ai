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

- :material-server-network: __No Server State__
  <br>Conversations are stored in your AWS account, so any gateway instance serves any conversation.

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
| `limit`   | `20`     | 1 to 100 items per page.                                     |
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
| Requests that add or delete items, per conversation | 1,000                |
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
without them, conversation requests fail with `403`. Set
[`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-bedrock-session-encryption-key-arn)
to encrypt conversation content with your own AWS KMS key.

## See Also

- [Responses API](api_openai_responses.md) — the `conversation` parameter and stored responses.
- [Configuration](operations_configuration.md#bedrock-session-storage-optional) — session storage settings.
- [IAM Permissions](operations_iam_permissions.md#bedrock-session-storage-optional) — the policy statement.
