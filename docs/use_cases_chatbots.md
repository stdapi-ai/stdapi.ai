# Chat Platform Bots — Slack, Discord & Teams

Build intelligent chatbots for your team communication platforms powered by Amazon Bedrock models through stdapi.ai. Deploy AI assistants to Slack, Discord, Microsoft Teams, and other chat platforms—using OpenAI-compatible bot frameworks with enterprise-grade models.

## About Chat Platform Bots

Many open-source bot frameworks and templates are designed to work with OpenAI's API, making them perfect for stdapi.ai integration. This guide covers popular platforms and bot frameworks.

### Supported Platforms

**🔗 Slack:** [API Docs](https://api.slack.com/) | [Bolt Framework](https://slack.dev/bolt-python/concepts) | [Bot Examples](https://github.com/slackapi/bolt-python)

**🔗 Discord:** [Developer Portal](https://discord.com/developers/docs) | [Discord.py](https://discordpy.readthedocs.io/) | [Discord.js](https://discord.js.org/)

**🔗 Microsoft Teams:** [Developer Docs](https://learn.microsoft.com/en-us/microsoftteams/platform/) | [Bot Framework](https://dev.botframework.com/)

**🔗 Telegram:** [Bot API](https://core.telegram.org/bots/api) | [python-telegram-bot](https://python-telegram-bot.org/)

**🔗 Mattermost:** [Developer Docs](https://developers.mattermost.com/) | [Integration Guide](https://developers.mattermost.com/integrate/other-integrations/)

---

## Why Chat Bots + stdapi.ai?

!!! success "AI for Team Communication"
    Connect team communication platforms with AI assistants powered by Amazon Bedrock instead of OpenAI.

**Key Benefits:**

- Claude Sonnet and other Bedrock models for conversation and context understanding
- Your conversations stay in your AWS environment
- AWS pricing for high-volume bot interactions
- Choose different models for different bot purposes
- Avoid OpenAI's strict rate limiting
- Compliance, audit logs, and data residency control

---

!!! warning "Work in Progress"
    This integration guide is actively being developed and refined. While the configuration examples are based on documented APIs and best practices, they are pending practical validation. Complete end-to-end deployment examples will be added once testing is finalized.

## 🤖 Quick Start Examples

### Slack Bot with Python

Build a Slack bot using Bolt for Python and stdapi.ai.

**Installation:**

```bash
pip install slack-bolt openai
```

!!! example "Python - Slack Bot"
    ```python
    import os
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    from openai import OpenAI

    # Initialize Slack app
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    # Initialize OpenAI client with stdapi.ai
    client = OpenAI(
        api_key=os.environ["STDAPI_KEY"],
        base_url="https://YOUR_SERVER_URL/v1"
    )

    # Store conversation history per thread
    conversations = {}

    @app.event("app_mention")
    def handle_mention(event, say):
        """Respond when bot is mentioned"""
        thread_ts = event.get("thread_ts", event["ts"])
        user_message = event["text"]

        # Get or create conversation history
        if thread_ts not in conversations:
            conversations[thread_ts] = []

        # Add user message to history
        conversations[thread_ts].append({
            "role": "user",
            "content": user_message
        })

        # Get AI response
        response = client.chat.completions.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=conversations[thread_ts],
            max_tokens=1000
        )

        assistant_message = response.choices[0].message.content

        # Add assistant response to history
        conversations[thread_ts].append({
            "role": "assistant",
            "content": assistant_message
        })

        # Reply in thread
        say(text=assistant_message, thread_ts=thread_ts)

    @app.command("/ask")
    def handle_ask_command(ack, command, say):
        """Handle /ask slash command"""
        ack()

        response = client.chat.completions.create(
            model="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": command["text"]}]
        )

        say(response.choices[0].message.content)

    if __name__ == "__main__":
        # Start the app with Socket Mode
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start()
    ```

**Environment Variables:**

```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
STDAPI_KEY=your_stdapi_key
```

---

### Discord Bot with Python

Create a Discord bot using discord.py and stdapi.ai.

**Installation:**

```bash
pip install discord.py openai
```

!!! example "Python - Discord Bot"
    ```python
    import os
    import discord
    from discord.ext import commands
    from openai import OpenAI

    # Initialize bot
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    # Initialize OpenAI client with stdapi.ai
    client = OpenAI(
        api_key=os.environ["STDAPI_KEY"],
        base_url="https://YOUR_SERVER_URL/v1"
    )

    # Store conversations per channel
    conversations = {}

    @bot.event
    async def on_ready():
        print(f"Bot is ready as {bot.user}")

    @bot.command(name="ask")
    async def ask(ctx, *, question):
        """Ask the AI a question"""
        async with ctx.typing():
            response = client.chat.completions.create(
                model="anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[{"role": "user", "content": question}],
                max_tokens=1000
            )

            answer = response.choices[0].message.content

            # Split long messages
            if len(answer) > 2000:
                chunks = [answer[i:i+2000] for i in range(0, len(answer), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(answer)

    @bot.command(name="chat")
    async def chat(ctx, *, message):
        """Chat with context from conversation history"""
        channel_id = ctx.channel.id

        # Initialize conversation history for channel
        if channel_id not in conversations:
            conversations[channel_id] = []

        # Add user message
        conversations[channel_id].append({
            "role": "user",
            "content": message
        })

        # Keep only last 10 messages for context
        if len(conversations[channel_id]) > 20:
            conversations[channel_id] = conversations[channel_id][-20:]

        async with ctx.typing():
            response = client.chat.completions.create(
                model="anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=conversations[channel_id]
            )

            answer = response.choices[0].message.content

            # Add assistant response
            conversations[channel_id].append({
                "role": "assistant",
                "content": answer
            })

            await ctx.send(answer)

    @bot.command(name="reset")
    async def reset(ctx):
        """Clear conversation history"""
        channel_id = ctx.channel.id
        if channel_id in conversations:
            del conversations[channel_id]
        await ctx.send("Conversation history cleared!")

    # Run the bot
    bot.run(os.environ["DISCORD_TOKEN"])
    ```

**Environment Variables:**

```bash
DISCORD_TOKEN=your_discord_bot_token
STDAPI_KEY=your_stdapi_key
```

---

### Microsoft Teams Bot with Node.js

Build a Teams bot using Bot Framework and stdapi.ai.

**Installation:**

```bash
npm install botbuilder openai
```

!!! example "JavaScript - Teams Bot"
    ```javascript
    const { ActivityHandler } = require('botbuilder');
    const OpenAI = require('openai');

    class TeamsBot extends ActivityHandler {
        constructor() {
            super();

            // Initialize OpenAI client with stdapi.ai
            this.client = new OpenAI({
                apiKey: process.env.STDAPI_KEY,
                baseURL: 'https://YOUR_SERVER_URL/v1'
            });

            // Store conversations
            this.conversations = new Map();

            // Handle message activities
            this.onMessage(async (context, next) => {
                const userMessage = context.activity.text;
                const conversationId = context.activity.conversation.id;

                // Get or create conversation history
                if (!this.conversations.has(conversationId)) {
                    this.conversations.set(conversationId, []);
                }

                const history = this.conversations.get(conversationId);

                // Add user message
                history.push({
                    role: 'user',
                    content: userMessage
                });

                // Get AI response
                const response = await this.client.chat.completions.create({
                    model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
                    messages: history,
                    max_tokens: 1000
                });

                const botReply = response.choices[0].message.content;

                // Add bot response to history
                history.push({
                    role: 'assistant',
                    content: botReply
                });

                // Keep last 20 messages
                if (history.length > 40) {
                    this.conversations.set(
                        conversationId,
                        history.slice(-40)
                    );
                }

                // Send reply
                await context.sendActivity(botReply);
                await next();
            });

            // Handle members added
            this.onMembersAdded(async (context, next) => {
                const welcomeText = 'Hello! I\'m your AI assistant. Ask me anything!';
                for (const member of context.activity.membersAdded) {
                    if (member.id !== context.activity.recipient.id) {
                        await context.sendActivity(welcomeText);
                    }
                }
                await next();
            });
        }
    }

    module.exports.TeamsBot = TeamsBot;
    ```

---

## 🎯 Common Bot Features

### Conversation Management

**Thread/Channel Context:**

- Store conversation history per thread or channel
- Limit context to recent messages (last 10-20 messages)
- Clear history on command or timeout
- Support multiple simultaneous conversations

!!! example "Context Management Pattern"
    ```python
    MAX_HISTORY = 20  # Keep last 20 messages

    def manage_context(conversation_id, new_message):
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append(new_message)

        # Trim old messages
        if len(conversations[conversation_id]) > MAX_HISTORY:
            conversations[conversation_id] = conversations[conversation_id][-MAX_HISTORY:]

        return conversations[conversation_id]
    ```

---

### Command Systems

**Slash Commands / Bot Commands:**

- `/ask <question>` — Quick one-off questions
- `/chat <message>` — Contextual conversation
- `/reset` — Clear conversation history
- `/help` — Show available commands
- `/model <model_name>` — Switch AI model

---

### Smart Features

**Enhanced Capabilities:**

- **@mentions Detection** — Respond only when mentioned
- **Direct Messages** — Private 1-on-1 conversations
- **Reaction Triggers** — React to specific emoji reactions
- **Scheduled Messages** — Automated daily summaries or reminders
- **File Processing** — Analyze uploaded documents or images

---

### Access Control

**Security & Permissions:**

- **Channel Restrictions** — Limit bot to specific channels
- **Role-Based Access** — Different features for different user roles
- **Rate Limiting** — Prevent abuse with per-user rate limits
- **Audit Logging** — Track all bot interactions for compliance

---

## 📊 Model Selection by Use Case

Choose models based on bot purpose. **These are examples**—all Bedrock models are available.

| Bot Purpose | Example Model | Why |
|-------------|--------------|-----|
| **General Assistant** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Best conversation quality and context understanding |
| **Quick Q&A** | `amazon.nova-lite-v1:0` | Fast responses, cost-effective for high volume |
| **Technical Support** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Superior at technical explanations and debugging |
| **Simple Commands** | `amazon.nova-micro-v1:0` | Very fast, efficient for basic queries |
| **Long Discussions** | `amazon.nova-pro-v1:0` | Large context window for complex conversations |

---

## 🚀 Deployment Options

### Option 1: Cloud Hosting

Deploy bots to cloud platforms for reliability and scalability.

!!! example "Heroku Deployment"
    ```bash
    # Install Heroku CLI
    # Create app
    heroku create my-slack-bot

    # Set environment variables
    heroku config:set SLACK_BOT_TOKEN=xoxb-...
    heroku config:set STDAPI_KEY=your_key

    # Deploy
    git push heroku main
    ```

**Popular Platforms:**

- **Heroku** — Easy deployment, free tier available
- **AWS Lambda** — Serverless, pay per use
- **Google Cloud Run** — Container-based, autoscaling
- **Railway** — Simple, modern deployment
- **Fly.io** — Global edge deployment

---

### Option 2: Self-Hosted

Run bots on your own infrastructure for full control.

!!! example "Docker Deployment"
    ```dockerfile
    FROM python:3.11-slim

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install -r requirements.txt

    COPY bot.py .

    CMD ["python", "bot.py"]
    ```

    ```bash
    docker build -t my-bot .
    docker run -e SLACK_BOT_TOKEN=$TOKEN -e STDAPI_KEY=$KEY my-bot
    ```

---

### Option 3: Serverless

Use serverless functions for cost-effective, scalable bots.

!!! example "AWS Lambda + API Gateway"
    - Deploy bot as Lambda function
    - Use API Gateway for webhook endpoint
    - Store conversation state in DynamoDB
    - Scale automatically with demand

---

## 💡 Best Practices

!!! tip "Performance Optimization"
    **Async Operations:** Use async/await for non-blocking bot responses

    **Typing Indicators:** Show "typing..." while generating responses

    **Response Streaming:** Stream long responses in chunks

    **Caching:** Cache frequent queries to reduce API calls

!!! tip "User Experience"
    **Clear Commands:** Document all bot commands with `/help`

    **Error Handling:** Graceful error messages, not technical stack traces

    **Feedback:** Allow users to rate responses or report issues

    **Context Limits:** Inform users when context is cleared

!!! tip "Security & Privacy"
    **Environment Variables:** Never hardcode API keys in source code

    **Message Filtering:** Sanitize user input before sending to AI

    **Data Retention:** Clear conversation history after timeout

    **Access Logs:** Monitor bot usage for suspicious activity

    **Channel Privacy:** Respect private channel settings

!!! tip "Cost Management"
    **Model Selection:** Use efficient models for simple queries

    **Context Management:** Limit conversation history length

    **Rate Limiting:** Prevent abuse with per-user limits

    **Monitoring:** Track token usage and set budget alerts

---

## 🎨 Advanced Features

### RAG Integration

**Document Search & Retrieval:**

Integrate vector search to answer questions from your team's knowledge base.

!!! example "RAG-Enhanced Bot"
    ```python
    from openai import OpenAI
    import chromadb

    # Initialize clients
    ai_client = OpenAI(
        api_key=os.environ["STDAPI_KEY"],
        base_url="https://YOUR_SERVER_URL/v1"
    )

    # Vector database for documents
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("team_docs")

    def answer_with_context(question):
        # Search relevant documents
        results = collection.query(
            query_texts=[question],
            n_results=3
        )

        context = "\n\n".join(results['documents'][0])

        # Generate answer with context
        response = ai_client.chat.completions.create(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }]
        )

        return response.choices[0].message.content
    ```

---

### Multi-Modal Support

**Handle Images & Files:**

Process images, documents, and other files shared in chat.

!!! example "Image Analysis Bot"
    ```python
    @bot.command(name="analyze")
    async def analyze_image(ctx):
        """Analyze attached images"""
        if ctx.message.attachments:
            for attachment in ctx.message.attachments:
                if attachment.content_type.startswith('image/'):
                    # Download image
                    image_data = await attachment.read()

                    # Send to vision-capable model
                    # (Note: depends on model support)
                    response = client.chat.completions.create(
                        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
                        messages=[{
                            "role": "user",
                            "content": "Describe this image"
                        }]
                    )

                    await ctx.send(response.choices[0].message.content)
    ```

---

## 🚀 Next Steps & Resources

### Getting Started

1. **Choose Your Platform:** Start with Slack or Discord for easiest setup
2. **Create Bot Account:** Register bot in platform's developer portal
3. **Deploy Example Code:** Use one of the examples above as a starting point
4. **Configure stdapi.ai:** Add your credentials and select models
5. **Test & Iterate:** Start in test channel, gather feedback, improve

### Learn More

!!! info "Additional Resources"
    - **[Slack Bolt Framework](https://slack.dev/bolt-python/tutorial/getting-started)** — Official Slack bot framework
    - **[Discord.py Documentation](https://discordpy.readthedocs.io/)** — Comprehensive Discord bot guide
    - **[Bot Framework Docs](https://learn.microsoft.com/en-us/azure/bot-service/)** — Microsoft Teams bot development
    - **[API Overview](api_overview.md)** — Complete list of available Bedrock models
    - **[Chat Completions API](api_openai_chat_completions.md)** — Detailed API documentation

### Community & Support

!!! question "Need Help?"
    - 💬 Platform-specific developer communities (Slack, Discord, Teams)
    - 📖 Review Amazon Bedrock documentation for model-specific details
    - 🐛 Report issues on the [GitHub repository](https://github.com/stdapi-ai/stdapi.ai)

---

## ⚠️ Important Considerations

!!! warning "Model Availability"
    **Regional Differences:** Not all Amazon Bedrock models are available in every AWS region. Verify model availability before deployment.

    **Check availability:** See the [API Overview](api_overview.md) for supported models by region.

!!! info "Rate Limits & Scaling"
    **Bot Rate Limits:** Chat platforms have rate limits—implement proper throttling

    **Concurrent Users:** Plan for multiple simultaneous conversations

    **State Management:** Use databases for persistent conversation storage in production

!!! tip "Compliance & Data Privacy"
    **Message Logging:** Be transparent about what data is stored

    **Retention Policies:** Implement automatic data deletion policies

    **GDPR/Privacy Laws:** Ensure compliance with applicable regulations

    **User Consent:** Inform users about AI interactions and data usage
