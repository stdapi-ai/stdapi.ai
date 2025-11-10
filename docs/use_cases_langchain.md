# LangChain / LlamaIndex Integration

Build AI applications with LangChain and LlamaIndex using Amazon Bedrock models through stdapi.ai. Keep your existing code—just point it to stdapi.ai and access Claude, Amazon Nova, and all Bedrock models with zero code changes.

## About LangChain & LlamaIndex

**🔗 LangChain:** [Website](https://www.langchain.com/) | [Python Docs](https://python.langchain.com/) | [JS/TS Docs](https://js.langchain.com/) | [GitHub](https://github.com/langchain-ai/langchain) | [Discord](https://discord.gg/langchain)

**🔗 LlamaIndex:** [Website](https://www.llamaindex.ai/) | [Documentation](https://docs.llamaindex.ai/) | [GitHub](https://github.com/run-llama/llama_index) | [Discord](https://discord.gg/llamaindex)

Popular frameworks for building AI applications:

- LangChain - 90,000+ GitHub stars, comprehensive framework for chains, agents, and RAG
- LlamaIndex - 35,000+ GitHub stars, specialized for data indexing and retrieval
- Both support Python, JavaScript/TypeScript, and multiple LLM providers
- Production-ready - Used by companies building AI products

## Why LangChain/LlamaIndex + stdapi.ai?

!!! success "Drop-In Compatibility"
    Both frameworks are designed to work with OpenAI's API, making stdapi.ai a drop-in replacement for Amazon Bedrock models without changing your application code.

**Key Benefits:**

- Zero code changes - Change one parameter (base_url) to switch providers
- Keep your codebase - All existing LangChain/LlamaIndex code works as-is
- Access Claude, Amazon Nova, and all Bedrock models
- Use chains, agents, retrievers, and all framework capabilities
- AWS pricing instead of OpenAI rates
- Your data stays in your AWS environment

---

!!! warning "Work in Progress"
    This integration guide is actively being developed and refined. While the configuration examples are based on documented APIs and best practices, they are pending practical validation. Complete end-to-end deployment examples will be added once testing is finalized.

## Prerequisites

!!! info "What You'll Need"
    Before you begin, make sure you have:

    - ✓ Python 3.8+ or Node.js 18+ (depending on your language)
    - ✓ LangChain or LlamaIndex installed
    - ✓ Your stdapi.ai server URL (e.g., `https://api.example.com`)
    - ✓ An API key (if authentication is enabled)
    - ✓ AWS Bedrock access configured with desired models

---

## 🐍 LangChain Python Integration

### Installation

!!! example "Install LangChain"
    ```bash
    # Install LangChain with OpenAI support
    pip install langchain langchain-openai

    # For RAG applications, also install:
    pip install langchain-community chromadb
    ```

---

### Basic Chat Model Setup

Replace OpenAI with stdapi.ai in your LangChain applications.

!!! example "Python - Basic Chat"
    ```python
    from langchain_openai import ChatOpenAI

    # Configure chat model to use stdapi.ai
    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1",
        temperature=0.7
    )

    # Use it like any LangChain model
    response = llm.invoke("Explain quantum computing in simple terms")
    print(response.content)
    ```

**Available Models:**

Use any Amazon Bedrock chat model by specifying its model ID. Popular choices include Claude Sonnet, Claude Haiku, Amazon Nova Pro, Amazon Nova Lite, Amazon Nova Micro, and any other Bedrock models available in your region.

---

### Embeddings for RAG

Use Amazon Bedrock embeddings for semantic search and RAG applications.

!!! example "Python - Embeddings"
    ```python
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Configure embeddings to use stdapi.ai
    embeddings = OpenAIEmbeddings(
        model="amazon.titan-embed-text-v2:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1"
    )

    # Example: Create vector store from documents
    documents = ["Your document text here...", "Another document..."]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Query the vector store
    results = vectorstore.similarity_search("your query here", k=3)
    ```

**Available Embedding Models:**

- **Amazon Titan Embed Text v2** — `amazon.titan-embed-text-v2:0` (recommended, 8192 dimensions)
- **Amazon Titan Embed Text v1** — `amazon.titan-embed-text-v1` (legacy, 1536 dimensions)
- **Cohere Embed** — If enabled in your AWS region

---

### Complete RAG Pipeline

Build a full RAG application with stdapi.ai.

!!! example "Python - RAG Application"
    ```python
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    # 1. Configure LLM
    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1",
        temperature=0
    )

    # 2. Configure embeddings
    embeddings = OpenAIEmbeddings(
        model="amazon.titan-embed-text-v2:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1"
    )

    # 3. Load and process documents
    loader = TextLoader("path/to/your/documents.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 5. Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # 6. Query your documents
    query = "What are the key points in the document?"
    result = qa_chain.invoke({"query": query})

    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- {doc.page_content[:100]}...")
    ```

---

### Agents and Tools

Build LangChain agents powered by Bedrock models.

!!! example "Python - Agent with Tools"
    ```python
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.tools import Tool
    from langchain import hub

    # Configure LLM
    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1",
        temperature=0
    )

    # Define custom tools
    def search_documentation(query: str) -> str:
        """Search internal documentation."""
        # Your search logic here
        return f"Search results for: {query}"

    def calculate(expression: str) -> str:
        """Perform calculations."""
        try:
            return str(eval(expression))
        except:
            return "Invalid expression"

    tools = [
        Tool(
            name="SearchDocs",
            func=search_documentation,
            description="Search internal documentation for information"
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="Perform mathematical calculations"
        )
    ]

    # Get the prompt template
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the agent
    result = agent_executor.invoke({
        "input": "Search our docs for API authentication and calculate 15 * 23"
    })
    print(result["output"])
    ```

---

## 🦙 LlamaIndex Python Integration

### Installation

!!! example "Install LlamaIndex"
    ```bash
    # Install LlamaIndex with OpenAI support
    pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
    ```

---

### Basic Setup

Configure LlamaIndex to use stdapi.ai.

!!! example "Python - LlamaIndex Setup"
    ```python
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import Settings

    # Configure LLM
    llm = OpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        api_key="your_stdapi_key_here",
        api_base="https://YOUR_SERVER_URL/v1",
        temperature=0.7
    )

    # Configure embeddings
    embed_model = OpenAIEmbedding(
        model="amazon.titan-embed-text-v2:0",
        api_key="your_stdapi_key_here",
        api_base="https://YOUR_SERVER_URL/v1"
    )

    # Set global defaults
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Use the LLM
    response = llm.complete("Explain the benefits of vector databases")
    print(response)
    ```

---

### Document Indexing and Query

Build a searchable knowledge base with LlamaIndex.

!!! example "Python - LlamaIndex RAG"
    ```python
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        Settings
    )

    # Configure models
    Settings.llm = OpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        api_key="your_stdapi_key_here",
        api_base="https://YOUR_SERVER_URL/v1"
    )

    Settings.embed_model = OpenAIEmbedding(
        model="amazon.titan-embed-text-v2:0",
        api_key="your_stdapi_key_here",
        api_base="https://YOUR_SERVER_URL/v1"
    )

    # Load documents
    documents = SimpleDirectoryReader("./data").load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Query the index
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("What are the main topics in these documents?")

    print(response)
    ```

---

## 🟨 LangChain JavaScript/TypeScript Integration

### Installation

!!! example "Install LangChain.js"
    ```bash
    # Using npm
    npm install langchain @langchain/openai

    # Using yarn
    yarn add langchain @langchain/openai
    ```

---

### Basic Chat Model Setup

!!! example "TypeScript - Basic Chat"
    ```typescript
    import { ChatOpenAI } from "@langchain/openai";

    // Configure chat model to use stdapi.ai
    const llm = new ChatOpenAI({
      modelName: "anthropic.claude-sonnet-4-5-20250929-v1:0",
      openAIApiKey: "your_stdapi_key_here",
      configuration: {
        baseURL: "https://YOUR_SERVER_URL/v1",
      },
      temperature: 0.7,
    });

    // Use it
    const response = await llm.invoke("Explain machine learning in simple terms");
    console.log(response.content);
    ```

---

### RAG with LangChain.js

!!! example "TypeScript - RAG Application"
    ```typescript
    import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
    import { MemoryVectorStore } from "langchain/vectorstores/memory";
    import { RetrievalQAChain } from "langchain/chains";
    import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
    import { Document } from "langchain/document";

    // Configure models
    const llm = new ChatOpenAI({
      modelName: "anthropic.claude-sonnet-4-5-20250929-v1:0",
      openAIApiKey: "your_stdapi_key_here",
      configuration: {
        baseURL: "https://YOUR_SERVER_URL/v1",
      },
      temperature: 0,
    });

    const embeddings = new OpenAIEmbeddings({
      modelName: "amazon.titan-embed-text-v2:0",
      openAIApiKey: "your_stdapi_key_here",
      configuration: {
        baseURL: "https://YOUR_SERVER_URL/v1",
      },
    });

    // Load and split documents
    const documents = [
      new Document({ pageContent: "Your document text here..." }),
      new Document({ pageContent: "Another document..." }),
    ];

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const chunks = await textSplitter.splitDocuments(documents);

    // Create vector store
    const vectorStore = await MemoryVectorStore.fromDocuments(
      chunks,
      embeddings
    );

    // Create retrieval chain
    const chain = RetrievalQAChain.fromLLM(
      llm,
      vectorStore.asRetriever(3)
    );

    // Query
    const result = await chain.call({
      query: "What are the key points in the document?",
    });

    console.log(result.text);
    ```

---

## 💡 Common Patterns and Best Practices

### Environment Variables

Store credentials securely using environment variables.

!!! example "Python - Environment Variables"
    ```python
    import os
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        openai_api_key=os.getenv("STDAPI_KEY"),
        openai_api_base=os.getenv("STDAPI_BASE_URL"),
        temperature=0.7
    )
    ```

!!! example "TypeScript - Environment Variables"
    ```typescript
    import { ChatOpenAI } from "@langchain/openai";

    const llm = new ChatOpenAI({
      modelName: "anthropic.claude-sonnet-4-5-20250929-v1:0",
      openAIApiKey: process.env.STDAPI_KEY,
      configuration: {
        baseURL: process.env.STDAPI_BASE_URL,
      },
      temperature: 0.7,
    });
    ```

---

### Streaming Responses

Enable streaming for better user experience in interactive applications.

!!! example "Python - Streaming"
    ```python
    from langchain_openai import ChatOpenAI
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    # Response will stream to stdout as it's generated
    llm.invoke("Write a long story about artificial intelligence")
    ```

---

### Retry Logic and Error Handling

Implement robust error handling for production applications.

!!! example "Python - Error Handling"
    ```python
    from langchain_openai import ChatOpenAI
    from langchain.globals import set_llm_cache
    from langchain.cache import InMemoryCache
    import time

    # Enable caching to reduce redundant calls
    set_llm_cache(InMemoryCache())

    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        openai_api_key="your_stdapi_key_here",
        openai_api_base="https://YOUR_SERVER_URL/v1",
        max_retries=3,
        request_timeout=60
    )

    def query_with_retry(prompt: str, max_attempts: int = 3):
        for attempt in range(max_attempts):
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    result = query_with_retry("Your prompt here")
    ```

---

## 📊 Model Recommendations

Choose the right model for your application needs. **These are examples**—all Bedrock models are available.

| Use Case | Example Model | Why |
|----------|--------------|-----|
| **Complex RAG** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Superior reasoning and context understanding |
| **Fast Queries** | `amazon.nova-micro-v1:0` | Quick responses, cost-effective |
| **Long Documents** | `amazon.nova-pro-v1:0` | Large context window (300K tokens) |
| **Balanced** | `amazon.nova-lite-v1:0` | Good performance at reasonable cost |
| **Code Generation** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Excellent coding capabilities |
| **Embeddings** | `amazon.titan-embed-text-v2:0` | High-dimensional (8192), optimized for RAG |

---

## 🔧 Migration from OpenAI

Migrating existing LangChain/LlamaIndex applications is straightforward.

!!! tip "Migration Steps"
    **1. Update Configuration:**
    - Change `openai_api_base` to your stdapi.ai URL
    - Change `openai_api_key` to your stdapi.ai key
    - Change model names to Bedrock model IDs

    **2. Test Thoroughly:**
    - Start with non-critical workflows
    - Compare outputs between OpenAI and Bedrock models
    - Monitor performance and costs

    **3. Gradual Rollout:**
    - Deploy to staging/dev environments first
    - Run A/B tests if possible
    - Gather team feedback

    **4. Monitor and Optimize:**
    - Track token usage and costs
    - Adjust model selection based on performance
    - Fine-tune prompts for Bedrock models

---

## 🚀 Next Steps & Resources

### Getting Started

1. **Try Examples:** Run the code examples above with your stdapi.ai credentials
2. **Build a Prototype:** Start with a simple RAG application
3. **Experiment with Models:** Test different Bedrock models for your use case
4. **Scale Up:** Move to production with proper error handling and monitoring
5. **Optimize:** Fine-tune model selection and parameters for cost/performance

### Learn More

!!! info "Additional Resources"
    - **[LangChain Documentation](https://python.langchain.com/)** — Official LangChain guides
    - **[LlamaIndex Documentation](https://docs.llamaindex.ai/)** — Official LlamaIndex guides
    - **[API Overview](api_overview.md)** — Complete list of available Bedrock models
    - **[Chat Completions API](api_openai_chat_completions.md)** — Detailed API documentation
    - **[Configuration Guide](operations_configuration.md)** — Advanced stdapi.ai configuration

### Community & Support

!!! question "Need Help?"
    - 💬 Join the [LangChain Discord](https://discord.gg/langchain) for framework questions
    - 💬 Join the [LlamaIndex Discord](https://discord.gg/llamaindex) for framework questions
    - 📖 Review Amazon Bedrock documentation for model-specific details
    - 🐛 Report issues on the [GitHub repository](https://github.com/stdapi-ai/stdapi.ai)

---

## ⚠️ Important Considerations

!!! warning "Model Availability"
    **Regional Differences:** Not all Amazon Bedrock models are available in every AWS region. Verify model availability before deploying production applications.

    **Check availability:** See the [API Overview](api_overview.md) for supported models by region.

!!! info "API Compatibility"
    **OpenAI Compatibility:** stdapi.ai implements the OpenAI API specification. Most features work seamlessly, but some advanced OpenAI-specific features may have differences.

    **Function Calling:** Function calling support varies by model—test thoroughly if your application relies on it.

!!! tip "Cost Optimization"
    - **Cache Frequently Used Results:** Use LangChain's caching to reduce API calls
    - **Right-Size Context:** Only include necessary context in prompts to reduce token usage
    - **Choose Efficient Models:** Use Nova Micro/Lite for simple tasks, premium models for complex reasoning
    - **Batch Processing:** Process multiple items in batches when possible
    - **Monitor Token Usage:** Track consumption and set up alerts for unexpected spikes
