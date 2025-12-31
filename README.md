# 🤖 Agentic RAG System

An advanced Retrieval-Augmented Generation (RAG) system built with LangGraph that combines document retrieval with a ReAct agent for intelligent question-answering. The system can process documents from URLs, PDFs, and text files, and uses a hybrid approach combining vector search with Wikipedia knowledge for comprehensive answers.

## ✨ Features

- **Multi-Source Document Processing**: Load documents from URLs, PDF files, and text files
- **Vector-Based Retrieval**: Uses FAISS vector store with OpenAI embeddings for semantic search
- **ReAct Agent Architecture**: Implements a reasoning and acting agent that can use multiple tools
- **Hybrid Knowledge**: Combines retrieved documents with Wikipedia for comprehensive answers
- **LangGraph Workflow**: Built on LangGraph for robust, stateful agent workflows
- **Streamlit UI**: User-friendly web interface for interactive Q&A
- **CLI Support**: Command-line interface for programmatic usage
- **Flexible Configuration**: Easy-to-configure settings via environment variables

## 🏗️ Architecture

The system follows a modular architecture with the following components:

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Document       │  Loads from URLs/PDFs/TXT
│  Processor      │  Splits into chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  FAISS + OpenAI Embeddings
│  (FAISS)        │  Semantic search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangGraph      │  Stateful workflow
│  Workflow       │  ┌─────────────┐
│                 │  │  Retriever  │
│                 │  └──────┬──────┘
│                 │         │
│                 │  ┌──────▼──────┐
│                 │  │  ReAct      │
│                 │  │  Agent      │
│                 │  │  (Tools)    │
│                 │  └──────┬──────┘
└─────────────────┘         │
         │                  │
         └──────────┬───────┘
                    ▼
            ┌───────────────┐
            │   Answer     │
            └───────────────┘
```

### Components

1. **Document Processor** (`src/document_ingestion/document_processor.py`)
   - Loads documents from various sources (URLs, PDFs, TXT files)
   - Splits documents into chunks using RecursiveCharacterTextSplitter
   - Handles different document formats

2. **Vector Store** (`src/vectorstore/vectorstore.py`)
   - Creates FAISS vector store with OpenAI embeddings
   - Provides semantic search capabilities
   - Manages document retrieval

3. **Graph Builder** (`src/graph_builder/graph_builder.py`)
   - Constructs LangGraph workflow
   - Manages state transitions between nodes
   - Coordinates retrieval and generation

4. **RAG Nodes** (`src/node/reactnode.py`)
   - **Retriever Node**: Retrieves relevant documents
   - **Responder Node**: Uses ReAct agent with tools to generate answers
   - **Tools**: Custom retriever tool + Wikipedia tool

5. **State Management** (`src/state/rag_state.py`)
   - Defines RAGState schema using Pydantic
   - Tracks question, retrieved documents, and answer

## 📦 Installation

### Prerequisites

- Python 3.13+
- OpenAI API key

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## ⚙️ Configuration

Configuration is managed in `src/config/config.py`. Key settings:

- **LLM Model**: `openai:gpt-4o` (configurable)
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Default URLs**: Pre-configured URLs for document loading

To modify settings, edit `src/config/config.py`:

```python
class Config:
    LLM_MODEL = "openai:gpt-4o"  # Change model here
    CHUNK_SIZE = 500              # Adjust chunk size
    CHUNK_OVERLAP = 50            # Adjust overlap
    DEFAULT_URLS = [...]          # Add your URLs
```

## 🚀 Usage

### Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`. Features:
- Interactive Q&A interface
- Document source viewing
- Search history
- Response time tracking

### Command-Line Interface

Run the main script:

```bash
python main.py
```

The script will:
1. Initialize the RAG system
2. Process default URLs (or URLs from `data/urls.txt` if present)
3. Run example questions
4. Optionally enter interactive mode

#### Interactive Mode

After running example questions, you can enter interactive mode to ask custom questions:

```bash
python main.py
# ... after examples ...
Would you like to enter interactive mode? (y/n): y
Enter your question: What is the agent loop?
```

### Programmatic Usage

```python
from main import AgenticRAG

# Initialize with custom URLs
rag = AgenticRAG(urls=[
    "https://example.com/article1",
    "https://example.com/article2"
])

# Ask questions
answer = rag.ask("What is the main topic?")
print(answer)
```

## 📁 Project Structure

```
RAG_Proj/
├── data/                          # Document storage
│   ├── attention.pdf
│   ├── Nischitha.D.pdf
│   └── url.txt                    # Optional: URLs file
├── src/
│   ├── config/
│   │   └── config.py              # Configuration settings
│   ├── document_ingestion/
│   │   └── document_processor.py  # Document loading & splitting
│   ├── vectorstore/
│   │   └── vectorstore.py         # FAISS vector store
│   ├── graph_builder/
│   │   └── graph_builder.py       # LangGraph workflow builder
│   ├── node/
│   │   ├── nodes.py               # Basic RAG nodes (alternative)
│   │   └── reactnode.py           # ReAct agent nodes
│   └── state/
│       └── rag_state.py           # State schema
├── main.py                        # CLI entry point
├── streamlit_app.py               # Streamlit UI
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

## 🔧 How It Works

### Workflow Steps

1. **Document Ingestion**:
   - Documents are loaded from URLs, PDFs, or text files
   - Text is split into chunks with overlap for context preservation

2. **Vectorization**:
   - Document chunks are embedded using OpenAI embeddings
   - FAISS vector store is created for efficient similarity search

3. **Query Processing**:
   - User question is processed through the LangGraph workflow
   - **Retriever Node**: Finds relevant document chunks using semantic search

4. **Answer Generation**:
   - **Responder Node**: ReAct agent receives the question and retrieved documents
   - Agent has access to two tools:
     - **Retriever Tool**: Searches the indexed document corpus
     - **Wikipedia Tool**: Searches Wikipedia for general knowledge
   - Agent reasons about which tool to use and generates a comprehensive answer

5. **Response**:
   - Final answer is returned along with source documents

### ReAct Agent

The ReAct (Reasoning + Acting) agent:
- **Reasons** about the question and available information
- **Acts** by selecting appropriate tools (retriever or Wikipedia)
- **Observes** the results and iterates if needed
- Combines information from multiple sources for comprehensive answers

## 📋 Dependencies

Key dependencies (see `requirements.txt` for full list):

- **langchain**: Core LangChain framework
- **langchain-community**: Community integrations (Wikipedia, document loaders)
- **langchain-openai**: OpenAI integrations
- **langgraph**: LangGraph for agent workflows
- **faiss-cpu**: Vector similarity search
- **streamlit**: Web UI framework
- **pydantic**: Data validation
- **python-dotenv**: Environment variable management

## 🔍 Example Questions

The system is pre-configured with example questions:

- "What is the concept of agent loop in autonomous agents?"
- "What are the key components of LLM-powered agents?"
- "Explain the concept of diffusion models for video generation."

## 🛠️ Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure `.env` file exists with `OPENAI_API_KEY` set
   - Verify the API key is valid and has credits

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.13+ required)

3. **Document Loading Issues**:
   - Verify URLs are accessible
   - Check PDF files are not corrupted
   - Ensure text files use UTF-8 encoding

4. **Type Hint Errors**:
   - The Wikipedia tool wrapper fixes uuid type hint issues
   - If issues persist, check Python version compatibility

## 🚧 Future Enhancements

Potential improvements:

- [ ] Support for more document formats (DOCX, Markdown, etc.)
- [ ] Multi-modal document support (images, tables)
- [ ] Advanced retrieval strategies (hybrid search, reranking)
- [ ] Conversation memory/history
- [ ] Custom tool development framework
- [ ] Performance optimization and caching
- [ ] Docker containerization
- [ ] API endpoint for external integrations

## 📝 License

This project is open source. Please check the license file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ using LangChain, LangGraph, and OpenAI**

