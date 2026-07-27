# Flair 7 Tech Support Bot

A local retrieval-augmented generation (RAG) chatbot for answering technical-support questions about Flair 7 robotic camera arms. It searches documents stored locally, then uses a local Hugging Face language model to draft an answer from the relevant material.

## Current stack

- **Application framework:** LlamaIndex
- **Vector database:** ChromaDB, persisted locally in `chroma_db/`
- **Embedding model:** `BAAI/bge-small-en-v1.5`
- **Language model currently configured:** `Qwen/Qwen2.5-3B-Instruct`
- **Model runtime:** Hugging Face Transformers + PyTorch

> **Note:** The source file’s comments mention Qwen3:4B, but the model actually loaded by the current code is `Qwen/Qwen2.5-3B-Instruct`.

## How it works

1. Put manuals, release notes, troubleshooting guides, and other support documents in `flair7_docs/`.
2. LlamaIndex reads those files and breaks their content into searchable chunks.
3. The embedding model converts chunks into vectors and ChromaDB saves them locally.
4. When a question is asked, the bot retrieves the three most relevant chunks.
5. The Qwen model generates a concise answer using the retrieved content.

## Project layout

```text
SupportLLM/
├── flair7_support_bot.py        # Main application
├── flair7_docs/                 # Add Flair manuals and support documents here
├── chroma_db/                   # Generated local ChromaDB data
└── README.md
```

The `flair7_docs/` and `chroma_db/` folders are created by the script as needed. If `flair7_docs/` does not exist, the script creates sample setup, troubleshooting, and maintenance text files.

## Prerequisites

- Windows, macOS, or Linux
- Python installed in the environment used to run the script
- A Conda environment is recommended on Windows
- Internet access the first time you run the program, so Hugging Face can download the embedding and Qwen models

A GPU is optional. The bot can run on CPU, though generation can be slower. A Transformers message saying that parameters were offloaded to CPU is informational: it means the model does not fit entirely in GPU memory.

## Installation

### 1. Activate your Conda environment

On Windows, open **Anaconda Prompt** and activate the environment you intend to use:

```powershell
conda activate videogamebench
```

### 2. Install dependencies in that exact environment

Use the Python executable from the active environment to avoid installing packages into Anaconda `base` by mistake:

```powershell
python -m pip install --upgrade pip
python -m pip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-llms-huggingface chromadb transformers torch sentence-transformers
```

If `python` is not recognized in PowerShell, run these commands in **Anaconda Prompt**, or use the full executable path:

```powershell
& C:/Users/beebs/anaconda3/envs/videogamebench/python.exe -m pip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-llms-huggingface chromadb transformers torch sentence-transformers
```

### 3. Verify LlamaIndex is installed in the intended environment

```powershell
python -m pip show llama-index
```

Check that `Location:` includes the environment path, for example:

```text
C:\Users\beebs\anaconda3\envs\videogamebench\Lib\site-packages
```

If it instead points to `C:\Users\beebs\anaconda3\Lib\site-packages`, it was installed in the base environment. Re-run the installation using the `python -m pip` command above.

## Add Flair 7 documentation

Create the document folder beside the script if it does not already exist:

```powershell
mkdir flair7_docs
```

Copy the Flair operator manual and any relevant support material into it. For example:

```text
flair7_docs/
├── Flair-7.4-Operators-Manual-MRMC-1081-70.pdf
├── release-notes.txt
├── known-issues.txt
└── calibration-guide.pdf
```

Useful materials include:

- Operator manuals
- Software release notes and changelogs
- Known issues and troubleshooting procedures
- Error-code references
- Installation and calibration guides
- Camera compatibility documentation

For a ZIP archive of current and beta software, extract it first. Add documentation-oriented files such as `README`, release notes, changelogs, manuals, and text-based help files to `flair7_docs/`. Do not add installers or executable binaries; they are not useful as RAG source material.

## Run the bot

From the project folder, run:

```powershell
& C:/Users/beebs/anaconda3/envs/videogamebench/python.exe c:/Coding/SupportLLM/flair7_support_bot.py
```

Or, if the environment is activated and `python` works:

```powershell
python flair7_support_bot.py
```

On first run, model downloads and indexing may take longer. The program first runs a test question, then starts the terminal chat because the following line is enabled at the bottom of the script:

```python
chat_with_support_bot()
```

At the prompt, ask a support question, for example:

```text
You: How do I fix error code F001?
You: What are the calibration steps?
You: How should the arm be prepared for storage?
```

Exit by entering one of:

```text
quit
exit
bye
```

## Test-only mode

To run only the built-in test question and not start a chat session, comment out the final line:

```python
# chat_with_support_bot()
```

## Re-indexing documents

After adding, changing, or deleting documentation, delete the existing Chroma database and run the program again so the database is rebuilt from the current source files:

```powershell
Remove-Item -Recurse -Force .\chroma_db
```

Do this only while the bot is not running.

## Troubleshooting

### `ModuleNotFoundError: No module named 'llama_index'`

The package was installed into a different Python environment. Install it using the same Python executable used to launch the bot:

```powershell
& C:/Users/beebs/anaconda3/envs/videogamebench/python.exe -m pip install llama-index
```

### `python` is not recognized in PowerShell

Use Anaconda Prompt after activating the environment, or invoke the environment’s Python executable directly as shown above.

### `HuggingFaceLLM.__init__() got an unexpected keyword argument 'system_message'`

Remove the unsupported `system_message=` argument from `HuggingFaceLLM(...)`. The current version of `flair7_support_bot.py` uses `query_wrapper_prompt` instead.

### Some model parameters are offloaded to CPU

This is a normal Transformers status message. The model is using both GPU and system RAM. It may be slower, but it can still run.

### The answers do not reflect newly added documents

Delete `chroma_db/`, then restart the bot to force a rebuild of the index.

## Current limitations

- The bot is intended to answer from the local documents, but language models can still make mistakes. Verify safety-critical motion, mounting, power, and camera-rigging guidance against the official documentation.
- The bot does not currently display source citations to the user.
- Rebuilding the index on each run can be slow with a large document set.
- The interface is terminal-only; it does not currently provide drag-and-drop uploads.
