# Tech Support Bot with LlamaIndex + ChromaDB + Qwen3:4B
# Install required packages first:
# pip install llama-index chromadb transformers torch

import os
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Set up the embedding model (for converting text to vectors)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"  # Good, lightweight embedding model
)

# Step 2: Set up Qwen3:4B as the LLM
def setup_qwen_llm():
    model_name = "Qwen/Qwen2.5-3B-Instruct"  # Using Qwen2.5-3B as it's more available
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Create LlamaIndex LLM wrapper
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=4096,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        query_wrapper_prompt="<|im_start|>user\\n{query_str}<|im_end|>\\n<|im_start|>assistant\\n"
    )
    
    return llm

# Step 3: Set up ChromaDB vector store
def setup_vector_store():
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("flair7_docs")
    
    # Create ChromaDB vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return storage_context

# Step 4: Load and index your Flair 7 documentation
def create_index_from_docs(docs_folder="./flair7_docs"):
    """
    Load documents from a folder and create a searchable index.
    Put your Flair 7 manuals, PDFs, text files, and release notes in the ./flair7_docs folder.
    """
    # Load documents
    if os.path.exists(docs_folder):
        documents = SimpleDirectoryReader(docs_folder).load_data()
        print(f"Loaded {len(documents)} documents")
    else:
        # Create sample documents if folder doesn't exist
        print("No docs folder found. Creating sample documents...")
        os.makedirs(docs_folder, exist_ok=True)
        
        # Sample Flair 7 documentation
        sample_docs = {
            "setup_guide.txt": """
Flair 7 Setup Guide

1. Unbox your Flair 7 robotic arm
2. Connect the power cable to the base unit
3. Connect USB-C cable to your computer
4. Install the Flair Control software
5. Calibrate the arm using the auto-calibration feature
6. Mount your camera using the universal camera mount

Common Setup Issues:
- If arm doesn't respond, check USB connection
- Calibration fails: Ensure arm has full range of motion
- Camera mount loose: Tighten the thumb screws
            """,
            "troubleshooting.txt": """
Flair 7 Troubleshooting

Error Code F001: Communication Error
- Check USB cable connection
- Restart Flair Control software
- Try different USB port

Error Code F002: Calibration Failed
- Clear obstacles around arm
- Reset arm to home position
- Run calibration again

Camera Mount Issues:
- Loose mount: Tighten all screws
- Camera not level: Use built-in bubble level
- Vibration during recording: Enable stabilization mode
            """,
            "maintenance.txt": """
Flair 7 Maintenance

Weekly:
- Clean camera mount with soft cloth
- Check all cable connections
- Test full range of motion

Monthly:
- Update firmware if available
- Lubricate joints using provided lubricant only
- Inspect cables for wear

Storage:
- Power down completely
- Store in protective case
- Keep in dry environment
            """
        }
        
        for filename, content in sample_docs.items():
            with open(os.path.join(docs_folder, filename), "w", encoding="utf-8") as f:
                f.write(content)
        
        documents = SimpleDirectoryReader(docs_folder).load_data()
        print(f"Created and loaded {len(documents)} sample documents")
    
    # Set up vector store and LLM
    storage_context = setup_vector_store()
    Settings.llm = setup_qwen_llm()
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    
    return index

# Step 5: Create the query engine
def create_support_bot():
    index = create_index_from_docs()
    
    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Retrieve top 3 most relevant chunks
        response_mode="compact"
    )
    
    return query_engine

# Step 6: Interactive chat function
def chat_with_support_bot():
    print("Setting up Flair 7 Tech Support Bot...")
    print("This may take a moment to load the model...")
    
    try:
        query_engine = create_support_bot()
        print("\n🤖 Flair 7 Tech Support Bot is ready!")
        print("Ask me anything about your Flair 7 robotic arm.")
        print("Type 'quit' to exit.\n")
        
        while True:
            question = input("You: ")
            if question.lower() in ["quit", "exit", "bye"]:
                print("Thanks for using Flair 7 Support! Goodbye!")
                break
            
            print("Bot: Thinking...")
            try:
                response = query_engine.query(question)
                print(f"Bot: {response}\n")
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error: {e}\n")
                
    except Exception as e:
        print(f"Error setting up bot: {e}")
        print("Make sure you have all required packages installed:")
        print(
            "pip install llama-index "
            "llama-index-vector-stores-chroma "
            "llama-index-embeddings-huggingface "
            "llama-index-llms-huggingface "
            "chromadb transformers torch sentence-transformers"
        )

# Example usage
if __name__ == "__main__":
    # Quick test before interactive mode
    print("Creating Flair 7 Support Bot...")
    
    try:
        query_engine = create_support_bot()
        
        # Test question
        test_question = "How do I fix error code F001?"
        print(f"\nTest Question: {test_question}")
        response = query_engine.query(test_question)
        print(f"Bot Response: {response}")
        
        print("\n✅ Setup successful! Starting interactive mode...")
        
    except Exception as e:
        print(f"Setup error: {e}")
        print("Note: This requires GPU/CPU resources to run the Qwen model locally.")
    
    chat_with_support_bot()
