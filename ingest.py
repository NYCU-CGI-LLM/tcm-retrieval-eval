
import os
import logging
import argparse
import json
import requests
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv(override=True)

# --- Constants ---
# The base directory where all ChromaDB databases will be stored
BASE_PERSIST_DIRECTORY = "chroma_dbs"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100 # Process N documents at a time to avoid API rate limits

# Supported embedding providers and models
EMBEDDING_PROVIDERS = {
    "openai": {
        "models": [
            "text-embedding-3-large"
        ],
        "default": "text-embedding-3-large"
    },
    "vllm": {
        "models": [
            "Qwen3-Embedding-8B"
        ],
        "default": "Qwen3-Embedding-8B",
        "base_url": "http://localhost:8010/v1"
    },
    "custom": {
        "models": [
            "Qwen3-Embedding-0.6B-base",
            "Qwen3-Embedding-0.6B-finetuned",
            "Qwen3-Embedding-4B-base",
            "Qwen3-Embedding-4B-finetuned"
        ],
        "default": "Qwen3-Embedding-0.6B-finetuned",
        "base_url": "http://localhost:8000/v1"
    },
    "huggingface": {
        "models": [
            "BAAI/bge-large-zh-v1.5"
        ],
        "default": "BAAI/bge-large-zh-v1.5"
    }
}

class CustomEmbeddings(Embeddings):
    """Custom embeddings class that works with your fine-tuned model API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.endpoint = f"{base_url}/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = []
        for text in texts:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "input": text}
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "input": text}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

# Valid field names for SelectiveRAG mode
VALID_FIELDS = {"Name", "Definition", "Typical_performance", "Common_disease"}

# Define field name mappings for Chinese labels
FIELD_LABELS = {
    'Name': '名称',
    'Definition': '定义', 
    'Typical_performance': '典型表现',
    'Common_disease': '常见疾病'
}
    
def load_and_split_documents(json_file_path: str, is_jsonl: bool, jq_schema: str | None, selected_fields: list[str] | None = None):
    """Loads documents from the JSON file, with each item as a separate document."""
    logging.info(f"Loading documents from {json_file_path}...")
    
    # Log the mode being used
    if selected_fields:
        logging.info(f"SelectiveRAG mode: Using fields {selected_fields}")
    elif jq_schema:
        logging.info(f"Schema mode: Using jq_schema {jq_schema}")
    else:
        logging.info("SimpleRAG mode: Using default field combination")
    
    documents = []
    
    if is_jsonl:
        # Process JSONL file (one JSON object per line)
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        # Create content for embedding (page_content)
                        page_content = create_content_from_data(data, jq_schema, selected_fields)
                        # Create complete content for prompt (original_content)
                        original_content = create_complete_content(data)
                        
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                "source": json_file_path,
                                "line_number": line_num,
                                "id": data.get("id", f"line_{line_num}"),
                                "original_content": original_content,
                                "rag_mode": "selective" if selected_fields else ("schema" if jq_schema else "simple"),
                                "selected_fields": ",".join(selected_fields) if selected_fields else None
                            }
                        )
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse line {line_num}: {e}")
    else:
        # Process regular JSON file (array of objects)
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            
        for index, data in enumerate(data_list):
            # Create content for embedding (page_content)
            page_content = create_content_from_data(data, jq_schema, selected_fields)
            # Create complete content for prompt (original_content)
            original_content = create_complete_content(data)
            
            doc = Document(
                page_content=page_content,
                metadata={
                    "source": json_file_path,
                    "index": index,
                    "id": data.get("id", f"item_{index}"),
                    "original_content": original_content,
                    "rag_mode": "selective" if selected_fields else ("schema" if jq_schema else "simple"),
                    "selected_fields": ",".join(selected_fields) if selected_fields else None
                }
            )
            documents.append(doc)
    
    logging.info(f"Loaded {len(documents)} documents (each original data item as one document).")
    return documents


def create_content_from_data(data: dict, jq_schema: str | None, selected_fields: list[str] | None = None) -> str:
    """Create document content from data based on jq_schema, selected fields, or default format."""
    
    # Priority 1: Use selected_fields if provided (SelectiveRAG mode)
    if selected_fields:
        return create_selective_content(data, selected_fields)
    
    # Priority 2: Use jq_schema if provided (existing functionality)
    if jq_schema:
        # If jq_schema is provided, we need to apply it to extract the content
        # For now, we'll handle the specific case from load_data.sh
        if '"名称: " + .Name' in jq_schema:
            # This is the specific schema from load_data.sh
            content = f"名称: {data.get('Name', '')}\n定义: {data.get('Definition', '')}\n典型表现: {data.get('Typical_performance', '')}\n常见疾病: {data.get('Common_disease', '')}"
        else:
            # For other schemas, try to use the 'text' field or convert to string
            content = data.get('text', str(data))
    else:
        # Priority 3: Default format (SimpleRAG mode - existing functionality)
        if 'Name' in data and 'Definition' in data:
            # This looks like syndrome data
            content = f"名称: {data.get('Name', '')}\n定义: {data.get('Definition', '')}\n典型表现: {data.get('Typical_performance', '')}\n常见疾病: {data.get('Common_disease', '')}"
        else:
            # Fallback to 'text' field or string representation
            content = data.get('text', str(data))
    
    return content


def create_selective_content(data: dict, selected_fields: list[str]) -> str:
    """Create selective content based on specified fields (SelectiveRAG mode)."""
    
    # Validate that all selected fields exist in the data
    available_fields = set(data.keys())
    missing_fields = [field for field in selected_fields if field not in available_fields]
    if missing_fields:
        logging.warning(f"Missing fields in data: {missing_fields}. Available fields: {list(available_fields)}")
    
    # Build content with selected fields
    content_parts = []
    for field in selected_fields:
        if field in data and data[field]:  # Only include fields that exist and are not empty
            label = FIELD_LABELS.get(field, field)  # Use Chinese label if available, otherwise use field name
            field_content = str(data[field]).strip()
            if field_content:  # Only add non-empty content
                content_parts.append(f"{label}: {field_content}")
    
    if not content_parts:
        # Fallback if no valid content found
        logging.warning("No valid content found for selected fields.")
        return ""
    
    return "\n".join(content_parts)


def create_complete_content(data: dict) -> str:
    """Create complete content from data including all available fields for original_content."""
    
    # Define the standard field order for consistency
    standard_fields = ['Name', 'Definition', 'Typical_performance', 'Common_disease']
    
    content_parts = []
    
    # Add standard fields in order if they exist
    for field in standard_fields:
        if field in data and data[field]:
            label = FIELD_LABELS.get(field, field)
            field_content = str(data[field]).strip()
            if field_content:
                content_parts.append(f"{label}: {field_content}")
    
    # Add any additional fields that aren't in the standard list
    additional_fields = []
    for field, value in data.items():
        if field not in standard_fields and value and str(value).strip():
            # Skip common metadata fields
            if field not in ['id', 'source', '_id']:
                additional_fields.append(f"{field}: {str(value).strip()}")
    
    if additional_fields:
        content_parts.extend(additional_fields)
    
    if not content_parts:
        # Fallback if no valid content found
        return str(data)
    
    return "\n".join(content_parts)


def get_provider_and_model(embedding_model: str) -> tuple[str, str]:
    """Determine the provider and model name from the embedding_model string."""
    
    # Check if it's a provider:model format (e.g., "vllm:Qwen3-Embedding-8B")
    if ":" in embedding_model:
        provider, model = embedding_model.split(":", 1)
        if provider in EMBEDDING_PROVIDERS:
            if model in EMBEDDING_PROVIDERS[provider]["models"]:
                return provider, model
            else:
                logging.error(f"Model '{model}' not supported for provider '{provider}'")
                logging.info(f"Supported models for {provider}: {EMBEDDING_PROVIDERS[provider]['models']}")
                raise ValueError(f"Unsupported model '{model}' for provider '{provider}'")
        else:
            logging.error(f"Unsupported provider: {provider}")
            logging.info(f"Supported providers: {list(EMBEDDING_PROVIDERS.keys())}")
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Check if it's a direct model name and find the matching provider
    for provider, config in EMBEDDING_PROVIDERS.items():
        if embedding_model in config["models"]:
            return provider, embedding_model
    
    # If not found, assume it's an OpenAI model (for backward compatibility)
    logging.warning(f"Model '{embedding_model}' not found in predefined models. Assuming OpenAI provider.")
    return "openai", embedding_model


def initialize_embeddings(embedding_model: str):
    """Initialize the appropriate embeddings class based on the model specification."""
    
    provider, model = get_provider_and_model(embedding_model)
    
    logging.info(f"Initializing {provider} embeddings with model: {model}")
    
    if provider == "openai":
        # Check if OPENAI_API_KEY is available
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY environment variable not set for OpenAI embeddings.")
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        return OpenAIEmbeddings(model=model)
    
    elif provider == "vllm":
        # Use OpenAI-compatible client pointing to vLLM server
        base_url = EMBEDDING_PROVIDERS[provider]["base_url"]
        logging.info(f"Using vLLM server at: {base_url}")
        
        # Try to create OpenAI embeddings with different parameter names for compatibility
        try:
            # First try with newer langchain-openai version parameters
            return OpenAIEmbeddings(
                model=model,
                api_key="unused",  # vLLM doesn't require a real API key
                base_url=base_url
            )
        except TypeError:
            try:
                # Try with older parameter names
                return OpenAIEmbeddings(
                    model=model,
                    openai_api_key="unused",
                    openai_api_base=base_url
                )
            except Exception as e:
                logging.error(f"Failed to initialize vLLM embeddings: {e}")
                logging.info(f"Please ensure vLLM server is running at {base_url}")
                raise
    
    elif provider == "custom":
        # Use custom embeddings class for fine-tuned model server
        base_url = EMBEDDING_PROVIDERS[provider]["base_url"]
        logging.info(f"Using custom fine-tuned model server at: {base_url}")
        
        try:
            return CustomEmbeddings(base_url=base_url, model=model)
        except Exception as e:
            logging.error(f"Failed to initialize custom embeddings: {e}")
            logging.info(f"Please ensure custom server is running at {base_url}")
            raise
    
    elif provider == "huggingface":
        # Use local Hugging Face model
        logging.info(f"Loading Hugging Face model: {model}")
        
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def list_available_models():
    """List all available embedding models by provider."""
    print("Available embedding models by provider:")
    print("=" * 50)
    
    for provider, config in EMBEDDING_PROVIDERS.items():
        print(f"\n{provider.upper()}:")
        for model in config["models"]:
            default_marker = " (default)" if model == config["default"] else ""
            print(f"  - {provider}:{model}{default_marker}")
    
    print(f"\nUsage examples:")
    print(f"  --embedding-model openai:text-embedding-3-large")
    print(f"  --embedding-model vllm:Qwen3-Embedding-8B")
    print(f"  --embedding-model custom:Qwen3-Embedding-0.6B-finetuned")
    print(f"  --embedding-model huggingface:BAAI/bge-large-zh-v1.5")
    print(f"  --embedding-model text-embedding-3-large  # OpenAI provider")


def create_and_store_embeddings(docs, db_name: str, embedding_model: str = None):
    """Creates embeddings and stores them in a named ChromaDB, processing complete documents in batches."""
    if not docs:
        logging.warning("No documents to process. Exiting.")
        return

    # Construct the path for the new database
    persist_directory = os.path.join(BASE_PERSIST_DIRECTORY, db_name)
    
    # Initialize embeddings based on the specified model
    embeddings = initialize_embeddings(embedding_model or DEFAULT_EMBEDDING_MODEL)
    
    # Create the base directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    total_docs = len(docs)
    logging.info(f"Processing {total_docs} complete documents in batches of {BATCH_SIZE}...")
    
    # Initialize the DB with the first batch
    first_batch = docs[:BATCH_SIZE]
    db = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info(f"Successfully initialized ChromaDB with the first batch of {len(first_batch)} documents.")

    # Add remaining documents in subsequent batches
    if total_docs > BATCH_SIZE:
        remaining_indices = range(BATCH_SIZE, total_docs, BATCH_SIZE)
        logging.info("Adding remaining document batches...")
        for i in tqdm(remaining_indices, desc="Adding batches"):
            batch = docs[i:i + BATCH_SIZE]
            if not batch:
                continue
            
            # The progress bar replaces the need for per-batch logging
            db.add_documents(documents=batch)

    logging.info(f"Successfully created and stored all embeddings in ChromaDB at '{persist_directory}'.")

def parse_fields(fields_str: str) -> list[str]:
    """Parse comma-separated fields string into a list."""
    if not fields_str:
        return None
    
    # Split by comma and clean up whitespace
    fields = [field.strip() for field in fields_str.split(',') if field.strip()]
    
    # Validate field names
    invalid_fields = [field for field in fields if field not in VALID_FIELDS]
    
    if invalid_fields:
        logging.error(f"Invalid field names: {invalid_fields}")
        logging.error(f"Valid field names are: {', '.join(sorted(VALID_FIELDS))}")
        raise ValueError(f"Invalid field names: {invalid_fields}")
    
    return fields


def main():
    """Main function to run the ingestion process."""
    parser = argparse.ArgumentParser(
        description="Ingest a JSON or JSONL file into a named ChromaDB with SelectiveRAG support and multiple embedding model options.",
        epilog="""
Examples:
  # SimpleRAG mode (default - all fields):
  python ingest.py --json-file data.json --db-name my_db
  
  # SelectiveRAG mode (specific fields only):
  python ingest.py --json-file data.json --db-name my_db --fields Name,Definition
  python ingest.py --json-file data.json --db-name my_db --fields Name,Typical_performance
  
  # Using different embedding models:
  python ingest.py --json-file data.json --db-name my_db --embedding-model vllm:Qwen3-Embedding-8B
  python ingest.py --json-file data.json --db-name my_db --embedding-model custom:Qwen3-Embedding-0.6B-finetuned
  python ingest.py --json-file data.json --db-name my_db --embedding-model huggingface:BAAI/bge-large-zh-v1.5
  python ingest.py --json-file data.json --db-name my_db --embedding-model openai:text-embedding-3-large
  
  # List available models:
  python ingest.py --list-models
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json-file", 
        type=str, 
        required=False,  # Make it optional so --list-models can work without it
        help="Path to the input JSON or JSONL file."
    )
    parser.add_argument(
        "--db-name", 
        type=str, 
        required=False,  # Make it optional so --list-models can work without it
        help="A unique name for the database."
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Set this flag if the input file is in JSONL format (one JSON object per line)."
    )
    parser.add_argument(
        "--jq-schema",
        type=str,
        default=None,
        help="Optional: The jq schema to extract text content. Defaults will be used if not provided."
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="SelectiveRAG: Comma-separated list of fields to include in embeddings (e.g., 'Name,Definition' or 'Name,Typical_performance'). Valid fields: Name, Definition, Typical_performance, Common_disease"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model to use. Format: 'provider:model' or just 'model'. Default: {DEFAULT_EMBEDDING_MODEL}. Use --list-models to see all options."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available embedding models and exit."
    )
    args = parser.parse_args()

    # Handle --list-models option
    if args.list_models:
        list_available_models()
        return

    # Require json-file and db-name for ingestion
    if not args.json_file or not args.db_name:
        logging.error("Both --json-file and --db-name are required for ingestion.")
        return

    # Validate embedding model and check requirements
    try:
        provider, model = get_provider_and_model(args.embedding_model)
        
        # Check provider-specific requirements
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY environment variable not set for OpenAI embeddings.")
            logging.error("Please create a .env file and add your OpenAI API key.")
            return
            
        logging.info(f"Using {provider} embedding model: {model}")
        
    except ValueError as e:
        logging.error(str(e))
        logging.info("Use --list-models to see available embedding models.")
        return

    # Parse selected fields if provided
    try:
        selected_fields = parse_fields(args.fields) if args.fields else None
    except ValueError:
        return

    # Validate arguments combination
    if selected_fields and args.jq_schema:
        logging.warning("Both --fields and --jq-schema specified. --fields takes priority.")
    
    documents_to_process = load_and_split_documents(args.json_file, args.jsonl, args.jq_schema, selected_fields)
    create_and_store_embeddings(documents_to_process, args.db_name, args.embedding_model)

if __name__ == "__main__":
    main() 