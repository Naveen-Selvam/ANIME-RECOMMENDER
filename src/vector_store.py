from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv
load_dotenv()

class VectorStoreBuilder:
    def __init__(self, processed_csv:str, persistant_path="chroma_db"):
        self.processed_csv = processed_csv
        self.persistant_dir = persistant_path
        self.embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_and_save_vectorstore(self):
        try:
            loader = CSVLoader(file_path=self.processed_csv, encoding='utf-8')
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            split_docs = text_splitter.split_documents(documents)

            vector_store = Chroma.from_documents(split_docs, self.embeddings_model, collection_name="anime_recommendations", persist_directory=self.persistant_dir)
            vector_store.persist()
        
        except Exception as e:
            raise Exception(f"Error building vector store: {e}")
        
    def load_vector_store(self):
        try:
            vector_store = Chroma(collection_name="anime_recommendations", persist_directory=self.persistant_dir, embedding_function=self.embeddings_model)
            return vector_store
        except Exception as e:
            raise Exception(f"Error loading vector store: {e}")