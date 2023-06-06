import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
import logging
import os

def init_hf_embeddings(model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                       device: str = "cpu") -> HuggingFaceEmbeddings:
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf_embeddings

def init_openai_embeddings(model_name: str = "text-embeddings-ada-002", openai_api_key: str = os.getenv('OPENAI_API_KEY')) -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(
        document_model_name = model_name,
        query_model_name = model_name,
        open_api_key = openai_api_key
    )
    return embeddings

def init_sentence_embeddings(model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                            device: str = "cpu") -> SentenceTransformer:
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    model = SentenceTransformer(
        model_name,
        device=device
    )
    return model



class DocumentDB:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 device: str = "cup"):
        self.device = device
        self.model_name = model_name
        self.index = None
        self.clustering = None
        self.quantizer = None
        self.embedder = init_sentence_embeddings(model_name=model_name)
        self.embeddings_dim = self.embedder.get_sentence_embedding_dimension()
        self.sentence_map = []

        # Configure logging
        log_directory = "log"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        log_file_path = os.path.join(log_directory, "DocumentDB.log")

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        # Log initialization information
        self.logger.info(f"Initializing DocumentDB with embeddings model {self.model_name} with device {self.device}")

    def __str__(self):
        return f"DocumentDB:\nEmbeddings model: {self.model_name}\n" \
               f"Embeddings Dimension: {self.embeddings_dim}\n" \
               f"Number of Sentences: {len(self.sentence_map)}\n" \
               f"Index: {self.index}\n" \
               f"Quantizer: {self.quantizer}\n" \
               f"Clustering: {self.clustering}"

    # HOW TO CHOOSE INDEX https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    # few searches + medium dataset + exact results :=> Flat
    # memory concern - If you have a lots of RAM or the dataset is small :=> HNSW
    # dataset size: ~ up to 1M vectors: IVF
    def build_index(self, documents, embeddings):
        self.index = faiss.IndexFlatL2(self.embeddings_dim)
        self.index.add(embeddings)
        self.sentence_map = documents

    def build_clustering(self, num_clusters):
        self.clustering = KMeans(n_clusters=num_clusters)
        embeddings = self.index.reconstruct_n(0, len(self.sentence_map))
        self.clustering.fit(embeddings)

    def get_cluster(self, index):
        if self.clustering is None:
            raise ValueError("Clustering has not been built yet.")
        return self.clustering.labels_[index]

    def search_documents(self, query_embedding, top_k):
        if self.index is None:
            raise ValueError("Index has not been built yet.")
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices





class DocumentIndex:
    def __init__(self, embeddings_dim, index='IndexFlatL2'):
        if index == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embeddings_dim)
        else:
            #@TODO: add other indexes like IVFPQ [https://github.com/facebookresearch/faiss/wiki/Faiss-indexes]
            self.index = faiss.IndexFlatL2(embeddings_dim)
    def add_documents(self, documents):
        #@TODO: transfrom documents into embeddings
        self.index.add(documents)
        # Store the document information in a separate data structure

    def search_documents(self, query_embedding, top_k):
        distances, indices = self.index.search(query_embedding, top_k)
        # Retrieve the document information based on the indices
        return distances, indices

class QuestionAnsweringSystem:
    def __init__(self, model_name):
        self.nlp = pipeline("question-answering", model=model_name)

    def process_question(self, question, documents, document_embeddings):
        query_embedding = self.nlp(question)["question_embedding"]

        # Search the document index for related documents
        distances, indices = documents.search_documents(query_embedding, top_k=5)

        # Retrieve the relevant chunks of text based on the document indices
        relevant_texts = [documents.get_text(index) for index in indices]

        # Pass the relevant texts to the question answering model
        answers = self.nlp(question, relevant_texts)

        # Return the answers
        return answers