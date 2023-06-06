import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import tiktoken
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import hashlib
from langchain.document_loaders import UnstructuredURLLoader
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

class PreprocessText:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        #self.tokenizer = tiktoken.get_encoding("cl100k_base") # best to use with "cl100k_base" with gpt-3.5-turbo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tokens_len,
            separators=['\n\n', '\n', ' ', '']
        )

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def tokens_len(self, text: str) -> int:
        return len(self.tokenize(text))

    # this method creates a dictionary with key that is unique id and values
    def load_html_webpages(self, urls : List[str]) -> List[Dict[str, str]]:
        """
        :param urls: list of urls to parse html from
        :return: documents: list of dict objects read to be embedded
        """
        documents  = []
        for url in tqdm(urls):
            print(f"Processing data from url:{url}")
            loader = UnstructuredURLLoader(urls=[url])
            pages = loader.load()

            if len(pages) > 1:
                print('Pages list has more than 1 object ! Check it')
            for i, t in enumerate(self.text_splitter.split_text(pages[0].page_content)):
                documents.append({
                    'id' : self.to_hash(pages[0].metadata['source'], i),
                    'text' : t,
                    'source': pages[0].metadata['source']
                })
        print('Number of unique chunks loaded from the urls:', len(documents))
        return documents

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, chunk: str,
                            device: str = 'cpu') -> torch.tensor:
        """
        :param chunk: @TODO: should be a list of sentences instead of one text
        :param embedder:
        :return: torch.tensor with text representation in the embedding space
        source: https://huggingface.co/learn/nlp-course/chapter5/6?fw=pt
        """
        device = torch.device(device)
        self.model.to(device)

        encoded_input = self.tokenizer(
            chunk,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)



    def to_hash(self, key : str, chunk_id : int) -> str:
        m = hashlib.md5()
        m.update(key.encode('utf-8'))
        #return m.hexdigest()[:12], m.hexdigest()
        return f"{m.hexdigest()}-{chunk_id:03d}"




