import sys

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os

os.environ['OPENAI_API_KEY'] = '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'

# load text data from website
urls = ['https://www.praha6.cz/potrebuji-vyresit/vydani-nebo-vymena-obcanskeho-prukazu-1700-001_8520.html',
        'https://www.praha6.cz/potrebuji-vyresit/uzavreni-manzelstvi-0700-009_8379.html',
        'https://www.praha6.cz/potrebuji-vyresit/pozadavek-na-vydani-stavebniho-povoleni-ohlaseni-0900-020_8418.html',
        'https://www.praha6.cz/potrebuji-vyresit/stavebni-zakon-c-1832006-sb-0900-0001_8408.html',
        'https://www.praha6.cz/potrebuji-vyresit/potrebuji-vyresit-rozcestnik.html',
        'https://www.praha6.cz/potrebuji-vyresit/mistni-poplatek-ze-psa-0400-010_8251.html']
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# get embeddings
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1" #"sentence-transformers/distiluse-base-multilingual-cased-v1"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

## create chroma db
persist_directory = 'db'
db = Chroma.from_documents(texts, hf_embeddings,persist_directory=persist_directory)
db.persist()
#db = None

#query = "Úřední hodiny pro vyzvednutí občanského průkazu?"
#query = "Kde je obřadní síň?"
#query = "informace o Stavebním záměru"
#query = "jaký je poplatek za psa?"
query = "jaký je poplatek za psa chovaného v bytě?"
#docs = db.similarity_search(query)
#docs = db.similarity_search_with_score(query)
docs = db.max_marginal_relevance_search(query)


for d in docs:
    print(d)
#print(docs[0])#.page_content)
#print(docs[1])#.page_content)
#print(docs[2])#.page_content)

#index = VectorstoreIndexCreator().from_loaders([loader])

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])
index.query_with_sources(query)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)


sys.exit()

'''
retriever = db.as_retriever(search_type="mmr")
print(retriever.get_relevant_documents(query)[0])
print(retriever.get_relevant_documents(query)[1])
print(retriever.get_relevant_documents(query)[2])
'''
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.prompts import PromptTemplate


prompt_template = """You are a personal Bot assistant for answering any questions.
If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.
If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details.
Use bullet points if you have to make a list, only if necessary.

QUESTION: {question}

DOCUMENTS:
=========
{context}
=========
Finish by proposing your help for anything else.
"""

prompt_template = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

doc_chain = load_qa_chain(
    llm=OpenAI(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="text-davinci-003",
        temperature=0,
        max_tokens=300,
    ),
    chain_type="stuff",
    prompt=prompt_template,
)

from langchain import VectorDBQA

qa = VectorDBQA(vectorstore=db, combine_documents_chain=doc_chain, k=4)

# Call the VectorDBQA object to generate an answer to the prompt.
result = qa({"query": query})
answer = result["result"]


