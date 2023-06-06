import sys

from src.PreprocessText import PreprocessText
from src.DocumentDB import DocumentDB
from langchain.document_loaders import UnstructuredURLLoader

# load text data from website
urls = ['https://www.praha6.cz/potrebuji-vyresit/mistni-poplatek-ze-psa-0400-010_8251.html',
        'https://www.praha6.cz/potrebuji-vyresit/vydani-nebo-vymena-obcanskeho-prukazu-1700-001_8520.html',
        'https://www.praha6.cz/potrebuji-vyresit/uzavreni-manzelstvi-0700-009_8379.html',
        'https://www.praha6.cz/potrebuji-vyresit/pozadavek-na-vydani-stavebniho-povoleni-ohlaseni-0900-020_8418.html',
        'https://www.praha6.cz/potrebuji-vyresit/stavebni-zakon-c-1832006-sb-0900-0001_8408.html',
        'https://www.praha6.cz/potrebuji-vyresit/potrebuji-vyresit-rozcestnik.html']
loader = UnstructuredURLLoader(urls=urls)
pages = loader.load()

documents = [page.page_content for page in pages]
prec = PreprocessText(model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1')

docs = prec.load_html_webpages(urls)


#embeddings_dataset = docs.map(lambda x: {"embeddings": prec.get_embeddings(x["text"]).detach().cpu().numpy()[0]})
for record in docs:
        text = record['text']
        record['embed'] = prec.get_embeddings([text]).cpu().detach().numpy()
print(docs)

#print('number of unique vectors:', len(docs))
sample_text = docs[0]['text']

# example of the embeddings
embed = prec.get_embeddings(sample_text) #.cpu().detach().numpy()
print(embed)
print(type(embed))
print(embed.shape)


question = "Jake jsou poplatky za psa?"
question_embedding = prec.get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape

#
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)


