from langchain.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
import os

# api key not valid
os.environ['OPENAI_API_KEY'] = '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'
## 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

loader = WebBaseLoader('https://www.praha6.cz/potrebuji-vyresit/vydani-nebo-vymena-obcanskeho-prukazu-1700-001_8520.html')

index = VectorstoreIndexCreator().from_loaders([loader])


query = "V kolik hodin si mohu prijit vyzvednout obcansky prukaz"
print(index.query(query))
