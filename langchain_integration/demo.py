from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

db = Chroma(persist_directory="./db-hormozi",
            embedding_function=embeddings)


# llm = ChatOllama(model="gemma2:2b")
# print(llm.invoke("Why is the sky blue?"))
#
#
# def search_elastic_search():
#     pass
