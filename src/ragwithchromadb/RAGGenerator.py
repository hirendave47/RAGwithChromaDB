from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.chat_models import ChatOllama
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.vectorstores import Chroma
import shutil

MODEL = "llama2:13b"
# MODEL = "gemma:instruct"
# MODEL = "gemma:7b"
llm = ChatOllama(model=MODEL)

embeddings = HuggingFaceEmbeddings()


def create_database(target_file):
    persist_directory = "db"
    shutil.rmtree(persist_directory)
    # target_files = "./*.txt"
    # target_dir = "../../news_articles"
    # loader = DirectoryLoader(target_dir, glob=target_files, loader_cls=TextLoader)
    loader = TextLoader(target_file, encoding="utf-8")
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(document)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()


def load_database(query):
    # query = "How much money did Microsoft raise?"
    persist_directory = "db"
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                           return_source_documents=True)
    llm_response = qa_chain.invoke(query)
    unique_sources = set()
    unique_sources_string = ""
    for source in llm_response["source_documents"]:
        if source.metadata["source"] not in unique_sources:
            unique_sources.add(source.metadata["source"])
            unique_sources_string += source.metadata["source"]
            unique_sources_string += "\n"
            print(source.metadata["source"])
    print(llm_response["result"])
    print(source.metadata["source"] for source in unique_sources)
    return llm_response["result"] + "\n\nSources: \n" + unique_sources_string


def process_llm_response(llm_response):
    print(f"\nLLM Response: \n", llm_response["result"])
    print(f"\nSource Documents:")
    # Create an empty set to store unique sources
    unique_sources = set()

    for source in llm_response["source_documents"]:
        # Check if the source is already in the set
        if source.metadata["source"] not in unique_sources:
            unique_sources.add(source.metadata["source"])
            print(source.metadata["source"])
    print(f"\nunique_sources: {unique_sources}")

    # for source in llm_response["source_documents"]:
    #     print(source.metadata["source"])


# load_database("db", "How much money did Microsoft raise?")



# loader = DirectoryLoader("../../news_articles", glob="./*.txt", loader_cls=TextLoader)
#
# document = loader.load()
#
# print(len(document))
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#
# text = text_splitter.split_documents(document)
#
# print(len(text))
#
# print(text[:5])  # Print the first 5 items of the 'text' list
#
#
# HUGGING_API_TOKEN = ""
#
# huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     api_key=HUGGING_API_TOKEN,
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
#
# vectordb = Chroma.from_documents(documents=text, embedding_function=huggingface_ef, persist_directory="db")
#



# from langchain_community import Document

# loader = DirectoryLoader("../../news_articles", glob="./*.txt", loader_cls=TextLoader)
# raw_documents = loader.load()  # Assuming this loads your text

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# documents = text_splitter.split_documents(raw_documents)


### HUGGING_API_TOKEN = "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"
### huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
###     api_key=HUGGING_API_TOKEN,
###     model_name="sentence-transformers/all-MiniLM-L6-v2"
### )

### vectordb = Chroma.from_documents(documents=text, embedding_function=huggingface_ef, persist_directory="db")
### vectordb = Chroma.from_documents(documents, huggingface_ef, "db")

# embeddings = HuggingFaceEmbeddings()
# vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
# vectorstore.persist()

# Retriever
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# retriever = vectordb.as_retriever()
# retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# docs = retriever.get_relevant_documents("How much money did Microsoft raise?")
#
# print(docs)

# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


# def process_llm_response(llm_response):
#     print(f"llm response: ", llm_response["result"])
#     print(f"source documents: ")
#     for source in llm_response["source_documents"]:
#         print(source.metadata["source"])
#
#
# query = "How much money did Microsoft raise?"
# query1 = "What does OpenAI describes the forthcoming offering as?"
# llm_response = qa_chain(query)
# process_llm_response(llm_response)
