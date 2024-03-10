
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.vectorstores import Chroma

loader = DirectoryLoader("../../news_articles", glob="./*.txt", loader_cls=TextLoader)

document = loader.load()

print(len(document))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text = text_splitter.split_documents(document)

print(len(text))

print(text[:5])  # Print the first 5 items of the 'text' list

# model = SentenceTransformer("paraphrase-distilroberta-base-v2") # Load a pre-trained model

# import chromadb
# chroma_client = chromadb.Client()


# chroma_client = chromadb.PersistentClient(path="db")
#
# from chromadb import Documents, EmbeddingFunction, Embeddings
#
#
# class MyEmbeddingFunction(EmbeddingFunction):
#     def __call__(self, input: Documents) -> Embeddings:
#         # embed the documents somehow
#         return embeddings
#

# emb_fn = MyEmbeddingFunction()

# collection = chroma_client.create_collection(name="my_collection")
# collection = chroma_client.get_or_create_collection(name="my_collection1")
# collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=emb_fn)
#
# textList = []
#
# for txt in text:
#     textList.append(txt.page_content)
#
# metadatas = []
#
# for md in text:
#     metadatas.append(md.metadata)
#
# embeddings = model.encode(textList)
# print(f"embeddings:", embeddings)

# ids = ["id" + str(i + 1) for i in range(len(textList))]

# print(len(embeddings))  # Should be 233
# print(len(textList))
# print(len(metadatas))
# print(len(ids))
# print(embeddings)
# collection.add(
#     embeddings=embeddings,
#     documents=textList,
#     metadatas=metadatas,
#     ids=ids
# )





# collection = chroma_client.get_collection(name="my_collection", )

# collection.save()

# results = collection.query(
#     query_texts=["What is AI-powered chatbot"],
#     n_results=2
# )



HUGGING_API_TOKEN = "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"

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=HUGGING_API_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(documents=text, embedding_function=huggingface_ef, persist_directory="db")

