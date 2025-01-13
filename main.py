import os

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_KEY"),
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models in less than 100 words",
#         }
#     ],
#     model="llama-3.1-70b-versatile",
# )

# print(chat_completion.choices[0].message.content)

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader)

from llama_index.core.node_parser import (
    SentenceSplitter,
)

reader = SimpleDirectoryReader( 
    input_dir="documents",
)

documents = reader.load_data()

# documents = [Document(text=doc_txt) for doc_txt in texts]

node_parser = SentenceSplitter()
nodes  = node_parser.get_nodes_from_documents(documents, show_progress=True)

print(len(documents))
print(len(nodes))

# import chromadb
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import StorageContext


# db = chromadb.PersistentClient(path="./chroma_db")

# # create collection
# chroma_collection = db.get_or_create_collection("quickstart")

# # assign chroma as the vector_store to the context
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
# index = VectorStoreIndex.from_documents(
#     documents
# )

# # # for n in nodes[:10]:
# # #   print(n)

# # """We extract 5 random documents (chunks) and ask the LLM to generate two questions for each document.
# # In the last version of the code, we ask the LLM to keep the question length short (20 words) and to return the data in JSON format.

# # The `model` variable can be changed to see how different models treat the same prompt. You can check the list of allowed models [here](https://console.groq.com/docs/models).
# # """

# import random
# random_documents = random.choices(nodes, k=5)

# for this_doc in documents:
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Can you give me two short (max 20 words) questions which answer is contained in the following text? Please include in the questions as much context as you can. Give me also the answers. Use JSON format such as: [{question: '[QUESTION]', answer: '[ANSWER]' }, ...].\n\n" + this_doc.text,
#             },
#         ],
#         # model="llama-3.2-90b-text-preview",
#         model="mixtral-8x7b-32768"
#     )

#     with open("questions.json", "a") as f:
#         f.write(chat_completion.choices[0].message.content + "\n")

from llama_index.retrievers.bm25 import BM25Retriever

retriever = BM25Retriever.from_defaults(nodes=nodes,
                                        similarity_top_k=8)

query = "Quali sono le materie del primo anno del corso di laurea in Matematica"
res = retriever.retrieve(query)

res = res[:10]
context = ""
for r in res:
    context += r.text + "\n"

# write context to file
with open("context.txt", "w") as f:
    f.write(context)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"{query}\nUse only the documents provided below and just give me the answer.\n\n{context}",
        },
    ],
    model="llama-3.1-70b-versatile",
)

print(chat_completion.choices[0].message.content)

# while True:
#     query = input("> ")
#     res = retriever.retrieve(query)
#     res = res[:10]
#     context = ""
#     for r in res:
#         context += r.text + "\n"

#     # write context to file
#     with open("context.txt", "w") as f:
#         f.write(context)

#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"{query}\nUse only the documents provided below and just give me the answer.\n\n{context}",
#             },
#         ],
#         model="llama-3.3-70b-versatile",
#     )

#     print(chat_completion.choices[0].message.content)