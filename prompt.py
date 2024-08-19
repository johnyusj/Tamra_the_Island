from openai import OpenAI
from langchain import hub
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_upstage import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



client = OpenAI(
  api_key="up_kGzaFmJXhtbE3hAC3oqTi6uZZSGvb",
  base_url="https://api.upstage.ai/v1/solar"
)


loader = CSVLoader(
    file_path="optimized_jeju_itinerary.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["hostel","location","price","locations","restaurants"],
    },
    encoding="utf-8"  
)

data = loader.load()


print("Adfdfdaf1")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)

embeddings = UpstageEmbeddings(model ="solar-embedding-1-large-query", api_key="up_kGzaFmJXhtbE3hAC3oqTi6uZZSGvb")


print("Adfdfdaf2")

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)


print("Adfdfdaf4")

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

print("Adfdfdaf3")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | client
    | StrOutputParser()
)



stream = rag_chain.invoke(
  messages=[
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": """Plan a itinerary from the jeju_itinerary_data.csv file that keeps track budget and location. Format it like this
| Time      | Day 1        | Day 2        | Day 3        | Day 4        | Day 5        | Day 6        | Day 7        |
|-----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| 7:00 AM   |              |              |              |              |              |              |              |
| 8:00 AM   |              |              |              |              |              |              |              |
| 9:00 AM   |              |              |              |              |              |              |              |
| 10:00 AM  |              |              |              |              |              |              |              |
| 11:00 AM  |              |              |              |              |              |              |              |
| 12:00 PM  |              |              |              |              |              |              |              |
| 1:00 PM   |              |              |              |              |              |              |              |
| 2:00 PM   |              |              |              |              |              |              |              |
| 3:00 PM   |              |              |              |              |              |              |              |
| 4:00 PM   |              |              |              |              |              |              |              |
| 5:00 PM   |              |              |              |              |              |              |              |
| 6:00 PM   |              |              |              |              |              |              |              |
| 7:00 PM   |              |              |              |              |              |              |              |
| 8:00 PM   |              |              |              |              |              |              |              |
| 9:00 PM   |              |              |              |              |              |              |              |
| 10:00 PM  |              |              |              |              |              |              |              |
| 11:00 PM  |              |              |              |              |              |              |              |"""
    }
  ],
  stream=True,
)

for chunk in stream:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")