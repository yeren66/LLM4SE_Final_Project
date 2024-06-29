import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import Milvus
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 加载嵌入模型
model_kwargs = {'local_files_only': True}
embedding = HuggingFaceEmbeddings(model_name="/Users/mac/Desktop/RAG_experiment/all-MiniLM-L12-v2", model_kwargs=model_kwargs)

# 加载数据库
db = Milvus(embedding_function=embedding, collection_name="arXiv",connection_args={"host": "172.29.4.47", "port": "19530"})

# 加载QWen大模型
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = ChatOpenAI(model="Qwen1.5-14B")

# 构建retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# 构建message_history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
with_message_history = RunnableWithMessageHistory(model, get_session_history)


# 构建prompt
def query_prompt(question: str, context: str) -> str:
    return f"""Answer this question in your best professional capacity. 
You can use the context provided, which is the abstract of an arXiv paper related to the question.

Question:
{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a professional expert in arXiv papers."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# 构建chain

rag_chain = prompt | model
config = {"configurable": {"session_id": "answer"}}

with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="messages",
)

def query_runner(question: str):
    original = retriever.invoke(question)
    context = original[0].page_content
    refer = original[0].metadata
    content = query_prompt(question, context)
    response = with_message_history.invoke(
        {"messages": [HumanMessage(content=content)]},
        config=config
    )
    return {"response": response.content, "refer": refer, "abstract": context}

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    question_path = "/Users/mac/Desktop/RAG_experiment/questions.json"
    questions = read_json(question_path)
    answers = []
    for data in tqdm(questions):
        question = data["question"]
        response = query_runner(question)
        answer = response["response"]
        refer = response["refer"]
        abstract = response["abstract"]
        ret_data = {"question": question, "answer": answer, "abstract": abstract, "refer": refer}
        answers.append(ret_data)
    write_json(answers, "/Users/mac/Desktop/RAG_experiment/answers.json")
    

    # question = "你好，我滴宝～"
    # response = query_runner(question)
    # print("Question:", question)
    # print("Response:", response["response"])
    # print("Refer", response["refer"])
    
    