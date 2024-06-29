# Import relevant functionality
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from CourseTools import QueryCourse, SelectCourse, DeleteCourse

# 初始化存储机制、模型和其他工具
memory = SqliteSaver.from_conn_string(":memory:")
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = ChatOpenAI(model="Qwen1.5-14B")
agent_executor = create_react_agent(model, tools=[], checkpointer=memory)


# Create the agent
memory = SqliteSaver.from_conn_string(":memory:")
tools = [QueryCourse(), SelectCourse(), DeleteCourse()]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="请问有哪些课程我可以选择？ 必修课程有哪些？ 选修课程有哪些？如果你认为无法提供具体的帮助，可以考虑尝试使用工具。")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="请你帮我选择代码为MA101的课程")]}, config
):
    print(chunk)
    print("----")