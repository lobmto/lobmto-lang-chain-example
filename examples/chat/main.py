import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

# プロンプトの手動定義例
# from langchain_core.messages import HumanMessage, SystemMessage
# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]
# res = model.invoke(prompt)

from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
# print(prompt)
# [SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}),
#  HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]

# ブロッキング
# res = model.invoke(prompt)

# ストリーミング
for token in model.stream(prompt):
    print(token.content, end="", flush=True)
