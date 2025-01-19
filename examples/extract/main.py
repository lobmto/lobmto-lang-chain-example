from typing import List, Optional
from pydantic import BaseModel, Field
import getpass
import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAIのAPIキーを入力してください: ")

llm = ChatOpenAI(model="gpt-4o-mini")

class Person(BaseModel):
    """人物に関する情報。"""
    # ^ DocString はPersonスキーマの説明としてLLMに送信される

    # 判断不能だった場合を考慮してOptionalをつける
    # descriptionはLLMによって使用される
    name: Optional[str] = Field(default=None, description="人物の名前")
    hair_color: Optional[str] = Field(
        default=None, description="人物の髪の色(分かる場合のみ)"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="人物の身長(m)"
    )

class Data(BaseModel):
    """人物から抽出したデータ。"""

    people: List[Person]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは情報抽出の専門家です。"
            "テキストから関連情報のみを抽出してください。"
            "抽出を求められた属性の値が不明な場合は、その属性の値としてnullを返してください。",
        ),
        ("human", "{text}"),
    ]
)
structured_llm = llm.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
res = structured_llm.invoke(prompt)
print(res)
