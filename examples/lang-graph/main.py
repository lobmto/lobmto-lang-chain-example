import getpass
import os
from typing import Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAIのAPIキーを入力してください: ")

llm = ChatOpenAI(model="gpt-4o-mini")

def stream_message(model: ChatOpenAI, messages: Sequence[BaseMessage]) -> str:
    content = ""
    for token in model.stream(messages):
        content += token.content
        print(token.content, end="", flush=True)
    print()
    return content

class Route(BaseModel):
    step: Literal["idiom", "haiku"] = Field(
        None, description="次に生成する対象"
    )

router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    first_output: str
    second_output: str
    third_output: str
    combined_output: str
    tool_calls: list[dict]
    tool_results: list[ToolMessage]
    output: str

# Nodes
def llm_call_1(state: State):
    """熟語を生成する"""

    print("\n[熟語の生成処理]")
    return {
        "first_output": stream_message(llm,
            [
                SystemMessage(content="架空の熟語を1つ生成してください。出力には読み仮名と意味を含めてください。"),
                HumanMessage(content=state['input'])
            ]
        )
    }

def llm_call_2(state: State):
    """俳句を生成する"""

    print("\n[俳句の生成処理]")
    return {
        "first_output": stream_message(llm,
            [
                SystemMessage(content="架空の俳句を1つ生成してください。出力には意味を含めてください。"),
                HumanMessage(content=state['input'])
            ]
        )
    }

def llm_call_3(state: State):
    return state

def llm_call_router(state: State):
    """入力を適切なノードにルーティング"""

    decision = router.invoke(
        [
            SystemMessage(
                content="ユーザの入力をもとに、熟語か俳句のどちらかへルーティングしてください"
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    print(f"\n[次のノード]\n{decision.step}")
    return {"decision": decision.step}

def generate_second_output(state: State):
    """架空の{state['decision']}から、より高品質かつ架空の別パターンを出力する"""

    msg = llm.invoke(f"以下の架空の{state['decision']}について、より高品質かつ架空の別パターンを出力してください。 {state['first_output']}")
    print(f"\n[2つ目の出力]\n{msg.content}")
    return {"second_output": msg.content}

def generate_third_output(state: State):
    """架空の{state['decision']}から、より高品質かつ架空の別パターンを出力する"""

    msg = llm.invoke(f"以下の架空の{state['decision']}について、よりユーモアのある架空の別パターンを出力してください。 {state['first_output']}")
    print(f"\n[3つ目の出力]\n{msg.content}")
    return {"third_output": msg.content}

def aggregator(state: State):
    """3パターンの出力を統合する"""

    combined = ""
    combined += f"1.\n{state['first_output']}\n\n"
    combined += f"2.\n{state['second_output']}\n\n"
    combined += f"3.\n{state['third_output']}"
    return {"combined_output": combined}
    
def route_decision(state: State):
    if state["decision"] == "idiom":
        return "llm_call_1"
    elif state["decision"] == "haiku":
        return "llm_call_2"

@tool
def sum_numbers(nums: list[int]) -> int:
    """numsに含まれる数字の合計を返す

    Args:
        nums: list of int
    """
    return sum(nums)

tools = [sum_numbers]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["tool_calls"]:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"tool_results": result}

def llm_call_4(state: State):
    """それぞれに含まれる数字の合計を計算する"""

    tool_calls = llm_with_tools.invoke(
        [
            SystemMessage(content="3つの作品中に含まれる数字を列挙して合計してください。説明・解説文は除外します。ツールの呼び出しは1度で行ってください。"),
            HumanMessage(content=state['combined_output'])
        ]
    ).tool_calls
    print(f"\n[合計の計算]\n{tool_calls}")
    return {
        "tool_calls": tool_calls
    }

# Build workflow
workflow_router = StateGraph(State)

# Add nodes
workflow_router.add_node("llm_call_1", llm_call_1)
workflow_router.add_node("llm_call_2", llm_call_2)
workflow_router.add_node("llm_call_router", llm_call_router)
workflow_router.add_node("llm_call_3", llm_call_3)
workflow_router.add_node("generate_second_output", generate_second_output)
workflow_router.add_node("generate_third_output", generate_third_output)
workflow_router.add_node("aggregator", aggregator)
workflow_router.add_node("llm_call_4", llm_call_4)
workflow_router.add_node("tool_node", tool_node)

# Add edges to connect nodes
workflow_router.add_edge(START, "llm_call_router")
workflow_router.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
    },
)
workflow_router.add_edge("llm_call_1", "llm_call_3")
workflow_router.add_edge("llm_call_2", "llm_call_3")
workflow_router.add_edge("llm_call_3", "generate_second_output")
workflow_router.add_edge("llm_call_3", "generate_third_output")
workflow_router.add_edge("generate_second_output", "aggregator")
workflow_router.add_edge("generate_third_output", "aggregator")
workflow_router.add_edge("aggregator", "llm_call_4")
workflow_router.add_edge("llm_call_4", "tool_node")
workflow_router.add_edge("tool_node", END)

# Compile workflow
workflow = workflow_router.compile()

state = workflow.invoke({"input": "数字を1つだけ含む漢字の熟語を生成してください"})
print(f"\n[出力]\n{state['combined_output']}")
print(f"\n[数字の合計 (無理やり付け足したので不正確)]\n{state['tool_results']}")
