from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Pydantic 스키마 정의
class ModeratorDecision(BaseModel):
    """중재자의 다음 행동 결정 및 요약"""
    next_speaker: Literal["cost_agent", "quality_agent", "marketing_agent", "FINISH"] = Field(
        description="다음에 발언할 에이전트를 선택하십시오. 합의가 이루어졌거나 토론이 충분히 진행되었다면 'FINISH'를 선택하십시오."
    )
    summary: str = Field(
        description="FINISH를 선택한 경우, 지금까지의 토론 내용을 바탕으로 최종 제품 기획안을 요약하십시오. FINISH가 아니라면 빈 문자열을 남기십시오."
    )

# 상태 정의
class DebateState(TypedDict):
    topic: str
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    next_speaker: str
    final_summary: str

# 노드 구현
MAX_TURNS = 6 #무한 논쟁 방지

def moderator_node(state: DebateState):
    """토론의 흐름을 제어하고, 발언자를 지정하거나 토론을 종료시킵니다."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    structured_llm = llm.with_structured_output(ModeratorDecision)

    topic = state.get("topic", "")
    messages = state.get("messages", [])
    turn_count = state.get("turn_count", 0)

    # 제한된 턴 수에 도달하면 강제로 결론을 도출하도록 프롬프트 지시
    turn_warning = ""
    if turn_count >= MAX_TURNS:
        turn_warning = "경고: 토론 제한 시간에 도달했습니다. 반드시 'FINISH'를 선택하고 최종 합의안을 강제로 도출하여 요약하십시오."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 신제품 기획 위원회의 CEO이자 중재자입니다.
현재 주제: {topic}
세 명의 전문가(Cost, Quality, Marketing)가 토론 중입니다. 대화 내역을 읽고 다음에 반박이나 의견을 내야 할 에이전트를 지정하십시오.
모두가 어느 정도 타협점을 찾았다고 판단되면 'FINISH'를 선언하고 최종 결론을 요약하십시오.
{warning}"""),
        MessagesPlaceholder(variable_name="messages")
    ])

    decision: ModeratorDecision = (prompt | structured_llm).invoke({
        "topic": topic,
        "messages": messages,
        "warning": turn_warning
    })

    return {"next_speaker": decision.next_speaker, "final_summary": decision.summary}

def cost_agent_node(state: DebateState):
    """원가 절감 전문가 페르소나"""
    llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")
    topic = state.get("topic", "")
    messages = state.get("messages", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"주제: {topic}\n당신은 '원가 절감 전문가(Cost)'입니다. 무조건 생산 비용을 줄이고 불필요한 프리미엄 기능을 빼서 이윤을 남기는 것을 목표로 1~2문장으로 강하게 주장하십시오. 다른 전문가의 의견에 반박하십시오."),
        MessagesPlaceholder(variable_name="messages")
    ])
    response = (prompt | llm).invoke({"messages": messages})

    # turn_count를 1 증가시키고 메시지 기록 (작성자 명시)
    return {
        "messages": [AIMessage(content=response.content, name="Cost")],
        "turn_count": state.get("turn_count", 0) + 1
    }

def quality_agent_node(state: DebateState):
    """품질 극대화 전문가 페르소나"""
    llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")
    topic = state.get("topic", "")
    messages = state.get("messages", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"주제: {topic}\n당신은 '품질 극대화 전문가(Quality)'입니다. 비용이 얼마가 들든 최고급 소재와 최신 기술을 넣어 최고의 성능을 내는 제품을 만들어야 한다고 1~2문장으로 강하게 주장하십시오. 다른 전문가의 의견에 반박하십시오."),
        MessagesPlaceholder(variable_name="messages")
    ])
    response = (prompt | llm).invoke({"messages": messages})

    return {
        "messages": [AIMessage(content=response.content, name="Quality")],
        "turn_count": state.get("turn_count", 0) + 1
    }

def marketing_agent_node(state: DebateState):
    """마케팅 전문가 페르소나"""
    llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")
    topic = state.get("topic", "")
    messages = state.get("messages", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"주제: {topic}\n당신은 '마케팅 전문가(Marketing)'입니다. 소비자의 눈길을 끄는 디자인과 트렌디한 감성, 그리고 빠른 시장 출시가 제일 중요하다고 1~2문장으로 강하게 주장하십시오. 원가나 품질 사이에서 타협안을 제시하며 중재를 시도해 보십시오."),
        MessagesPlaceholder(variable_name="messages")
    ])
    response = (prompt | llm).invoke({"messages": messages})

    return {
        "messages": [AIMessage(content=response.content, name="Marketing")],
        "turn_count": state.get("turn_count", 0) + 1
    }

# 라우팅 로직
def route_speaker(state: DebateState):
    return state.get("next_speaker", "FINISH")

# 그래프 조립
workflow = StateGraph(DebateState)

workflow.add_node("moderator_node", moderator_node)
workflow.add_node("cost_agent", cost_agent_node)
workflow.add_node("quality_agent", quality_agent_node)
workflow.add_node("marketing_agent", marketing_agent_node)

# 토론의 시작과 라우팅은 항상 중재자가 담당
workflow.add_edge(START, "moderator_node")

# 중재자의 결정에 따라 3명의 에이전트 또는 종료로 동적 분기
workflow.add_conditional_edges(
    "moderator_node",
    route_speaker,
    {
        "cost_agent": "cost_agent",
        "quality_agent": "quality_agent",
        "marketing_agent": "marketing_agent",
        "FINISH": END
    }
)

# 발언을 마친 에이전트들은 무조건 중재자에게 마이크를 넘김
workflow.add_edge("cost_agent", "moderator_node")
workflow.add_edge("quality_agent", "moderator_node")
workflow.add_edge("marketing_agent", "moderator_node")

app_graph = workflow.compile()