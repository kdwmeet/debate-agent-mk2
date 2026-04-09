import streamlit as st
from langchain_core.messages import HumanMessage
from app.graph import app_graph

st.set_page_config(page_title="자율 토론 에이전트", layout="wide")

st.title("자율 토론 및 합의 도출 (Multi-Agent Debate)")
st.markdown("CEO(중재자)의 주도하에 원가, 품질, 마케팅 에이전트가 신제품 기획을 두고 치열하게 논쟁하며 최적의 타협점을 찾아냅니다.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("토론 안건 발제")
    with st.form(key="debate_form"):
        topic_input = st.text_input(
            "신제품 기획 주제를 입력하십시오.",
            placeholder="예: 30대 직장인을 타겟으로 한 스마트 텀블러 기획"
        )
        submit_btn = st.form_submit_button("토론 시작", use_container_width=True)

with col2:
    st.subheader("실시간 토론 회의실")

    if submit_btn and topic_input.strip():
        # 시스템에 전달할 초기 발제 메시지
        initial_message = HumanMessage(content=f"오늘 회의 안건은 '{topic_input}'입니다. 각 전문가들은 의견을 내주십시오.", name="CEO")
        
        initial_state = {
            "topic": topic_input,
            "messages": [initial_message],
            "turn_count": 0,
            "next_speaker": "",
            "final_summary": ""
        }

        final_state = {}

        with st.container(height=500, border=True):
            st.chat_message("CEO").write(initial_message.content)

            with st.spinner("전문가들이 의견을 조율하고 있습니다..."):
                for output in app_graph.stream(initial_state):
                    for node_name, state_update in output.items():

                        # 최종 상태 업데이트용
                        for k, v in state_update.items():
                            if k == "messages":
                                # 방금 추가된 최신 메시지 하나만 화면에 렌더링
                                new_msg = v[-1]
                                agent_name = new_msg.name

                                with st.chat_message(agent_name):
                                    st.markdown(f"**[{agent_name} 전문가]**")
                                    st.write(new_msg.content)
                            elif k == "final_summary":
                                final_state["final_summary"] = v
        
        # 토론 종료 후 최종 요약 출력
        if final_state.get("final_summary"):
            st.success("토론이 종료되어 합의안이 도출되었습니다.")
            with st.expander("CEO 최종 회의 요약록", expanded=True):
                st.markdown(final_state["final_summary"])
    
    elif not submit_btn:
        st.info("좌측에 안건을 입력하고 회의를 시작하십시오.")