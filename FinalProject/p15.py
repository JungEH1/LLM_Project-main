import streamlit as st
import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# --- LangChain 초기화 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# 세션 상태 초기화
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = []

# --- Streamlit UI ---
st.title("📅 달력에 메모하기")

# 오늘 날짜
today = datetime.date.today()
date = st.date_input("날짜 선택", today)

# 메모 입력
memo = st.text_area("메모 입력")
if st.button("저장"):
    if memo.strip():
        doc = Document(page_content=memo, metadata={"date": str(date)})
        st.session_state.docs.append(doc)
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.docs, embeddings)
        st.success(f"{date} 메모 저장 완료!")

# 전체 메모 보기
st.subheader("📂 전체 메모")
if st.session_state.docs:
    for i, d in enumerate(st.session_state.docs):
        col1, col2 = st.columns([8, 1])
        with col1:
            st.markdown(f"**{d.metadata['date']}** : {d.page_content}")
        with col2:
            if st.button("🗑️", key=f"delete_{i}"):
                # 삭제: 해당 문서 제거 후 vectorstore 재생성
                st.session_state.docs.pop(i)
                if st.session_state.docs:
                    st.session_state.vectorstore = FAISS.from_documents(st.session_state.docs, embeddings)
                else:
                    st.session_state.vectorstore = None
                st.experimental_rerun()
else:
    st.info("아직 저장된 메모가 없습니다.")

# AI에게 질문
st.subheader("❓ AI에게 질문하기")
question = st.text_input("질문을 입력하세요")
if st.button("질문하기"):
    if question.strip():
        if st.session_state.vectorstore is None:
            st.warning("먼저 메모를 저장해주세요.")
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(question)
            st.write("🧠 AI 답변:", answer)
