import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
import os

def setup_streamlit_page():
    st.set_page_config(page_title="LlamaIndex Chat", page_icon="🦙")
    st.title("문서 기반 RAG 채팅")

def setup_openai_api():
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    return openai_api_key

def initialize_llm_and_settings():
    llm = OpenAI(
        temperature=0.5,
        model="gpt-4o",
        max_tokens=512,
        context_window=4096,
    )
    Settings.llm = llm
    return llm

def process_uploaded_files(uploaded_files):
    """업로드된 파일들을 처리하여 문서 리스트 생성"""
    documents = []
    pdf_reader = PDFReader()
    
    for file in uploaded_files:
        try:
            # 파일 내용을 바이트로 읽기
            content = file.read()
            
            # 텍스트 파일인 경우
            if file.type == "text/plain":
                text_content = content.decode('utf-8')
                doc = Document(text=text_content, metadata={"filename": file.name})
                documents.append(doc)
            # PDF 파일인 경우    
            elif file.type == "application/pdf":
                # 임시 파일로 저장
                with open(f"temp_{file.name}", "wb") as f:
                    f.write(content)
                # PDF 파일 읽기
                pdf_docs = pdf_reader.load_data(f"temp_{file.name}")
                documents.extend(pdf_docs)
                # 임시 파일 삭제
                os.remove(f"temp_{file.name}")
            else:
                st.error(f"지원하지 않는 파일 형식입니다: {file.name}")
                
        except Exception as e:
            st.error(f"파일 처리 중 오류 발생: {file.name} - {str(e)}")
            
    return documents

def initialize_chat_engine(index):
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        verbose=True,
        system_prompt="""
        당신은 업로드된 문서를 기반으로 답변하는 AI 어시스턴트입니다.
        이전 대화 내용을 고려하여 답변하되, 필요한 경우에만 문서를 검색하여 답변해주세요.
        업로드된 문서에서 찾을 수 없는 내용은 솔직히 모른다고 말씀해주세요.
        """
    )
    return chat_engine

def main():
    setup_streamlit_page()
    openai_api_key = setup_openai_api()

    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # 파일 업로드 섹션 
    st.sidebar.header("문서 업로드")
    uploaded_files = st.sidebar.file_uploader(
        "텍스트 파일이나 PDF 파일을 업로드하세요",
        accept_multiple_files=True,
        type=["txt", "pdf"]
    )

    # 업로드된 파일 처리 및 인덱스 생성
    if uploaded_files:
        with st.spinner("문서를 처리하고 있습니다..."):
            initialize_llm_and_settings()
            documents = process_uploaded_files(uploaded_files)
            
            if documents:
                # 문서 정보 표시
                st.sidebar.success(f"처리된 문서 수: {len(documents)}")
                
                # 인덱스 생성
                index = VectorStoreIndex.from_documents(documents)
                st.session_state.chat_engine = initialize_chat_engine(index)
                st.success("문서 처리가 완료되었습니다!")
            else:
                st.error("처리할 수 있는 문서가 없습니다.")
                return

    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 엔진이 준비된 경우에만 입력 활성화
    if "chat_engine" in st.session_state:
        # 사용자 입력
        if prompt := st.chat_input("무엇이 궁금하신가요?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(response.response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.response}
                    )

            # 디버그 정보
            with st.expander("Debug Info"):
                st.write("Response Type:", type(response))
                st.write("Source Nodes:", response.source_nodes if hasattr(response, 'source_nodes') else None)
    else:
        st.info("문서를 업로드하면 질문할 수 있습니다.")

    # 채팅 초기화 버튼
    if st.sidebar.button("채팅 초기화"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()