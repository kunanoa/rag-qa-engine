# ============================================
# app.py - RAG 문서 Q&A 엔진 Streamlit UI
# ============================================
# 목적: PDF 업로드 → 임베딩 → 질문/답변까지 전체 파이프라인을
#       웹 브라우저에서 사용할 수 있는 인터페이스로 제공한다
#
# 실행 방법: streamlit run app.py
#
# [변경] v2.0 - 문서 CRUD + TTL 기능 추가
#   - 문서별 해시 기반 중복 방지 및 개별 삭제
#   - 유효기간(TTL) 설정 및 만료 문서 자동 삭제
#   - 탭 방식 UI (💬 채팅 / 📁 문서 관리)
# ============================================

import streamlit as st  # 웹 UI 프레임워크
import os  # 파일/폴더 관련
import tempfile  # 업로드된 PDF를 임시 저장할 폴더 생성
import shutil  # 폴더 복사/삭제
import pandas as pd  # [신규] 문서 관리 테이블 표시용

# 우리가 만든 모듈들 import
from modules.parser import parse_single_pdf, parse_single_file, SUPPORTED_EXTENSIONS  # [변경] 멀티 포맷 파싱
from modules.chunker import chunk_all_documents  # 텍스트 → 청크 분할
# [변경] 새 CRUD 함수들 import 추가
from modules.embedder import (
    get_embedding_function,
    store_chunks_to_chromadb,       # 기존 함수 (테스트 스크립트 호환용, 여기서는 미사용)
    EMBEDDING_PROVIDERS,
    TTL_OPTIONS,                    # [신규] 유효기간 선택지
    compute_file_hash,              # [신규] 파일 해시 계산
    add_document_to_chromadb,       # [신규] 문서 단위 추가
    delete_document_from_chromadb,  # [신규] 문서 단위 삭제
    get_stored_documents,           # [신규] 저장된 문서 목록 조회
    cleanup_expired_documents,      # [신규] 만료 문서 자동 삭제
    reset_all_documents,            # [신규] 전체 초기화
)
from modules.retriever import load_collection, search_similar_chunks, format_search_results  # 유사도 검색
from modules.generator import generate_answer, parse_llm_response, LLM_PROVIDERS  # LLM 답변 생성


# ============================================
# 페이지 기본 설정
# ============================================
st.set_page_config(
    page_title="RAG 문서 Q&A 엔진",  # 브라우저 탭 제목
    page_icon="📄",  # 탭 아이콘
    layout="wide",  # 넓은 레이아웃 (사이드바 + 메인 영역)
)

# ============================================
# 사이드바 너비 조절
# ============================================
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 380px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 세션 상태 초기화
# ============================================
# Streamlit은 매 상호작용마다 스크립트 전체를 다시 실행함
# session_state에 저장해야 데이터가 유지됨

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 채팅 기록 [{role, content}, ...]

if "collection" not in st.session_state:
    st.session_state.collection = None  # ChromaDB 컬렉션 객체

if "embedding_fn" not in st.session_state:
    st.session_state.embedding_fn = None  # 임베딩 함수 객체

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []  # 업로드된 파일명 목록

if "is_embedded" not in st.session_state:
    st.session_state.is_embedded = False  # 임베딩 완료 여부

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False  # 임베딩/로드 처리 중 여부 (UI 잠금용)

if "last_status" not in st.session_state:
    st.session_state.last_status = None  # 마지막 처리 결과 메시지 {"type": "success"/"error", "msg": "..."}

if "current_llm" not in st.session_state:
    st.session_state.current_llm = "ollama"  # 검색모델 기본값: Ollama (gemma4:31b)

if "current_top_k" not in st.session_state:
    st.session_state.current_top_k = 10

if "current_threshold" not in st.session_state:
    st.session_state.current_threshold = 0.5

if "current_temperature" not in st.session_state:
    st.session_state.current_temperature = 0.3

if "current_answer_mode" not in st.session_state:
    st.session_state.current_answer_mode = "strict"

if "current_chunk_size" not in st.session_state:
    st.session_state.current_chunk_size = 1024

if "current_chunk_overlap" not in st.session_state:
    st.session_state.current_chunk_overlap = 50

# [신규] 문서 관리 탭에서 삭제/초기화 결과 메시지를 유지하기 위한 세션 상태
if "doc_mgmt_status" not in st.session_state:
    st.session_state.doc_mgmt_status = None  # {"type": "success"/"error"/"warning", "msg": "..."}

# ============================================
# 앱 시작 시 기존 ChromaDB 자동 로드 + 만료 문서 정리
# ============================================
# 이미 저장된 임베딩이 있으면 자동으로 불러옴
# (디스크에서 인덱스만 읽는 거라 1~2초, API 호출 없음)
if not st.session_state.is_embedded:
    try:
        # [신규] 만료된 문서 자동 삭제 (앱 시작 시 한 번만 실행)
        expired_files = cleanup_expired_documents()
        if expired_files:
            st.session_state.doc_mgmt_status = {
                "type": "warning",
                "msg": f"🕐 만료된 문서 {len(expired_files)}개 자동 삭제: {', '.join(expired_files)}"
            }

        auto_collection = load_collection()  # 기존 컬렉션 불러오기 시도
        if auto_collection is not None:
            # 저장된 임베딩 모델 정보 읽기 (없으면 "google"을 기본값으로)
            saved_provider = auto_collection.metadata.get("embedding_provider", "google")
            auto_embedding_fn = get_embedding_function(provider=saved_provider)  # 저장된 모델로 로드
            st.session_state.collection = auto_collection
            st.session_state.embedding_fn = auto_embedding_fn
            st.session_state.is_embedded = True
            st.session_state.embedded_provider = saved_provider
    except Exception:
        pass  # 실패하면 무시 (수동으로 불러오면 됨)


# ============================================
# 사이드바 구성
# ============================================
with st.sidebar:
    st.title("⚙️ 설정")

    # ------------------------------------------
    # 상단: 문서 처리 영역 (변경 시 재임베딩 필요)
    # ------------------------------------------
    st.header("📁 문서 처리")
    st.caption("이 설정을 변경하면 [임베딩 실행]을 다시 눌러야 합니다.")

    # 처리 중이면 모든 입력 위젯을 비활성화
    _disabled = st.session_state.is_processing

    # [변경] 멀티 포맷 업로드 (PDF, DOCX, PPTX, TXT, XLSX)
    uploaded_files = st.file_uploader(
        "문서 파일 업로드",
        type=["pdf", "docx", "pptx", "txt", "xlsx"],  # [변경] XLSX도 추가
        accept_multiple_files=True,  # 여러 파일 동시 업로드 가능
        help="PDF, DOCX, PPTX, TXT, XLSX 파일을 드래그하거나 선택하세요 (여러 개 가능)",
        disabled=_disabled,  # 처리 중 비활성화
    )

    # 현재 임베딩 모델 표시
    if hasattr(st.session_state, "embedded_provider"):
        current_emb_name = EMBEDDING_PROVIDERS.get(
            st.session_state.embedded_provider, {}
        ).get("name", st.session_state.embedded_provider)
        st.caption(
            f"🔗 현재: {current_emb_name} / "
            f"청크={st.session_state.current_chunk_size} / "
            f"겹침={st.session_state.current_chunk_overlap}"
        )

    # 문서 처리 설정을 form으로 감싸서 조작 시 즉시 재실행 방지
    with st.form("doc_settings_form"):
        # 임베딩 모델 선택
        embedding_options = {key: val["name"] for key, val in EMBEDDING_PROVIDERS.items()}

        # 저장된 임베딩 모델이 있으면 그걸 기본값으로 설정
        embedding_keys = list(embedding_options.keys())
        default_embedding_index = 0  # 기본값: 첫 번째 (google)
        if hasattr(st.session_state, "embedded_provider"):
            if st.session_state.embedded_provider in embedding_keys:
                default_embedding_index = embedding_keys.index(st.session_state.embedded_provider)

        selected_embedding = st.selectbox(
            "임베딩 모델",
            options=list(embedding_options.keys()),
            format_func=lambda x: embedding_options[x],
            index=default_embedding_index,
            help="문서를 벡터로 변환하는 모델. 변경 시 재임베딩 필요.",
            disabled=_disabled,
        )

        # 청크 크기 선택
        chunk_size = st.select_slider(
            "청크 크기 (chunk_size)",
            options=[128, 256, 512, 768, 1024],
            value=1024,  # 기본값: 실험 결과 최적
            help="한 청크의 최대 문자 수. 작을수록 검색 정확도↑ 문맥↓, 클수록 반대",
            disabled=_disabled,
        )

        # 청크 겹침 선택
        chunk_overlap = st.select_slider(
            "청크 겹침 (chunk_overlap)",
            options=[0, 25, 50, 75, 100],
            value=50,  # 기본값
            help="청크 간 겹치는 문자 수. 겹침이 있어야 문맥 단절 방지",
            disabled=_disabled,
        )

        # [신규] 유효기간(TTL) 선택
        # TTL_OPTIONS: {"7d": {"label": "7일", "days": 7}, ...}
        ttl_options = {key: val["label"] for key, val in TTL_OPTIONS.items()}
        selected_ttl = st.selectbox(
            "문서 유효기간",
            options=list(ttl_options.keys()),  # ["7d", "30d", "90d", "365d", "none"]
            format_func=lambda x: ttl_options[x],  # 키 대신 "7일", "30일" 등 표시
            index=4,  # 기본값: "none" (무제한) — 리스트의 5번째(인덱스 4)
            help="이 기간이 지나면 문서가 자동 삭제됩니다. 구버전 데이터 혼선 방지용.",
            disabled=_disabled,
        )

        # 임베딩 실행 버튼 (form의 submit 버튼 역할)
        embed_submitted = st.form_submit_button(
            "🚀 임베딩 실행", use_container_width=True, type="primary", disabled=_disabled
        )

    # 임베딩 실행 처리
    if embed_submitted:
        if not uploaded_files:
            st.error("문서 파일을 먼저 업로드하세요!")
        else:
            # rerun 후에도 파일 데이터가 유지되도록 세션에 저장
            st.session_state.pending_action = "embed"
            st.session_state.pending_files = [
                {"name": f.name, "data": bytes(f.getbuffer())} for f in uploaded_files
                # [변경] bytes()로 감싸기 — getbuffer()는 memoryview인데,
                # 나중에 compute_file_hash()에서 bytes가 필요하기 때문
            ]
            st.session_state.pending_embedding = selected_embedding
            st.session_state.pending_chunk_size = chunk_size
            st.session_state.pending_chunk_overlap = chunk_overlap
            st.session_state.pending_ttl = selected_ttl  # [신규] TTL도 세션에 저장
            st.session_state.is_processing = True
            st.rerun()

    # 기존 임베딩 불러오기 버튼 (form 밖 - 즉시 실행 필요)
    if st.button("📂 기존 임베딩 불러오기", use_container_width=True, disabled=_disabled):
        st.session_state.pending_action = "load"
        st.session_state.pending_embedding = selected_embedding
        st.session_state.is_processing = True
        st.rerun()

    # ------------------------------------------
    # 실제 처리 로직 (is_processing이 True일 때 실행)
    # ------------------------------------------
    if st.session_state.is_processing:
        st.warning("⏳ 처리 중입니다. 완료될 때까지 잠시 기다려주세요.")
        action = st.session_state.get("pending_action", "load")

        if action == "embed" and "pending_files" in st.session_state:
            with st.spinner("임베딩 처리 중... (시간이 걸릴 수 있습니다)"):
                try:
                    # 세션에 저장된 파일 데이터와 설정값 가져오기
                    pending_files = st.session_state.pending_files
                    emb_provider = st.session_state.pending_embedding
                    c_size = st.session_state.pending_chunk_size
                    c_overlap = st.session_state.pending_chunk_overlap
                    ttl = st.session_state.get("pending_ttl", "none")  # [신규]

                    # 임베딩 함수 로드
                    embedding_fn = get_embedding_function(provider=emb_provider)

                    # [변경] 파일별로 개별 처리 (기존: 전체 삭제 후 일괄 저장)
                    # 이제 각 파일마다:
                    #   1. 해시 계산 → 중복 체크
                    #   2. 파싱 → 청킹 → 임베딩
                    #   3. add_document_to_chromadb()로 개별 저장
                    results = []  # 각 파일의 처리 결과를 저장할 리스트
                    total_chunks = 0  # 새로 추가된 총 청크 수

                    for file_info in pending_files:
                        file_data = file_info["data"]
                        filename = file_info["name"]

                        # 1. 파일 해시 계산 (문서 고유 ID)
                        doc_hash = compute_file_hash(file_data)

                        # 2. 임시 파일에 저장 후 PDF 파싱
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, filename)
                        with open(temp_path, "wb") as f:
                            f.write(file_data)

                        documents = parse_single_file(temp_path)  # [변경] 멀티 포맷 대응
                        shutil.rmtree(temp_dir)  # 임시 폴더 즉시 삭제

                        if not documents:
                            results.append({
                                "status": "error",
                                "message": f"⚠️ '{filename}' — 파싱 결과 없음 (건너뜀)",
                                "num_chunks": 0,
                            })
                            continue

                        # 3. 청킹
                        chunks = chunk_all_documents(
                            documents, chunk_size=c_size, chunk_overlap=c_overlap
                        )

                        # 4. 문서 단위 추가 (중복 체크 + TTL 포함)
                        add_result = add_document_to_chromadb(
                            chunks=chunks,
                            doc_hash=doc_hash,
                            file_info={
                                "filename": filename,
                                "num_pages": len(documents),
                            },
                            embedding_function=embedding_fn,
                            provider=emb_provider,
                            ttl=ttl,
                        )

                        results.append(add_result)  # [변경] 메시지가 아닌 결과 전체를 저장
                        total_chunks += add_result["num_chunks"]

                    # 컬렉션 다시 로드 (새로 추가된 데이터 반영)
                    collection = load_collection()

                    # 세션 상태 업데이트
                    st.session_state.collection = collection
                    st.session_state.embedding_fn = embedding_fn
                    st.session_state.uploaded_files_list = [f["name"] for f in pending_files]
                    st.session_state.is_embedded = True
                    st.session_state.embedded_provider = emb_provider
                    st.session_state.current_chunk_size = c_size
                    st.session_state.current_chunk_overlap = c_overlap

                    # [변경] 결과 메시지 타입을 파일별 성공/실패에 따라 결정
                    # status별 분류: "added" = 성공, "skipped" = 중복, "error" = 실패
                    result_messages = [r["message"] for r in results]
                    result_msg = "\n".join(result_messages)

                    has_error = any(r["status"] == "error" for r in results)
                    has_added = any(r["status"] == "added" for r in results)

                    if has_error and has_added:
                        # 일부 성공, 일부 실패
                        status_type = "warning"
                        header = "⚠️ 일부 파일 임베딩 실패"
                    elif has_error and not has_added:
                        # 전부 실패
                        status_type = "error"
                        header = "❌ 임베딩 실패"
                    else:
                        # 전부 성공 (added 또는 skipped)
                        status_type = "success"
                        header = "✅ 임베딩 처리 완료"

                    st.session_state.last_status = {
                        "type": status_type,
                        "msg": f"{header}\n\n{result_msg}"
                    }

                except Exception as e:
                    st.session_state.last_status = {"type": "error", "msg": f"❌ 임베딩 실패: {e}"}
                finally:
                    # 임시 데이터 정리
                    st.session_state.pop("pending_files", None)
                    st.session_state.pop("pending_action", None)
                    st.session_state.pop("pending_ttl", None)
                    st.session_state.is_processing = False
                    st.rerun()
        else:
            # 기존 임베딩 불러오기
            with st.spinner("기존 임베딩 불러오는 중..."):
                try:
                    emb_provider = st.session_state.get("pending_embedding", "google")
                    collection = load_collection()
                    if collection:
                        saved_provider = collection.metadata.get("embedding_provider", emb_provider)
                        embedding_fn = get_embedding_function(provider=saved_provider)
                        st.session_state.collection = collection
                        st.session_state.embedding_fn = embedding_fn
                        st.session_state.is_embedded = True
                        st.session_state.embedded_provider = saved_provider
                        st.session_state.last_status = {
                            "type": "success",
                            "msg": f"✅ 기존 데이터 로드 완료! ({collection.count()}개 청크)"
                        }
                    else:
                        st.session_state.last_status = {
                            "type": "error",
                            "msg": "저장된 임베딩이 없습니다. 먼저 임베딩을 실행하세요."
                        }
                except Exception as e:
                    st.session_state.last_status = {"type": "error", "msg": f"❌ 로드 실패: {e}"}
                finally:
                    st.session_state.pop("pending_action", None)
                    st.session_state.is_processing = False
                    st.rerun()

    # 처리 결과 메시지 표시 (rerun 후에도 유지됨)
    if st.session_state.last_status:
        if st.session_state.last_status["type"] == "success":
            st.success(st.session_state.last_status["msg"])
        elif st.session_state.last_status["type"] == "warning":
            st.warning(st.session_state.last_status["msg"])
        elif st.session_state.last_status["type"] == "error":
            st.error(st.session_state.last_status["msg"])
        st.session_state.last_status = None  # 한 번 표시 후 초기화

    # 임베딩 상태 표시
    if st.session_state.is_embedded:
        try:
            # [변경] 저장된 문서 수와 청크 수를 함께 표시
            doc_list = get_stored_documents()
            doc_count = len(doc_list)
            chunk_count = st.session_state.collection.count()
            st.info(f"💾 문서 {doc_count}개 / {chunk_count}개 청크 활성")
        except Exception:
            st.session_state.is_embedded = False
            st.session_state.collection = None
            st.rerun()

    st.divider()  # 구분선

    # ------------------------------------------
    # 하단: 검색/답변 영역 (자유롭게 변경 가능)
    # ------------------------------------------
    st.header("🔍 검색 / 답변")
    st.caption("이 설정은 자유롭게 바꿔도 됩니다. (재임베딩 불필요)")

    # 현재 적용된 설정 표시
    current_llm_name = LLM_PROVIDERS.get(
        st.session_state.current_llm, {}
    ).get("name", st.session_state.current_llm)
    mode_text = "엄격" if st.session_state.current_answer_mode == "strict" else "유연"
    st.caption(
        f"🤖 현재: {current_llm_name} / "
        f"Top-K={st.session_state.current_top_k} / "
        f"임계값={st.session_state.current_threshold} / "
        f"Temp={st.session_state.current_temperature} / "
        f"{mode_text}"
    )

    with st.form("search_settings_form"):
        # LLM 모델 선택
        llm_options = {key: val["name"] for key, val in LLM_PROVIDERS.items()}
        llm_keys = list(llm_options.keys())
        default_llm_index = llm_keys.index(st.session_state.current_llm) if st.session_state.current_llm in llm_keys else 0
        selected_llm = st.selectbox(
            "LLM 모델",
            options=llm_keys,
            format_func=lambda x: llm_options[x],
            index=default_llm_index,
            help="답변을 생성하는 AI 모델",
            disabled=_disabled,
        )

        # Top-K 선택
        top_k = st.select_slider(
            "Top-K (검색 청크 수)",
            options=[1, 3, 5, 10],
            value=10,
            help="질문과 유사한 청크를 몇 개 검색할지. 많을수록 정보↑ 노이즈↑",
            disabled=_disabled,
        )

        # 유사도 임계값
        threshold = st.slider(
            "유사도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="이 값 미만의 유사도를 가진 청크는 제외. 0.0이면 필터링 안 함",
            disabled=_disabled,
        )

        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="낮을수록 일관된 답변, 높을수록 창의적 답변",
            disabled=_disabled,
        )

        # 답변 범위 선택
        answer_mode = st.radio(
            "답변 범위",
            options=["strict", "flexible"],
            format_func=lambda x: "📄 문서만 (엄격)" if x == "strict" else "🌐 문서 우선 (유연)",
            index=0,
            help="엄격: 문서에 없으면 '찾을 수 없음'. 유연: 문서에 없으면 AI 지식으로 보충",
            disabled=_disabled,
        )

        search_submitted = st.form_submit_button("✅ 설정 적용", use_container_width=True, disabled=_disabled)

    if search_submitted:
        st.session_state.current_llm = selected_llm
        st.session_state.current_top_k = top_k
        st.session_state.current_threshold = threshold
        st.session_state.current_temperature = temperature
        st.session_state.current_answer_mode = answer_mode
        st.rerun()

    st.divider()

    # 채팅 기록 초기화 버튼
    if st.button("🗑️ 채팅 기록 초기화", use_container_width=True, disabled=_disabled):
        st.session_state.chat_history = []
        st.rerun()


# ============================================
# [변경] 메인 영역: 탭 방식 UI (채팅 / 문서 관리)
# ============================================
# 기존: 메인 영역이 채팅 인터페이스만 있었음
# 변경: st.tabs()로 채팅과 문서 관리를 탭으로 분리
st.title("📄 RAG 문서 Q&A 엔진")
st.caption("문서(PDF, DOCX, PPTX, TXT, XLSX)를 업로드하고 질문하면 AI가 문서 기반으로 답변합니다.")

tab_chat, tab_docs = st.tabs(["💬 채팅", "📁 문서 관리"])

# ============================================
# 탭 1: 채팅 인터페이스 (기존 코드와 동일)
# ============================================
with tab_chat:
    # 임베딩이 안 되어 있으면 안내 메시지
    if not st.session_state.is_embedded:
        st.info("👈 왼쪽 사이드바에서 문서를 업로드하고 [임베딩 실행]을 눌러주세요.")

    # 기존 채팅 기록 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # assistant 메시지에 출처 정보가 있으면 표시
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📚 참고 출처 ({len(message['sources'])}개 청크)"):
                    for i, result in enumerate(message["sources"]):
                        source = result["metadata"].get("source", "알 수 없음")
                        page = result["metadata"].get("page", "?")
                        similarity = result["similarity"]
                        st.markdown(
                            f"**[출처 {i+1}]** {source} / {page}페이지 "
                            f"(유사도: {similarity:.2f})"
                        )
                        st.caption(result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"])
                        st.divider()

# 사용자 입력 (채팅 입력창) — 탭 밖에 배치
# st.chat_input은 Streamlit 특성상 페이지 하단에 고정되므로 탭 안에 넣지 않음
prompt = st.chat_input("질문을 입력하세요...", disabled=_disabled)
if prompt:
    with tab_chat:
        # [변경] 길이 체크를 렌더링보다 먼저 수행
        # 긴 텍스트를 st.markdown으로 전부 렌더링하면 앱이 느려지므로
        # 2000자 초과 시 잘라서 표시
        if len(prompt) > 2000:
            # 긴 입력은 앞 200자만 보여주고 나머지는 생략
            display_prompt = prompt[:200] + f"\n\n... ({len(prompt)}자 중 200자만 표시)"
            with st.chat_message("user"):
                st.markdown(display_prompt)
            st.session_state.chat_history.append({"role": "user", "content": display_prompt})

            with st.chat_message("assistant"):
                too_long_msg = f"⚠️ 질문이 너무 깁니다. ({len(prompt)}자)\n\n2000자 이내로 줄여주세요."
                st.markdown(too_long_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": too_long_msg,
                "sources": [],
            })

        else:
            # 정상 길이 입력
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # 임베딩 모델 불일치 검사
            if (st.session_state.is_embedded
                    and hasattr(st.session_state, "embedded_provider")
                    and selected_embedding != st.session_state.embedded_provider):
                with st.chat_message("assistant"):
                    saved_name = EMBEDDING_PROVIDERS.get(
                        st.session_state.embedded_provider, {}
                    ).get("name", st.session_state.embedded_provider)
                    selected_name = EMBEDDING_PROVIDERS.get(
                        selected_embedding, {}
                    ).get("name", selected_embedding)
                    mismatch_msg = (
                        f"⚠️ 현재 선택된 임베딩 모델(**{selected_name}**)과 "
                        f"저장된 임베딩 모델(**{saved_name}**)이 다릅니다.\n\n"
                        f"임베딩 모델을 **{saved_name}**(으)로 되돌리거나, "
                        f"새로 [임베딩 실행]을 해주세요."
                    )
                    st.markdown(mismatch_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": mismatch_msg,
                    "sources": [],
                })

            # 임베딩이 안 되어 있으면 경고
            elif not st.session_state.is_embedded:
                with st.chat_message("assistant"):
                    st.markdown("⚠️ 먼저 문서를 업로드하고 임베딩을 실행해주세요.")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⚠️ 먼저 문서를 업로드하고 임베딩을 실행해주세요.",
                    "sources": [],
                })

            else:
                # AI 답변 생성
                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중..."):
                        try:
                            # 1단계: 유사 청크 검색
                            search_results = search_similar_chunks(
                                query=prompt,
                                embedding_function=st.session_state.embedding_fn,
                                collection=st.session_state.collection,
                                top_k=st.session_state.current_top_k,
                                threshold=st.session_state.current_threshold,
                            )

                            # 2단계: 검색 결과를 문맥 텍스트로 변환
                            context = format_search_results(search_results)

                            # 3단계: LLM 답변 생성
                            raw_response = generate_answer(
                                context=context,
                                query=prompt,
                                provider=st.session_state.current_llm,
                                temperature=st.session_state.current_temperature,
                                answer_mode=st.session_state.current_answer_mode,
                            )

                            # 4단계: LLM 응답 파싱
                            if raw_response.startswith("❌"):
                                answer = raw_response
                                found_in_document = False
                            else:
                                parsed = parse_llm_response(raw_response)
                                answer = parsed["answer"]
                                found_in_document = parsed["found_in_document"]

                            # 답변 표시
                            st.markdown(answer)

                            # 출처 표시 (접이식)
                            if search_results and found_in_document:
                                with st.expander(f"📚 참고 출처 ({len(search_results)}개 청크)"):
                                    for i, result in enumerate(search_results):
                                        source = result["metadata"].get("source", "알 수 없음")
                                        page = result["metadata"].get("page", "?")
                                        similarity = result["similarity"]
                                        st.markdown(
                                            f"**[출처 {i+1}]** {source} / {page}페이지 "
                                            f"(유사도: {similarity:.2f})"
                                        )
                                        st.caption(result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"])
                                        st.divider()
                            elif not search_results:
                                st.caption("⚠️ 유사도 임계값을 넘는 관련 문서를 찾지 못했습니다.")

                            # 채팅 기록에 추가
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": search_results if (search_results and found_in_document) else [],
                            })

                        except Exception as e:
                            error_msg = f"❌ 오류 발생: {e}"
                            st.markdown(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg,
                                "sources": [],
                            })


# ============================================
# [신규] 탭 2: 문서 관리
# ============================================
# 저장된 문서 목록을 테이블로 보여주고,
# 개별 삭제 / 전체 초기화 기능을 제공한다.
with tab_docs:
    st.subheader("📁 저장된 문서 목록")

    # 문서 관리 탭 상태 메시지 표시 (삭제/초기화 결과)
    if st.session_state.doc_mgmt_status:
        status = st.session_state.doc_mgmt_status
        if status["type"] == "success":
            st.success(status["msg"])
        elif status["type"] == "error":
            st.error(status["msg"])
        elif status["type"] == "warning":
            st.warning(status["msg"])
        st.session_state.doc_mgmt_status = None  # 한 번 표시 후 초기화

    # 저장된 문서 목록 조회
    doc_list = get_stored_documents()

    if not doc_list:
        # 문서가 없는 경우
        st.info("저장된 문서가 없습니다. 사이드바에서 문서를 업로드하고 임베딩을 실행하세요.")
    else:
        # ── 요약 정보 표시 ──
        total_chunks = sum(d["num_chunks"] for d in doc_list)
        expired_count = sum(1 for d in doc_list if d["is_expired"])

        # 만료 문서가 있으면 경고
        if expired_count > 0:
            st.warning(f"⚠️ 만료된 문서 {expired_count}개 — 페이지 새로고침 시 자동 삭제됩니다.")

        st.caption(f"총 {len(doc_list)}개 문서 / {total_chunks}개 청크")

        # ── 검색 필터 (파일명으로 필터링) ──
        search_query = st.text_input(
            "🔎 파일명 검색",
            placeholder="파일명을 입력하세요...",
            label_visibility="collapsed",  # 라벨 숨김 (placeholder로 대체)
        )

        # 검색어가 있으면 필터링 (대소문자 무시)
        if search_query:
            filtered_docs = [d for d in doc_list if search_query.lower() in d["filename"].lower()]
        else:
            filtered_docs = doc_list

        if not filtered_docs:
            st.info(f"'{search_query}'에 해당하는 문서가 없습니다.")
        else:
            # ── 페이지네이션 ──
            # 문서가 수백 개일 때 한 번에 다 보여주면 느려지므로
            # 페이지당 20개씩 나눠서 보여줌
            PAGE_SIZE = 20  # 한 페이지당 보여줄 문서 수
            total_pages = max(1, (len(filtered_docs) + PAGE_SIZE - 1) // PAGE_SIZE)
            # 올림 나눗셈: 문서 45개 → (45 + 19) // 20 = 3페이지

            # 페이지 번호 선택 (문서 21개 이상일 때만 표시)
            if total_pages > 1:
                current_page = st.number_input(
                    f"페이지 (1~{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                )
            else:
                current_page = 1

            # 현재 페이지에 해당하는 문서만 추출
            start_idx = (current_page - 1) * PAGE_SIZE
            end_idx = start_idx + PAGE_SIZE
            page_docs = filtered_docs[start_idx:end_idx]

            # ── 테이블 표시 (pandas DataFrame) ──
            table_data = []
            for doc in page_docs:
                # 만료일 표시 형식
                if doc["expires_at"] is None:
                    expires_display = "♾️ 무제한"
                elif doc["is_expired"]:
                    expires_display = f"❌ {doc['expires_at']} (만료)"
                else:
                    expires_display = f"✅ {doc['expires_at']}"

                table_data.append({
                    "파일명": doc["filename"],
                    "페이지": doc["num_pages"],
                    "청크": doc["num_chunks"],
                    "업로드일": doc["uploaded_at"],
                    "만료일": expires_display,
                    "임베딩": doc["embedding_provider"],
                })

            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,  # 전체 너비 사용
                hide_index=True,  # 인덱스(0, 1, 2...) 숨김
            )

            # ── 개별 삭제 ──
            st.subheader("🗑️ 문서 삭제")

            # 드롭다운 선택지: "파일명 (해시앞8자)"
            delete_options = {
                doc["doc_hash"]: f"{doc['filename']}  ({doc['doc_hash'][:8]}...)"
                for doc in filtered_docs
            }

            if delete_options:
                selected_delete = st.selectbox(
                    "삭제할 문서 선택",
                    options=list(delete_options.keys()),
                    format_func=lambda x: delete_options[x],
                )

                # 삭제 버튼 / 전체 초기화 버튼을 나란히 배치
                col_del, col_reset = st.columns(2)

                with col_del:
                    if st.button("🗑️ 선택 문서 삭제", use_container_width=True, type="secondary"):
                        result = delete_document_from_chromadb(selected_delete)

                        if result["status"] == "deleted":
                            st.session_state.doc_mgmt_status = {
                                "type": "success", "msg": result["message"]
                            }
                            # 컬렉션 다시 로드
                            new_collection = load_collection()
                            if new_collection:
                                st.session_state.collection = new_collection
                            else:
                                # 모든 문서가 삭제되어 컬렉션이 비었을 때
                                st.session_state.is_embedded = False
                                st.session_state.collection = None
                        else:
                            st.session_state.doc_mgmt_status = {
                                "type": "error", "msg": result["message"]
                            }

                        st.rerun()  # 테이블 갱신

                with col_reset:
                    if st.button("⚠️ 전체 초기화", use_container_width=True, type="primary"):
                        # 전체 삭제 확인을 위해 세션에 플래그 설정
                        st.session_state.confirm_reset = True
                        st.rerun()

            # ── 전체 초기화 확인 대화상자 ──
            # "전체 초기화" 버튼을 누르면 confirm_reset 플래그가 켜지고,
            # 확인/취소 버튼이 추가로 표시됨 (실수로 삭제 방지)
            if st.session_state.get("confirm_reset", False):
                st.error("⚠️ 모든 문서와 임베딩 데이터가 삭제됩니다. 정말 초기화하시겠습니까?")
                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button("✅ 예, 초기화", use_container_width=True, type="primary"):
                        result = reset_all_documents()
                        st.session_state.doc_mgmt_status = {
                            "type": "success" if result["status"] == "reset" else "error",
                            "msg": result["message"],
                        }
                        st.session_state.is_embedded = False
                        st.session_state.collection = None
                        st.session_state.embedding_fn = None
                        st.session_state.confirm_reset = False
                        st.rerun()

                with col_no:
                    if st.button("❌ 취소", use_container_width=True):
                        st.session_state.confirm_reset = False
                        st.rerun()


# ============================================
# 푸터
# ============================================
st.sidebar.markdown("---")
st.sidebar.caption("RAG 문서 Q&A 엔진 v2.0")
st.sidebar.caption("개발: 광주인력개발원 프로젝트")