# ============================================
# embedder.py - 텍스트를 벡터로 변환하여 ChromaDB에 저장하는 모듈
# ============================================
# 목적: 청크 텍스트를 임베딩 모델로 벡터(숫자 리스트)로 변환하고,
#       ChromaDB에 저장하여 나중에 유사도 검색이 가능하게 만든다
#
# 임베딩이란?
#   "머신비전 품질검사" → [0.12, -0.34, 0.56, ...] (수백 차원의 숫자 리스트)
#   의미가 비슷한 텍스트는 비슷한 벡터값을 가짐
#   → 이걸 이용해서 "질문과 가장 비슷한 청크"를 찾을 수 있음
# ============================================

import os  # 파일/폴더 경로 관련 기능
import time  # 요청 간 딜레이용 (API 속도 제한 대응)
import hashlib  # 파일 해시 계산 (문서 고유 ID 생성용)
import json  # 문서 레지스트리 파일 읽기/쓰기
from datetime import datetime, timedelta  # 업로드 시각, 유효기간 계산
import requests  # HTTP 요청 라이브러리 (Google API 직접 호출용)
import chromadb  # 벡터 데이터베이스 (임베딩 저장 및 유사도 검색)
from dotenv import load_dotenv  # .env 파일에서 API 키 불러오기

# --- 임베딩 모델 선택지 ---
# 사용자가 선택한 모델에 따라 필요한 것만 import 됨
#
# [API 방식] - 우리 서버에서 텍스트만 보내고 벡터를 받아옴
#   Google: gemini-embedding-001 (무료 티어 있음)
#   Google: gemini-embedding-2-preview (무료 티어 있음, Embedding 1보다 성능 향상)
#   OpenAI: text-embedding-3-small (유료, 성능 좋음)
#
# [로컬 방식] - 우리 서버(또는 GPU 머신)에서 직접 모델을 돌림
#
# [Ollama 방식] - Ollama가 모델 관리/실행을 대신해줌
#   qwen3-embedding:8b: 한국어 우수, Ollama로 간편 실행

# ============================================
# Google 임베딩을 REST API로 직접 호출하는 커스텀 클래스
# ============================================
# 왜 직접 만들었나?
#   langchain-google-genai SDK에 한국어 텍스트 인코딩 버그가 있어서
#   requests 라이브러리로 Google API를 직접 호출하여 우회한다
# ============================================
class GoogleEmbeddingDirect:
    """
    Google Embedding API를 requests로 직접 호출하는 클래스

    LangChain의 임베딩 인터페이스와 호환되도록
    embed_documents()와 embed_query() 메서드를 구현했다
    """

    def __init__(self, api_key, model="gemini-embedding-001"):
        """
        파라미터:
            api_key (str): Google API 키
            model (str): 임베딩 모델명 (기본값: gemini-embedding-001)
        """
        self.api_key = api_key  # API 키 저장
        self.model = model  # 모델명 저장
        # Google 임베딩 API의 엔드포인트 URL
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
        

    def _embed_single(self, text):
        """
        텍스트 하나를 벡터로 변환하는 내부 함수
        429 에러(속도 제한) 발생 시 자동으로 대기 후 재시도한다

        파라미터:
            text (str): 임베딩할 텍스트

        반환값:
            list[float]: 임베딩 벡터 (768차원)
        """
        # API 요청 본문 구성
        payload = {
            "model": f"models/{self.model}",  # 사용할 모델
            "content": {
                "parts": [{"text": text}]  # 임베딩할 텍스트
            }
        }

        max_retries = 5  # 최대 재시도 횟수

        for attempt in range(max_retries):  # 최대 5번 시도
            # API 호출 (requests 라이브러리 사용 → 한국어 인코딩 문제 없음)
            response = requests.post(
                f"{self.url}?key={self.api_key}",  # URL에 API 키 포함
                json=payload,  # JSON 형태로 전송
            )

            if response.status_code == 200:  # 성공 시
                result = response.json()
                return result["embedding"]["values"]  # 벡터 반환

            elif response.status_code == 429:  # 속도 제한에 걸렸을 때
                wait_time = 15  # 15초 대기 (무료 티어: 분당 100회 제한)
                print(f"      ⏳ API 속도 제한 → {wait_time}초 대기 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)  # 대기

            else:  # 다른 에러
                raise Exception(f"Google API 에러: {response.status_code} - {response.text}")

        raise Exception("Google API 재시도 횟수 초과")  # 5번 다 실패하면 에러

    def embed_documents(self, texts):
        """
        여러 텍스트를 벡터로 변환하는 함수 (LangChain 호환 인터페이스)
        Google 무료 티어 속도 제한(분당 100회)에 맞춰 요청 간 딜레이를 준다

        파라미터:
            texts (list[str]): 임베딩할 텍스트 리스트

        반환값:
            list[list[float]]: 임베딩 벡터 리스트
        """
        embeddings = []  # 결과 벡터를 저장할 리스트
        for i, text in enumerate(texts):  # 각 텍스트를 순회
            embedding = self._embed_single(text)  # 하나씩 임베딩
            embeddings.append(embedding)  # 결과 추가

            # 분당 100회 제한 대응: 요청 사이에 0.7초 대기
            # (100회 / 60초 = 1.67회/초 → 0.6초 간격이 한계, 여유 두고 0.7초)
            if i < len(texts) - 1:  # 마지막 요청 뒤에는 대기 불필요
                time.sleep(0.7)

        return embeddings  # 전체 벡터 리스트 반환

    def embed_query(self, text):
        """
        질문 텍스트 하나를 벡터로 변환하는 함수 (LangChain 호환 인터페이스)

        파라미터:
            text (str): 질문 텍스트

        반환값:
            list[float]: 임베딩 벡터
        """
        return self._embed_single(text)  # 내부 함수 호출


# ============================================
# Ollama 임베딩을 REST API로 호출하는 커스텀 클래스
# ============================================
# Ollama란?
#   로컬 PC/서버에서 LLM과 임베딩 모델을 쉽게 실행해주는 도구
#   Docker처럼 "ollama pull 모델명"으로 모델을 다운받고,
#   http://localhost:11434 에서 REST API를 제공한다
#
# 왜 Ollama를 쓰나?
#   - API 비용 없음 (완전 무료)
#   - 인터넷 불필요 (오프라인 동작)
#   - qwen3-embedding은 한국어 임베딩 성능이 우수한 최신 모델
# ============================================
class OllamaEmbedding:
    """
    Ollama 임베딩 API를 호출하는 클래스

    LangChain의 임베딩 인터페이스와 호환되도록
    embed_documents()와 embed_query() 메서드를 구현했다

    Ollama API 엔드포인트:
        POST http://localhost:11434/api/embed
        요청: {"model": "모델명", "input": ["텍스트1", "텍스트2"]}
        응답: {"embeddings": [[벡터1], [벡터2]]}
    """

    def __init__(self, model="qwen3-embedding:8b", base_url="http://localhost:11434"):
        """
        파라미터:
            model (str): Ollama에 설치된 임베딩 모델명 (기본값: qwen3-embedding:8b)
                - 설치 방법: ollama pull qwen3-embedding:8b
            base_url (str): Ollama 서버 주소 (기본값: http://localhost:11434)
                - 같은 서버에서 실행 중이면 기본값 그대로 사용
                - 다른 서버에서 실행 중이면 해당 서버 IP로 변경
        """
        self.model = model  # 사용할 임베딩 모델명
        self.base_url = base_url  # Ollama 서버 주소
        # Ollama 임베딩 API 엔드포인트
        self.url = f"{base_url}/api/embed"

    def embed_documents(self, texts):
        """
        여러 텍스트를 한번에 벡터로 변환하는 함수 (LangChain 호환 인터페이스)

        Ollama의 /api/embed는 여러 텍스트를 한 번에 처리할 수 있어서
        Google API처럼 하나씩 보내지 않아도 된다 → 훨씬 빠름

        파라미터:
            texts (list[str]): 임베딩할 텍스트 리스트

        반환값:
            list[list[float]]: 임베딩 벡터 리스트
        """
        # Ollama /api/embed 요청 본문
        # "input"에 텍스트 리스트를 넣으면 한 번에 처리됨
        payload = {
            "model": self.model,  # 사용할 모델
            "input": texts,  # 임베딩할 텍스트 리스트
        }

        response = requests.post(self.url, json=payload)  # API 호출

        if response.status_code == 200:  # 성공 시
            result = response.json()
            return result["embeddings"]  # 벡터 리스트 반환
        else:
            raise Exception(f"❌ Ollama 임베딩 에러 ({response.status_code}): {response.text}")

    def embed_query(self, text):
        """
        질문 텍스트 하나를 벡터로 변환하는 함수 (LangChain 호환 인터페이스)

        파라미터:
            text (str): 질문 텍스트

        반환값:
            list[float]: 임베딩 벡터
        """
        # 텍스트 하나도 리스트로 감싸서 보내고, 첫 번째 결과만 반환
        result = self.embed_documents([text])
        return result[0]


# 지원하는 임베딩 모델 목록 (Streamlit UI에서 드롭다운으로 보여줄 용도)
EMBEDDING_PROVIDERS = {
    "google": {
        "name": "[유료] Google (embedding-001)",
        "type": "api",  # API 호출 방식
        "description": "Google API, $0.15/1M토큰, 텍스트 전용, 안정적",
    },
    "google2": {
        "name": "[유료] Google (Gemini Embedding 2)",
        "type": "api",  # API 호출 방식
        "description": "Google API, $0.20/1M토큰, 멀티모달 지원, 최신 모델",
    },
    "openai": {
        "name": "[유료] OpenAI (text-embedding-3-small)",
        "type": "api",
        "description": "OpenAI API, $0.02/1M토큰, 가장 저렴, 가성비 최고",
    },
    "ollama": {
        "name": "[무료] Qwen3 Embedding (Ollama)",
        "type": "local",  # 로컬 Ollama 서버에서 실행
        "description": "Ollama 로컬 실행, 한국어 우수, API 비용 없음",
    },
}


def get_embedding_function(provider="google"):
    """
    임베딩 모델을 생성하여 반환하는 함수

    파라미터:
        provider (str): 임베딩 모델 제공자
            - "google": Google 임베딩 (기본값, 무료 티어 있음)
            - "openai": OpenAI 임베딩 (유료)
            - "ollama": Ollama 로컬 임베딩 (qwen3-embedding)

    반환값:
        임베딩 함수 객체 (LangChain 호환)
    """
    load_dotenv()  # .env 파일에서 API 키를 환경변수로 로드

    # ============================
    # API 방식: Google (REST API 직접 호출)
    # ============================
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")  # .env에서 Google API 키 가져오기
        if not api_key:  # 키가 없으면
            raise ValueError("❌ .env 파일에 GOOGLE_API_KEY가 없습니다.")  # 에러 발생

        # SDK 대신 REST API를 직접 호출하는 커스텀 클래스 사용
        # (langchain-google-genai SDK에 한국어 인코딩 버그가 있어서 우회)
        embedding_function = GoogleEmbeddingDirect(api_key=api_key)
        print("✅ Google 임베딩 모델 로드 완료 (models/embedding-001, REST API)")

    # ============================
    # API 방식: Google Embedding 2 (REST API 직접 호출)
    # ============================
    elif provider == "google2":
        api_key = os.getenv("GOOGLE_API_KEY")  # .env에서 Google API 키 가져오기
        if not api_key:  # 키가 없으면
            raise ValueError("❌ .env 파일에 GOOGLE_API_KEY가 없습니다.")  # 에러 발생

        # Embedding 1과 같은 커스텀 클래스 사용, 모델명만 다름
        embedding_function = GoogleEmbeddingDirect(api_key=api_key, model="gemini-embedding-2-preview")
        print("✅ Google 임베딩 모델 로드 완료 (Gemini Embedding 2, REST API)")

    # ============================
    # API 방식: OpenAI
    # ============================
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # OpenAI 임베딩 클래스

        api_key = os.getenv("OPENAI_API_KEY")  # .env에서 OpenAI API 키 가져오기
        if not api_key:  # 키가 없으면
            raise ValueError("❌ .env 파일에 OPENAI_API_KEY가 없습니다.")  # 에러 발생

        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",  # OpenAI 임베딩 모델명 (가장 저렴)
            openai_api_key=api_key,  # API 키 전달
        )
        print("✅ OpenAI 임베딩 모델 로드 완료 (text-embedding-3-small)")

    # ============================
    # 로컬 방식: Ollama (qwen3-embedding:8b)
    # ============================
    # 사전 준비:
    #   1. Ollama 설치: curl -fsSL https://ollama.com/install.sh | sh
    #   2. 모델 다운로드: ollama pull qwen3-embedding:8b
    #   3. Ollama 서버 실행 중인지 확인: ollama list
    elif provider == "ollama":
        embedding_function = OllamaEmbedding(
            model="qwen3-embedding:8b",  # Ollama에 설치된 임베딩 모델명
            base_url="http://localhost:11434",  # Ollama 서버 주소
        )
        print("✅ Ollama 임베딩 모델 로드 완료 (qwen3-embedding:8b)")

    else:  # 지원하지 않는 provider가 입력된 경우
        supported = ", ".join(EMBEDDING_PROVIDERS.keys())  # 지원 목록 문자열 생성
        raise ValueError(f"❌ 지원하지 않는 임베딩 제공자: {provider} (지원: {supported})")

    return embedding_function  # 생성된 임베딩 함수 반환


def store_chunks_to_chromadb(chunks, embedding_function, collection_name="rag_documents", db_path="chroma_db", provider="google"):
    """
    청크들을 ChromaDB에 임베딩하여 저장하는 함수

    파라미터:
        chunks (list[dict]): chunker.py에서 생성된 청크 리스트
            - "text": 청크 텍스트
            - "metadata": 출처 정보 (파일명, 페이지 등)
        embedding_function: get_embedding_function()으로 생성한 임베딩 함수
        collection_name (str): ChromaDB 컬렉션 이름 (기본값: "rag_documents")
        db_path (str): ChromaDB 저장 경로 (기본값: "chroma_db")

    반환값:
        chromadb.Collection: 저장된 ChromaDB 컬렉션 객체
    """
    # --- ChromaDB 클라이언트 생성 ---
    # PersistentClient: 데이터를 디스크에 저장하여 서버 재시작 후에도 유지
    client = chromadb.PersistentClient(path=db_path)

    # --- 기존 컬렉션이 있으면 삭제 후 새로 생성 ---
    # PDF를 다시 업로드하면 기존 데이터와 중복되므로 초기화
    try:
        client.delete_collection(name=collection_name)  # 기존 컬렉션 삭제 시도
        print(f"🗑️  기존 컬렉션 '{collection_name}' 삭제 완료")
    except Exception:  # 컬렉션이 없으면 에러 발생 → 무시 (버전마다 에러 타입이 다름)
        pass  # 처음 실행할 때는 삭제할 컬렉션이 없으니 그냥 넘어감

# 새 컬렉션 생성
    collection = client.create_collection(
        name=collection_name,  # 컬렉션 이름
        metadata={
            "hnsw:space": "cosine",  # 유사도 계산 방식: 코사인 유사도
            "embedding_provider": provider,  # 어떤 임베딩 모델로 저장했는지 기록
        },
    )
    print(f"📦 새 컬렉션 '{collection_name}' 생성 완료")

    # --- 청크를 배치 단위로 임베딩하여 저장 ---
    # 한번에 너무 많이 보내면 API 제한에 걸릴 수 있으므로 배치로 나눔
    batch_size = 50  # 한 번에 처리할 청크 수
    total_chunks = len(chunks)  # 전체 청크 수

    for i in range(0, total_chunks, batch_size):  # 0, 50, 100, 150, ... 씩 건너뛰며 반복
        batch = chunks[i:i + batch_size]  # 현재 배치의 청크들을 슬라이싱

        # 각 청크에서 필요한 데이터 추출
        texts = [chunk["text"] for chunk in batch]  # 텍스트 리스트
        metadatas = [chunk["metadata"] for chunk in batch]  # 메타데이터 리스트

        # 각 청크에 고유 ID 부여 (ChromaDB에서 필수)
        # 예: "chunk_000", "chunk_001", ...
        ids = [f"chunk_{i + j:04d}" for j in range(len(batch))]
        # :04d → 4자리 숫자로 표현 (0001, 0002, ..., 0050, ...)

        # 임베딩 생성: 텍스트를 벡터로 변환
        embeddings = embedding_function.embed_documents(texts)
        # 예: ["머신비전 품질검사", "SVM 알고리즘"]
        # → [[0.12, -0.34, ...], [0.56, 0.23, ...]]

        # ChromaDB에 저장
        collection.add(
            ids=ids,  # 고유 ID 리스트
            documents=texts,  # 원본 텍스트 (검색 결과에서 보여줄 용도)
            embeddings=embeddings,  # 벡터 (유사도 검색에 사용)
            metadatas=metadatas,  # 메타데이터 (출처 표시 용도)
        )

        # 진행 상황 출력
        processed = min(i + batch_size, total_chunks)  # 현재까지 처리된 청크 수
        print(f"   💾 {processed}/{total_chunks} 청크 저장 완료")

    print(f"\n✅ ChromaDB 저장 완료! (총 {total_chunks}개 청크, 경로: {db_path})")
    return collection  # 저장된 컬렉션 반환


def load_chromadb_collection(collection_name="rag_documents", db_path="chroma_db"):
    """
    이미 저장된 ChromaDB 컬렉션을 불러오는 함수
    (서버 재시작 후 다시 임베딩하지 않고 기존 데이터를 사용할 때)

    파라미터:
        collection_name (str): 불러올 컬렉션 이름
        db_path (str): ChromaDB 저장 경로

    반환값:
        chromadb.Collection: 불러온 컬렉션 객체
    """
    client = chromadb.PersistentClient(path=db_path)  # 저장된 DB 경로로 클라이언트 생성

    try:
        collection = client.get_collection(name=collection_name)  # 컬렉션 가져오기
        print(f"✅ 기존 컬렉션 로드 완료: {collection.count()}개 청크")  # 저장된 청크 수 출력
        return collection  # 컬렉션 반환
    except Exception:  # 컬렉션이 없으면
        print(f"❌ 컬렉션 '{collection_name}'을 찾을 수 없습니다. 먼저 임베딩을 실행하세요.")
        return None  # None 반환


# ============================================
# 문서 관리 (CRUD + TTL) 함수들
# ============================================
# 왜 필요한가?
#   기존에는 PDF 업로드 시 DB 전체를 삭제하고 다시 넣었다.
#   이제는 문서별 고유 ID(파일 해시)를 사용하여:
#   - 이미 저장된 문서는 스킵 (중복 방지)
#   - 특정 문서만 삭제 (전체 재임베딩 불필요)
#   - 유효기간(TTL)이 지난 문서를 자동 삭제
#
# 문서 레지스트리란?
#   doc_registry.json 파일에 문서 단위 정보를 저장한다.
#   ChromaDB에는 청크 단위로 데이터가 저장되어 있어서,
#   문서 목록을 보려면 모든 청크를 스캔해야 하는데 이는 비효율적이다.
#   그래서 별도 JSON 파일에 문서 메타데이터를 관리한다.
#
#   구조 예시:
#   {
#     "a1b2c3d4": {
#       "filename": "머신비전.pdf",
#       "num_pages": 45,
#       "num_chunks": 120,
#       "uploaded_at": "2026-04-10 14:30:00",
#       "expires_at": "2026-05-10 14:30:00",  ← null이면 무제한
#       "embedding_provider": "ollama"
#     },
#     ...
#   }
# ============================================

# TTL(유효기간) 선택지 정의
# Streamlit UI의 드롭다운에서 보여줄 옵션들
TTL_OPTIONS = {
    "7d": {"label": "7일", "days": 7},
    "30d": {"label": "30일", "days": 30},
    "90d": {"label": "90일", "days": 90},
    "365d": {"label": "1년", "days": 365},
    "none": {"label": "무제한", "days": None},  # None = 만료 없음
}


def compute_file_hash(file_data):
    """
    파일 내용의 SHA-256 해시값을 계산하는 함수

    왜 해시를 쓰나?
        - 같은 내용의 파일이면 파일명이 달라도 같은 해시값이 나옴
          예: "보고서.pdf"와 "보고서(1).pdf"가 내용이 같으면 → 같은 해시
        - 다른 내용의 파일이면 파일명이 같아도 다른 해시값이 나옴
          예: 두 사람이 각각 "report.pdf"를 올려도 내용이 다르면 → 다른 해시
        - 이를 통해 파일명에 의존하지 않고 정확한 중복 판별이 가능

    파라미터:
        file_data (bytes): 파일의 바이너리 데이터
            - Streamlit의 file_uploader에서 f.getbuffer()로 얻은 데이터
            - 또는 open(path, "rb").read()로 읽은 데이터

    반환값:
        str: SHA-256 해시값의 앞 16자리 (8바이트)
            - 전체 64자리 중 16자리만 사용 (문서 ID로 충분히 고유)
            - 예: "a1b2c3d4e5f67890"
    """
    # SHA-256: 256비트(32바이트) 해시 → 16진수로 64자리 문자열
    # hexdigest()[:16]: 앞 16자리만 사용 (충돌 확률 극히 낮음)
    return hashlib.sha256(file_data).hexdigest()[:16]


def _get_registry_path(db_path="chroma_db"):
    """
    문서 레지스트리 JSON 파일의 경로를 반환하는 내부 함수

    파라미터:
        db_path (str): ChromaDB 저장 경로

    반환값:
        str: doc_registry.json의 전체 경로
    """
    return os.path.join(db_path, "doc_registry.json")


def _load_registry(db_path="chroma_db"):
    """
    문서 레지스트리를 파일에서 읽어오는 내부 함수

    파라미터:
        db_path (str): ChromaDB 저장 경로

    반환값:
        dict: 문서 레지스트리 딕셔너리
            - 키: doc_hash (파일 해시값)
            - 값: 문서 메타데이터 (filename, num_pages, num_chunks, ...)
            - 파일이 없으면 빈 딕셔너리 {} 반환
    """
    registry_path = _get_registry_path(db_path)

    if not os.path.exists(registry_path):  # 파일이 없으면 (첫 실행)
        return {}  # 빈 딕셔너리 반환

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)  # JSON 파일 → 파이썬 딕셔너리
    except (json.JSONDecodeError, IOError):  # 파일이 깨졌거나 읽기 실패
        print("⚠️  doc_registry.json 읽기 실패 → 빈 레지스트리로 초기화")
        return {}


def _save_registry(registry, db_path="chroma_db"):
    """
    문서 레지스트리를 JSON 파일로 저장하는 내부 함수

    파라미터:
        registry (dict): 저장할 문서 레지스트리 딕셔너리
        db_path (str): ChromaDB 저장 경로
    """
    # ChromaDB 폴더가 없으면 생성 (첫 실행 시)
    os.makedirs(db_path, exist_ok=True)

    registry_path = _get_registry_path(db_path)

    with open(registry_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False: 한국어 파일명이 \uXXXX로 깨지지 않도록
        # indent=2: 사람이 읽기 쉽게 들여쓰기
        json.dump(registry, f, ensure_ascii=False, indent=2)


def get_stored_documents(db_path="chroma_db"):
    """
    저장된 문서 목록을 조회하는 함수

    Streamlit의 문서 관리 탭에서 테이블로 보여줄 데이터를 반환한다.

    파라미터:
        db_path (str): ChromaDB 저장 경로

    반환값:
        list[dict]: 문서 정보 리스트 (업로드 최신순 정렬)
            각 항목:
            - "doc_hash": 문서 해시 (고유 ID)
            - "filename": 파일명
            - "num_pages": 페이지 수
            - "num_chunks": 청크 수
            - "uploaded_at": 업로드 일시 (문자열)
            - "expires_at": 만료 일시 (문자열 또는 None)
            - "is_expired": 현재 만료 여부 (bool)
            - "embedding_provider": 임베딩 모델명
    """
    registry = _load_registry(db_path)
    now = datetime.now()  # 현재 시각 (만료 여부 판단용)

    documents = []

    for doc_hash, info in registry.items():
        # 만료 여부 판단
        expires_at = info.get("expires_at")  # "2026-05-10 14:30:00" 또는 None
        is_expired = False

        if expires_at:  # 유효기간이 설정되어 있으면
            # 문자열 → datetime 객체로 변환하여 현재 시각과 비교
            expire_dt = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
            is_expired = now > expire_dt  # 현재가 만료일보다 후면 만료

        documents.append({
            "doc_hash": doc_hash,
            "filename": info.get("filename", "알 수 없음"),
            "num_pages": info.get("num_pages", 0),
            "num_chunks": info.get("num_chunks", 0),
            "uploaded_at": info.get("uploaded_at", "알 수 없음"),
            "expires_at": expires_at,  # None이면 무제한
            "is_expired": is_expired,
            "embedding_provider": info.get("embedding_provider", "알 수 없음"),
        })

    # 업로드 최신순으로 정렬 (가장 최근 업로드가 맨 위)
    documents.sort(key=lambda x: x["uploaded_at"], reverse=True)

    return documents


def add_document_to_chromadb(chunks, doc_hash, file_info, embedding_function,
                              collection_name="rag_documents", db_path="chroma_db",
                              provider="google", ttl="none"):
    """
    문서 하나를 ChromaDB에 추가하는 함수 (중복 체크 포함)

    기존 store_chunks_to_chromadb()와의 차이:
        - 기존: DB 전체를 삭제하고 다시 넣음
        - 이 함수: 기존 데이터는 유지하면서, 새 문서만 추가함

    파라미터:
        chunks (list[dict]): 이 문서에서 생성된 청크 리스트
            - "text": 청크 텍스트
            - "metadata": 출처 정보
        doc_hash (str): 문서 해시값 (compute_file_hash()의 반환값)
        file_info (dict): 파일 정보
            - "filename": 파일명 (예: "머신비전.pdf")
            - "num_pages": 페이지 수
        embedding_function: 임베딩 함수 객체
        collection_name (str): ChromaDB 컬렉션 이름
        db_path (str): ChromaDB 저장 경로
        provider (str): 임베딩 모델 제공자명
        ttl (str): 유효기간 키 ("7d", "30d", "90d", "365d", "none")

    반환값:
        dict: 처리 결과
            - "status": "added" (추가됨) / "skipped" (이미 존재) / "error" (실패)
            - "message": 사용자에게 보여줄 메시지
            - "doc_hash": 문서 해시
            - "num_chunks": 추가된 청크 수
    """
    # ── 1단계: 중복 체크 ──
    # 레지스트리에서 이 해시가 이미 있는지 확인
    registry = _load_registry(db_path)

    if doc_hash in registry:
        existing = registry[doc_hash]
        return {
            "status": "skipped",
            "message": f"⏭️ '{existing['filename']}' — 이미 저장된 문서입니다 (해시: {doc_hash[:8]}...)",
            "doc_hash": doc_hash,
            "num_chunks": 0,
        }

    # ── 2단계: ChromaDB 컬렉션 가져오기 또는 생성 ──
    client = chromadb.PersistentClient(path=db_path)

    try:
        # 기존 컬렉션이 있으면 가져오기 (데이터 유지)
        collection = client.get_collection(name=collection_name)

        # 임베딩 모델 불일치 체크
        # 기존 컬렉션에 저장된 임베딩 모델과 지금 선택한 모델이 다르면 거부
        # 같은 컬렉션에 서로 다른 임베딩 벡터가 섞이면 검색 품질이 망가지기 때문
        saved_provider = collection.metadata.get("embedding_provider", "unknown")
        if saved_provider != provider:
            saved_name = EMBEDDING_PROVIDERS.get(saved_provider, {}).get("name", saved_provider)
            new_name = EMBEDDING_PROVIDERS.get(provider, {}).get("name", provider)
            return {
                "status": "error",
                "message": (
                    f"⚠️ '{file_info['filename']}' — 임베딩 모델 불일치!\n"
                    f"   기존: {saved_name}\n"
                    f"   선택: {new_name}\n"
                    f"   같은 모델로 변경하거나, 전체 초기화 후 다시 임베딩하세요."
                ),
                "doc_hash": doc_hash,
                "num_chunks": 0,
            }

    except Exception:
        # 컬렉션이 없으면 새로 생성 (첫 문서 추가 시)
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # 코사인 유사도 사용
                "embedding_provider": provider,  # 임베딩 모델 기록
            },
        )
        print(f"📦 새 컬렉션 '{collection_name}' 생성 완료")

    # ── 3단계: 청크 임베딩 + ChromaDB 저장 ──
    batch_size = 50  # 한 번에 처리할 청크 수
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]

        texts = [chunk["text"] for chunk in batch]

        # 메타데이터에 doc_hash 추가 (나중에 문서 단위 삭제 시 사용)
        metadatas = []
        for chunk in batch:
            meta = chunk["metadata"].copy()  # 원본 메타데이터 복사
            meta["doc_hash"] = doc_hash  # 문서 해시 추가
            metadatas.append(meta)

        # 청크 ID: "{doc_hash}_chunk_{번호}" 형식
        # 예: "a1b2c3d4e5f67890_chunk_0001"
        # doc_hash를 포함하여 문서별로 고유한 ID가 되도록 함
        ids = [f"{doc_hash}_chunk_{i + j:04d}" for j in range(len(batch))]

        # 임베딩 생성
        embeddings = embedding_function.embed_documents(texts)

        # ChromaDB에 저장
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        processed = min(i + batch_size, total_chunks)
        print(f"   💾 {processed}/{total_chunks} 청크 저장 완료")

    # ── 4단계: 유효기간(TTL) 계산 ──
    now = datetime.now()
    uploaded_at = now.strftime("%Y-%m-%d %H:%M:%S")  # "2026-04-10 14:30:00"

    # TTL_OPTIONS에서 days 값을 가져와서 만료일 계산
    ttl_info = TTL_OPTIONS.get(ttl, TTL_OPTIONS["none"])
    if ttl_info["days"] is not None:
        # 현재 시각 + N일 = 만료 시각
        expires_at = (now + timedelta(days=ttl_info["days"])).strftime("%Y-%m-%d %H:%M:%S")
    else:
        expires_at = None  # 무제한

    # ── 5단계: 레지스트리에 문서 정보 기록 ──
    registry[doc_hash] = {
        "filename": file_info["filename"],
        "num_pages": file_info["num_pages"],
        "num_chunks": total_chunks,
        "uploaded_at": uploaded_at,
        "expires_at": expires_at,
        "embedding_provider": provider,
    }
    _save_registry(registry, db_path)

    # 만료일 표시 문자열 생성
    ttl_display = f" (만료: {expires_at})" if expires_at else " (무제한)"

    return {
        "status": "added",
        "message": f"✅ '{file_info['filename']}' — {total_chunks}개 청크 저장 완료{ttl_display}",
        "doc_hash": doc_hash,
        "num_chunks": total_chunks,
    }


def delete_document_from_chromadb(doc_hash, collection_name="rag_documents", db_path="chroma_db"):
    """
    특정 문서를 ChromaDB에서 삭제하는 함수

    문서의 모든 청크를 ChromaDB에서 제거하고,
    레지스트리에서도 해당 문서 정보를 삭제한다.

    파라미터:
        doc_hash (str): 삭제할 문서의 해시값
        collection_name (str): ChromaDB 컬렉션 이름
        db_path (str): ChromaDB 저장 경로

    반환값:
        dict: 처리 결과
            - "status": "deleted" (삭제됨) / "not_found" (문서 없음) / "error" (실패)
            - "message": 사용자에게 보여줄 메시지
            - "filename": 삭제된 파일명
    """
    # ── 1단계: 레지스트리에서 문서 확인 ──
    registry = _load_registry(db_path)

    if doc_hash not in registry:
        return {
            "status": "not_found",
            "message": f"❌ 해시 '{doc_hash[:8]}...'에 해당하는 문서를 찾을 수 없습니다.",
            "filename": None,
        }

    filename = registry[doc_hash]["filename"]

    # ── 2단계: ChromaDB에서 해당 문서의 청크 삭제 ──
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)

        # where 필터: metadata의 doc_hash가 일치하는 청크만 선택
        # ChromaDB의 where 절은 메타데이터 기반 필터링을 지원
        collection.delete(
            where={"doc_hash": doc_hash}  # 이 문서의 청크만 삭제
        )
        print(f"🗑️  ChromaDB에서 '{filename}' 청크 삭제 완료")

    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ ChromaDB 삭제 실패: {e}",
            "filename": filename,
        }

    # ── 3단계: 레지스트리에서 문서 정보 제거 ──
    del registry[doc_hash]
    _save_registry(registry, db_path)

    return {
        "status": "deleted",
        "message": f"🗑️ '{filename}' 삭제 완료",
        "filename": filename,
    }


def cleanup_expired_documents(collection_name="rag_documents", db_path="chroma_db"):
    """
    유효기간(TTL)이 만료된 문서를 자동 삭제하는 함수

    앱 시작 시 또는 문서 관리 탭 열 때 호출하여
    만료된 문서를 자동으로 정리한다.

    파라미터:
        collection_name (str): ChromaDB 컬렉션 이름
        db_path (str): ChromaDB 저장 경로

    반환값:
        list[str]: 삭제된 파일명 리스트
            - 예: ["구버전_보고서.pdf", "임시_테스트.pdf"]
            - 삭제된 문서가 없으면 빈 리스트 []
    """
    registry = _load_registry(db_path)
    now = datetime.now()
    deleted_files = []  # 삭제된 파일명을 저장할 리스트

    # 만료된 문서 해시를 먼저 수집 (순회 중 딕셔너리 수정 방지)
    expired_hashes = []

    for doc_hash, info in registry.items():
        expires_at = info.get("expires_at")

        if expires_at:  # 유효기간이 설정된 문서만 검사
            expire_dt = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")

            if now > expire_dt:  # 만료됨
                expired_hashes.append(doc_hash)

    # 만료된 문서들을 하나씩 삭제
    for doc_hash in expired_hashes:
        filename = registry[doc_hash]["filename"]
        result = delete_document_from_chromadb(doc_hash, collection_name, db_path)

        if result["status"] == "deleted":
            deleted_files.append(filename)
            print(f"🕐 만료 문서 자동 삭제: {filename}")
        else:
            print(f"⚠️  만료 문서 삭제 실패: {filename} — {result['message']}")

    if deleted_files:
        print(f"🧹 만료 문서 {len(deleted_files)}개 자동 삭제 완료")

    return deleted_files


def reset_all_documents(collection_name="rag_documents", db_path="chroma_db"):
    """
    모든 문서와 임베딩을 초기화하는 함수 (전체 삭제)

    ChromaDB 컬렉션과 문서 레지스트리를 모두 삭제한다.
    "전체 초기화" 버튼에서 사용.

    파라미터:
        collection_name (str): ChromaDB 컬렉션 이름
        db_path (str): ChromaDB 저장 경로

    반환값:
        dict: 처리 결과
            - "status": "reset" (초기화 완료) / "error" (실패)
            - "message": 사용자에게 보여줄 메시지
    """
    try:
        # 1. ChromaDB 컬렉션 삭제
        client = chromadb.PersistentClient(path=db_path)
        try:
            client.delete_collection(name=collection_name)
            print(f"🗑️  컬렉션 '{collection_name}' 삭제 완료")
        except Exception:
            pass  # 컬렉션이 없으면 무시

        # 2. 레지스트리 파일 초기화 (빈 딕셔너리로 덮어쓰기)
        _save_registry({}, db_path)
        print("🗑️  문서 레지스트리 초기화 완료")

        return {
            "status": "reset",
            "message": "🗑️ 모든 문서와 임베딩이 초기화되었습니다.",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ 초기화 실패: {e}",
        }




# ============================================
# 직접 실행 시 테스트 코드
# 실행 방법: python modules/embedder.py
# ============================================
if __name__ == "__main__":
    from parser import parse_all_pdfs  # PDF 파싱 모듈
    from chunker import chunk_all_documents  # 청킹 모듈

    PDF_DIR = "data/pdfs"  # PDF 폴더 경로
    DB_PATH = "chroma_db"  # ChromaDB 저장 경로

    # 1단계: PDF 파싱
    print("=" * 50)
    print("1단계: PDF 파싱")
    print("=" * 50)
    documents = parse_all_pdfs(PDF_DIR)

    if not documents:
        print("파싱된 문서가 없습니다. data/pdfs 폴더에 PDF를 넣어주세요.")
        exit()

    # 2단계: 청킹
    print("\n" + "=" * 50)
    print("2단계: 텍스트 청킹")
    print("=" * 50)
    chunks = chunk_all_documents(documents, chunk_size=512, chunk_overlap=50)

    # 3단계: 임베딩 + ChromaDB 저장
    print("\n" + "=" * 50)
    print("3단계: 임베딩 + ChromaDB 저장")
    print("=" * 50)
    embedding_fn = get_embedding_function(provider="google")  # Google 임베딩 사용
    collection = store_chunks_to_chromadb(chunks, embedding_fn, db_path=DB_PATH)

    # 4단계: 저장 확인 (간단한 검색 테스트)
    print("\n" + "=" * 50)
    print("4단계: 검색 테스트")
    print("=" * 50)
    test_query = "머신비전 품질검사 방법은?"  # 테스트 질문
    query_embedding = embedding_fn.embed_query(test_query)  # 질문을 벡터로 변환

    results = collection.query(
        query_embeddings=[query_embedding],  # 질문 벡터로 검색
        n_results=3,  # 상위 3개 결과
    )

    print(f"\n🔍 테스트 질문: '{test_query}'")
    print(f"   검색된 청크 수: {len(results['documents'][0])}개\n")

    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"   --- 결과 {i + 1} ---")
        print(f"   출처: {metadata['source']} / {metadata['page']}페이지")
        print(f"   내용: {doc[:150]}...")
        
        print()