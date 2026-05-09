# ============================================
# retriever.py - 사용자 질문과 유사한 청크를 검색하는 모듈
# ============================================
# 목적: 사용자가 질문을 입력하면, ChromaDB에 저장된 청크들 중에서
#       의미적으로 가장 유사한 청크 Top-K개를 찾아 반환한다
#
# 검색 원리:
#   1. 사용자 질문을 임베딩 모델로 벡터로 변환
#   2. ChromaDB에 저장된 청크 벡터들과 코사인 유사도 비교
#   3. 유사도가 높은 순서대로 Top-K개를 반환
#
# 예시:
#   질문: "머신비전 품질검사 방법은?"
#   → 질문 벡터와 가장 가까운 청크 3개를 ChromaDB에서 찾아 반환
# ============================================

import chromadb  # 벡터 데이터베이스 (저장된 임베딩에서 유사 청크 검색)


def load_collection(collection_name="rag_documents", db_path="chroma_db"):
    """
    저장된 ChromaDB 컬렉션을 불러오는 함수

    파라미터:
        collection_name (str): 컬렉션 이름 (기본값: "rag_documents")
        db_path (str): ChromaDB 저장 경로 (기본값: "chroma_db")

    반환값:
        chromadb.Collection: 불러온 컬렉션 객체
        None: 컬렉션이 없으면 None 반환
    """
    # PersistentClient: 디스크에 저장된 DB를 불러옴
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(name=collection_name)  # 컬렉션 가져오기
        print(f"✅ 컬렉션 로드 완료: {collection.count()}개 청크")  # 저장된 청크 수 출력
        return collection  # 컬렉션 반환
    except Exception:  # 컬렉션이 없으면
        print(f"❌ 컬렉션 '{collection_name}'을 찾을 수 없습니다.")
        return None  # None 반환


def search_similar_chunks(query, embedding_function, collection, top_k=3, threshold=0.0):
    """
    사용자 질문과 가장 유사한 청크를 ChromaDB에서 검색하는 함수

    파라미터:
        query (str): 사용자의 질문 텍스트 (예: "머신비전 품질검사 방법은?")
        embedding_function: embedder.py에서 생성한 임베딩 함수 객체
            - embed_query() 메서드를 가지고 있어야 함
        collection (chromadb.Collection): 검색할 ChromaDB 컬렉션
        top_k (int): 반환할 유사 청크 개수 (기본값: 3)
            - 너무 적으면 관련 정보를 놓칠 수 있음
            - 너무 많으면 LLM에 불필요한 정보(노이즈)가 전달됨
            - 일반적으로 3~5가 적절
        threshold (float): 유사도 임계값 (기본값: 0.0 = 필터링 안 함)
            - 이 값 미만의 유사도를 가진 청크는 결과에서 제외
            - 예: 0.7이면 유사도 70% 미만인 청크는 버림
            - "똷!" 같은 무의미한 질문에 억지 답변 방지

    반환값:
        list[dict]: 검색된 청크 정보 리스트 (유사도 높은 순)
            - "text": 청크 텍스트
            - "metadata": 출처 정보 (파일명, 페이지 등)
            - "distance": 코사인 거리 (0에 가까울수록 유사)
            - "similarity": 코사인 유사도 (1에 가까울수록 유사)
    """
    # 1단계: 사용자 질문을 벡터로 변환
    # "머신비전 품질검사 방법은?" → [0.12, -0.34, 0.56, ...] (768차원)
    query_embedding = embedding_function.embed_query(query)

    # 2단계: ChromaDB에서 유사도 검색
    # query_embeddings: 검색할 벡터 (리스트 안에 리스트 형태)
    # n_results: 상위 몇 개를 반환할지
    # include: 결과에 포함할 정보 (문서 텍스트, 메타데이터, 거리값)
    results = collection.query(
        query_embeddings=[query_embedding],  # 질문 벡터
        n_results=top_k,  # 상위 K개
        include=["documents", "metadatas", "distances"],  # 반환할 정보
    )

    # 3단계: 결과를 보기 좋은 형태로 정리
    # ChromaDB 결과는 이중 리스트 구조: results["documents"][0]이 실제 결과
    search_results = []  # 정리된 결과를 저장할 리스트
    filtered_count = 0  # 임계값 미달로 제외된 청크 수 (디버깅용)

    # zip으로 문서, 메타데이터, 거리를 묶어서 순회
    for doc, metadata, distance in zip(
        results["documents"][0],   # 청크 텍스트 리스트
        results["metadatas"][0],   # 메타데이터 리스트
        results["distances"][0],   # 코사인 거리 리스트
    ):
        similarity = 1 - distance  # 코사인 유사도 계산
        # ChromaDB의 cosine 거리 = 1 - 코사인유사도
        # 거리 0 = 유사도 1 (완전 동일)
        # 거리 1 = 유사도 0 (완전 무관)

        # 유사도 임계값 필터링: threshold 미만이면 제외
        # 예: threshold=0.7이면 유사도 0.64인 "똷!" 결과는 버려짐
        if similarity < threshold:
            filtered_count += 1  # 제외된 청크 카운트
            continue  # 이 청크는 건너뛰고 다음으로

        search_results.append({
            "text": doc,  # 청크 텍스트
            "metadata": metadata,  # 출처 정보 (source, page 등)
            "distance": distance,  # 코사인 거리 (낮을수록 유사)
            "similarity": similarity,  # 코사인 유사도 (높을수록 유사)
        })

    # 필터링 결과 출력 (몇 개가 걸러졌는지 알려줌)
    if filtered_count > 0:
        print(f"  ⚠️  유사도 {threshold} 미만 → {filtered_count}개 청크 제외됨")

    return search_results  # 검색 결과 반환


def format_search_results(search_results):
    """
    검색 결과를 LLM에 전달할 문맥(context) 텍스트로 포맷팅하는 함수

    파라미터:
        search_results (list[dict]): search_similar_chunks()의 반환값

    반환값:
        str: LLM 프롬프트에 삽입할 문맥 텍스트

    예시 출력:
        [출처 1] 06_Guidebook.pdf / 5페이지 (유사도: 0.82)
        머신비전은 카메라와 이미지 처리 기술을 활용하여...

        [출처 2] 06_Guidebook.pdf / 12페이지 (유사도: 0.76)
        품질검사 자동화 시스템은 SVM 알고리즘을 기반으로...
    """
    if not search_results:  # 검색 결과가 없으면
        return "관련 문서를 찾지 못했습니다."  # 기본 메시지 반환

    context_parts = []  # 각 청크의 포맷팅된 텍스트를 저장할 리스트

    for i, result in enumerate(search_results):  # 각 검색 결과를 순회
        source = result["metadata"].get("source", "알 수 없음")  # 파일명 (없으면 "알 수 없음")
        page = result["metadata"].get("page", "?")  # 페이지 번호 (없으면 "?")
        similarity = result["similarity"]  # 유사도 점수

        # 하나의 청크를 포맷팅
        part = (
            f"[출처 {i + 1}] {source} / {page}페이지 (유사도: {similarity:.2f})\n"  # 출처 헤더
            f"{result['text']}"  # 청크 텍스트
        )
        context_parts.append(part)  # 리스트에 추가

    # 모든 청크를 빈 줄 2개로 구분하여 합침
    return "\n\n".join(context_parts)


def print_search_results(search_results):
    """
    검색 결과를 터미널에 보기 좋게 출력하는 함수 (디버깅 및 테스트용)

    파라미터:
        search_results (list[dict]): search_similar_chunks()의 반환값
    """
    if not search_results:  # 결과가 없으면
        print("⚠️  검색 결과가 없습니다. (유사도 임계값 미달로 모두 제외되었을 수 있음)")
        return

    print(f"\n🔍 검색 결과: {len(search_results)}개 청크\n")

    for i, result in enumerate(search_results):  # 각 결과를 순회
        source = result["metadata"].get("source", "알 수 없음")
        page = result["metadata"].get("page", "?")
        similarity = result["similarity"]

        print(f"  --- 결과 {i + 1} (유사도: {similarity:.4f}) ---")  # 유사도 소수점 4자리
        print(f"  출처: {source} / {page}페이지")
        # 텍스트가 200자 넘으면 잘라서 보여줌
        text_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
        print(f"  내용: {text_preview}")
        print()  # 빈 줄


# ============================================
# 직접 실행 시 테스트 코드
# 실행 방법: python modules/retriever.py
# ============================================
if __name__ == "__main__":
    import sys
    import os

    # 프로젝트 루트를 Python 경로에 추가 (modules 폴더 밖의 설정 접근용)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from embedder import get_embedding_function  # 임베딩 함수 가져오기

    DB_PATH = "chroma_db"  # ChromaDB 저장 경로

    # 1단계: 컬렉션 로드
    print("=" * 50)
    print("1단계: ChromaDB 컬렉션 로드")
    print("=" * 50)
    collection = load_collection(db_path=DB_PATH)

    if collection is None:  # 컬렉션이 없으면
        print("먼저 embedder.py를 실행하여 임베딩을 저장하세요.")
        exit()

    # 2단계: 임베딩 함수 로드
    print("\n" + "=" * 50)
    print("2단계: 임베딩 함수 로드")
    print("=" * 50)
    embedding_fn = get_embedding_function(provider="google")

    # 3단계: 검색 테스트
    print("\n" + "=" * 50)
    print("3단계: 검색 테스트")
    print("=" * 50)

    # 여러 질문으로 테스트
    test_queries = [
        "머신비전 품질검사 방법은?",
        "열처리 공정에서 AI는 어떻게 활용되나요?",
        "OCR 기술이란 무엇인가요?",
    ]

    THRESHOLD = 0.7  # 유사도 임계값 (0.7 미만이면 결과 제외)

    for query in test_queries:
        print(f"\n{'─' * 40}")
        print(f"질문: {query}")
        print(f"{'─' * 40}")

        # 유사 청크 검색 (상위 3개, 유사도 0.7 이상만)
        results = search_similar_chunks(query, embedding_fn, collection, top_k=3, threshold=THRESHOLD)
        print_search_results(results)

        # LLM에 전달할 문맥 텍스트 확인
        context = format_search_results(results)
        print(f"📝 LLM에 전달할 문맥 길이: {len(context)}자")

    # 4단계: 무의미한 질문 테스트 (threshold 필터링 효과 확인)
    print("\n" + "=" * 50)
    print("4단계: 무의미한 질문 테스트 (threshold 필터링)")
    print("=" * 50)

    nonsense_query = "똷!"
    print(f"\n{'─' * 40}")
    print(f"질문: {nonsense_query}  (threshold={THRESHOLD})")
    print(f"{'─' * 40}")

    results = search_similar_chunks(nonsense_query, embedding_fn, collection, top_k=3, threshold=THRESHOLD)
    print_search_results(results)  # threshold 덕분에 결과가 0개여야 정상