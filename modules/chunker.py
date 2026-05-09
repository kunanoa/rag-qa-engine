# ============================================
# chunker.py - 텍스트를 청크(조각) 단위로 분할하는 모듈
# ============================================
# 목적: 긴 문서를 일정 크기의 청크로 나눠서 임베딩과 검색에 적합하게 만든다
# 핵심 개념:
#   - chunk_size: 한 청크의 최대 문자 수 (너무 크면 검색 정확도↓, 너무 작으면 문맥↓)
#   - chunk_overlap: 청크 간 겹치는 문자 수 (문맥 단절 방지)
# 사용 라이브러리: LangChain의 RecursiveCharacterTextSplitter
# ============================================

from langchain_text_splitters import RecursiveCharacterTextSplitter  # 재귀적 텍스트 분할기


def create_text_splitter(chunk_size=512, chunk_overlap=50):
    """
    텍스트 분할기(TextSplitter) 객체를 생성하는 함수

    파라미터:
        chunk_size (int): 한 청크의 최대 문자 수 (기본값: 512)
            - 256: 짧은 청크 → 검색 정확도 높지만 문맥 부족할 수 있음
            - 512: 중간 크기 → 균형 잡힌 선택 (기본 추천)
            - 1024: 긴 청크 → 문맥은 풍부하지만 검색 시 노이즈 가능
        chunk_overlap (int): 청크 간 겹치는 문자 수 (기본값: 50)
            - 겹침이 있어야 청크 경계에서 문맥이 끊기지 않음

    반환값:
        RecursiveCharacterTextSplitter: 설정된 텍스트 분할기 객체
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 청크 최대 크기 설정
        chunk_overlap=chunk_overlap,  # 청크 간 겹침 크기 설정
        separators=["\n\n", "\n", ". ", " ", ""],  # 분할 우선순위: 문단 > 줄바꿈 > 문장 > 단어 > 문자
        length_function=len,  # 길이 측정 함수 (문자 수 기준)
    )
    return text_splitter  # 생성된 분할기 반환


def chunk_single_document(document, text_splitter):
    """
    단일 문서(페이지)를 청크로 분할하는 함수

    파라미터:
        document (dict): parser.py에서 추출한 페이지 정보
            - "text": 페이지 텍스트
            - "metadata": 파일명, 페이지 번호 등
        text_splitter: create_text_splitter()로 생성한 분할기 객체

    반환값:
        list[dict]: 분할된 청크 리스트
            - "text": 청크 텍스트
            - "metadata": 원본 메타데이터 + 청크 번호
    """
    text = document["text"]  # 분할할 텍스트 추출
    metadata = document["metadata"]  # 메타데이터 추출

    # 텍스트가 비어있으면 빈 리스트 반환
    if not text or not text.strip():  # 텍스트가 없거나 공백만 있으면
        return []  # 빈 리스트 반환

    # RecursiveCharacterTextSplitter로 텍스트 분할
    text_chunks = text_splitter.split_text(text)  # 텍스트를 청크 리스트로 분할

    chunks = []  # 최종 청크 정보를 저장할 리스트

    for i, chunk_text in enumerate(text_chunks):  # 각 청크를 순회
        chunk = {  # 청크 정보를 딕셔너리로 구성
            "text": chunk_text,  # 청크 텍스트
            "metadata": {  # 메타데이터 (원본 정보 + 청크 고유 정보)
                "source": metadata["source"],  # 원본 파일명
                "page": metadata["page"],  # 원본 페이지 번호
                "total_pages": metadata["total_pages"],  # 전체 페이지 수
                "chunk_index": i,  # 이 페이지 내에서 몇 번째 청크인지 (0부터)
                "total_chunks_in_page": len(text_chunks),  # 이 페이지에서 총 몇 개 청크가 나왔는지
            }
        }
        chunks.append(chunk)  # 리스트에 추가

    return chunks  # 분할된 청크 리스트 반환


def chunk_all_documents(documents, chunk_size=512, chunk_overlap=50):
    """
    여러 문서(페이지)를 한번에 청킹하는 함수

    파라미터:
        documents (list[dict]): parser.py에서 추출한 전체 페이지 리스트
        chunk_size (int): 청크 크기 (기본값: 512)
        chunk_overlap (int): 청크 겹침 크기 (기본값: 50)

    반환값:
        list[dict]: 전체 청크 리스트
    """
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)  # 분할기 생성

    all_chunks = []  # 전체 청크를 저장할 리스트

    for document in documents:  # 각 문서(페이지)를 순회
        chunks = chunk_single_document(document, text_splitter)  # 해당 문서를 청크로 분할
        all_chunks.extend(chunks)  # 결과를 전체 리스트에 추가

    print(f"📦 청킹 완료: {len(documents)}개 페이지 → {len(all_chunks)}개 청크")  # 결과 출력
    print(f"   설정: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")  # 사용된 설정 출력

    return all_chunks  # 전체 청크 리스트 반환


def print_chunk_stats(chunks):
    """
    청크 통계를 출력하는 함수 (디버깅 및 실험용)

    파라미터:
        chunks (list[dict]): 청크 리스트
    """
    if not chunks:  # 청크가 없으면
        print("⚠️  청크가 없습니다.")  # 경고 출력
        return  # 함수 종료

    lengths = [len(chunk["text"]) for chunk in chunks]  # 각 청크의 문자 수를 리스트로 생성
    avg_length = sum(lengths) / len(lengths)  # 평균 문자 수 계산
    min_length = min(lengths)  # 가장 짧은 청크의 문자 수
    max_length = max(lengths)  # 가장 긴 청크의 문자 수

    # 어떤 PDF에서 왔는지 파일별 통계
    sources = {}  # 파일별 청크 수를 저장할 딕셔너리
    for chunk in chunks:  # 각 청크를 순회
        source = chunk["metadata"]["source"]  # 파일명 추출
        sources[source] = sources.get(source, 0) + 1  # 해당 파일의 청크 수 증가

    print(f"\n📊 청크 통계:")  # 통계 제목
    print(f"   총 청크 수: {len(chunks)}개")  # 전체 청크 수
    print(f"   평균 길이: {avg_length:.0f}자")  # 평균 길이 (소수점 없이)
    print(f"   최소 길이: {min_length}자")  # 최소 길이
    print(f"   최대 길이: {max_length}자")  # 최대 길이
    print(f"\n   📁 파일별 청크 수:")  # 파일별 통계 제목
    for source, count in sources.items():  # 각 파일을 순회
        print(f"      {source}: {count}개")  # 파일명과 청크 수 출력

def chunk_all_documents_semantic(documents, embedding_function):
    """
    Semantic Chunking: 의미 단위로 문서를 분할하는 함수

    기존 RecursiveCharacterTextSplitter는 글자 수로 자르지만,
    Semantic Chunker는 문장마다 임베딩을 구한 뒤
    의미가 크게 바뀌는 지점에서 자른다.

    예: "SVM 설명..." → "데이터 전처리..." 에서 의미가 바뀌면 여기서 분할

    파라미터:
        documents (list[dict]): parser.py에서 추출한 전체 페이지 리스트
            - "text": 페이지 텍스트
            - "metadata": 파일명, 페이지 번호 등
        embedding_function: embedder.py에서 생성한 임베딩 함수 객체
            - 문장 간 유사도를 계산하는 데 사용됨

    반환값:
        list[dict]: 의미 단위로 분할된 청크 리스트
            - "text": 청크 텍스트
            - "metadata": 원본 메타데이터 + 청크 번호
    """
    from langchain_experimental.text_splitter import SemanticChunker

    # SemanticChunker 생성
    # breakpoint_threshold_type="percentile": 유사도 변화가 상위 N%에 해당하면 분할
    # → 의미가 급격히 바뀌는 지점에서만 자름
    semantic_chunker = SemanticChunker(
        embedding_function,
        breakpoint_threshold_type="percentile",
    )

    all_chunks = []  # 전체 청크를 저장할 리스트
    skipped = 0  # 텍스트가 너무 짧아 건너뛴 페이지 수

    for document in documents:  # 각 문서(페이지)를 순회
        text = document["text"]
        metadata = document["metadata"]

        # 텍스트가 너무 짧으면 분할할 의미가 없으므로 건너뜀
        if not text or len(text.strip()) < 50:
            skipped += 1
            continue

        try:
            # Semantic Chunker로 텍스트 분할
            # 내부적으로 문장마다 임베딩 API를 호출하여 유사도를 계산함
            text_chunks = semantic_chunker.split_text(text)

            for i, chunk_text in enumerate(text_chunks):  # 각 청크를 순회
                # 빈 청크 방지
                if not chunk_text.strip():
                    continue

                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "total_pages": metadata["total_pages"],
                        "chunk_index": i,
                        "total_chunks_in_page": len(text_chunks),
                        "chunking_method": "semantic",  # 어떤 방식으로 청킹했는지 기록
                    }
                }
                all_chunks.append(chunk)

        except Exception as e:
            # Semantic Chunking 실패 시 해당 페이지는 통째로 하나의 청크로 저장
            print(f"    ⚠️  {metadata['source']} p.{metadata['page']}: Semantic Chunking 실패 ({e})")
            all_chunks.append({
                "text": text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "total_pages": metadata["total_pages"],
                    "chunk_index": 0,
                    "total_chunks_in_page": 1,
                    "chunking_method": "fallback",  # 실패 시 폴백
                }
            })

    print(f"📦 Semantic 청킹 완료: {len(documents)}개 페이지 → {len(all_chunks)}개 청크")
    if skipped > 0:
        print(f"   ⚠️  {skipped}개 페이지 건너뜀 (텍스트 50자 미만)")

    return all_chunks


# ============================================
# 직접 실행 시 테스트 코드
# 실행 방법: python modules/chunker.py
# ============================================
if __name__ == "__main__":
    from parser import parse_all_pdfs  # 같은 모듈 폴더의 parser 가져오기

    PDF_DIR = "data/pdfs"  # PDF 폴더 경로

    # 1단계: PDF 파싱
    print("=" * 50)  # 구분선
    print("1단계: PDF 파싱")  # 단계 표시
    print("=" * 50)  # 구분선
    documents = parse_all_pdfs(PDF_DIR)  # 모든 PDF 파싱

    if not documents:  # 파싱 결과가 없으면
        print("파싱된 문서가 없습니다. data/pdfs 폴더에 PDF를 넣어주세요.")  # 안내 메시지
        exit()  # 프로그램 종료

    # 2단계: 청킹 (여러 크기로 비교 실험)
    print("\n" + "=" * 50)  # 구분선
    print("2단계: 청킹 비교 실험")  # 단계 표시
    print("=" * 50)  # 구분선

    # 6일차 실험을 위한 3가지 청크 크기 비교
    for size in [256, 512, 1024]:  # 3가지 크기로 실험
        print(f"\n--- chunk_size={size} ---")  # 현재 크기 표시
        chunks = chunk_all_documents(documents, chunk_size=size, chunk_overlap=50)  # 청킹 실행
        print_chunk_stats(chunks)  # 통계 출력

        # 첫 번째 청크 미리보기
        if chunks:  # 청크가 있으면
            print(f"\n   첫 번째 청크 미리보기:")  # 미리보기 제목
            print(f"   {chunks[0]['text'][:150]}...")  # 텍스트 앞 150자만 출력