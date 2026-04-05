import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def load_documents(kb_dir: Path) -> List[Dict[str, Any]]:
    documents = []

    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    txt_files = sorted(kb_dir.glob("*.txt"))
    md_files = sorted(kb_dir.glob("*.md"))
    files = txt_files + md_files

    if not files:
        raise ValueError(f"No .txt or .md files found in {kb_dir}")

    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            {
                "doc_id": file_path.stem,
                "title": file_path.stem.replace("_", " ").replace("-", " ").title(),
                "source": str(file_path.as_posix()),
                "text": text,
            }
        )

    return documents


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    global_chunk_id = 0

    for doc in documents:
        doc_text = doc["text"]
        split_texts = splitter.split_text(doc_text)

        current_pos = 0

        for local_chunk_index, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            found_pos = doc_text.find(chunk_text[:100], current_pos)
            if found_pos == -1:
                found_pos = current_pos

            start_char = found_pos
            end_char = start_char + len(chunk_text)
            current_pos = end_char

            chunks.append(
                {
                    "chunk_id": global_chunk_id,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "chunk_index": local_chunk_index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": chunk_text,
                }
            )
            global_chunk_id += 1

    return chunks


def build_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def save_chunks(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def save_manifest(
    output_path: Path,
    model_name: str,
    embedding_dim: int,
    documents_count: int,
    chunks_count: int,
    kb_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    build_seconds: float,
) -> None:
    manifest = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "documents_count": documents_count,
        "chunks_count": chunks_count,
        "kb_dir": kb_dir,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "build_time_seconds": round(build_seconds, 2),
        "vector_db": "FAISS",
    }

    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search_index(
    query: str,
    model_name: str,
    device: str,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    model = SentenceTransformer(model_name, device=device)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text_preview": chunk["text"][:300].replace("\n", " "),
            }
        )

    return results


def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for knowledge base")
    parser.add_argument("--kb-dir", type=str, default="knowledge_base", help="Path to knowledge base directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/index", help="Directory to save index and metadata")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3", help="SentenceTransformer model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--query", action="append", default=[], help="Optional test query, can be used multiple times")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K results for search")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = output_dir / "faiss.index"
    chunks_path = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "manifest.json"

    start_time = time.time()

    print(f"[1/5] Loading documents from: {kb_dir}")
    documents = load_documents(kb_dir)
    print(f"Loaded documents: {len(documents)}")

    print("[2/5] Splitting documents into chunks...")
    chunks = split_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Created chunks: {len(chunks)}")

    print(f"[3/5] Building embeddings with model: {args.model_name}")
    embeddings = build_embeddings(
        chunks=chunks,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    print("[4/5] Creating FAISS index...")
    index = create_faiss_index(embeddings)
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS index saved to: {faiss_path}")

    print("[5/5] Saving metadata...")
    save_chunks(chunks, chunks_path)
    save_manifest(
        output_path=manifest_path,
        model_name=args.model_name,
        embedding_dim=embeddings.shape[1],
        documents_count=len(documents),
        chunks_count=len(chunks),
        kb_dir=str(kb_dir.as_posix()),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        build_seconds=time.time() - start_time,
    )
    print(f"Chunks saved to: {chunks_path}")
    print(f"Manifest saved to: {manifest_path}")

    total_time = time.time() - start_time
    print(f"\nDone. Total build time: {total_time:.2f} sec")

    if args.query:
        print("\n=== SEARCH TESTS ===")
        saved_index = faiss.read_index(str(faiss_path))
        saved_chunks = load_chunks(chunks_path)

        for q in args.query:
            print(f"\nQuery: {q}")
            results = search_index(
                query=q,
                model_name=args.model_name,
                device=args.device,
                index=saved_index,
                chunks=saved_chunks,
                top_k=args.top_k,
            )
            for i, result in enumerate(results, 1):
                print(f"\nResult #{i}")
                print(f"Score: {result['score']:.4f}")
                print(f"Document: {result['doc_id']}")
                print(f"Chunk index: {result['chunk_index']}")
                print(f"Source: {result['source']}")
                print(f"Preview: {result['text_preview']}")


if __name__ == "__main__":
    main()
