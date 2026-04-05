from app.rag import RAGService


def main() -> None:
    rag = RAGService()
    print("RAG CLI started. Type 'exit' to quit.\n")

    while True:
        query = input(">>> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        result = rag.ask(query)
        print("\nANSWER:\n")
        print(result["answer"])
        print("\nRETRIEVED CHUNKS:\n")
        for chunk in result["retrieved_chunks"]:
            print(
                f"- score={chunk['score']} "
                f"doc_id={chunk['doc_id']} "
                f"chunk_index={chunk['chunk_index']}"
            )
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
