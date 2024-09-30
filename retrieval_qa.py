from typing import Iterable, List

from langchain import hub
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def transform_to_docs(texts: Iterable[str]) -> Iterable[Document]:
    docs = [Document(page_content=text, metadata={"source": "local"}) for text in texts]
    return docs


def create_chunks(docs: Iterable[str], chunk_size: int = 200, chunk_overlap: int = 100) -> List[Document]:
    docs = transform_to_docs(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks


def load_embedding_model(model: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=model)


def init_gpt_model(model: str = 'gpt-3.5-turbo') -> ChatOpenAI:
    return ChatOpenAI(model_name=model, temperature=0.0)


class QASystem:
    def __init__(self) -> None:
        self.retriever = None
        self.qa_chain = None
        self.model = init_gpt_model()

    def init_retriever(self, documents: list[str], k: int = 3, persist_directory: str = 'chroma_db') -> None:
        chunks = create_chunks(documents)
        embedding_model = load_embedding_model(model='embedding_models/ruElectra-large')
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        vector_store.add_documents(chunks)
        vector_store.persist()
        vectorstore_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k = k
        ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever,
                                                           keyword_retriever],
                                               weights=[0.2, 0.8])
        self.retriever = ensemble_retriever

    def create_qa_chain(self) -> None:
        prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.model, retriever=self.retriever, chain_type_kwargs={"prompt": prompt}
        )

    def get_answer_by_context(self, question: str) -> str:
        if self.qa_chain and self.retriever:
            return self.qa_chain({"query": question})["result"]
        else:
            return "Не задан контекст"