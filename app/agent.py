"""
VisionAgent — LangGraph Agent Modülü.
Görsel analiz → RAG → Zenginleştirilmiş Yanıt pipeline'ını
LangGraph state graph olarak orkestre eder.

Akış:
  [Görsel Alındı] → analyze_node → retrieve_node → synthesize_node → [Son Yanıt]
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any, Optional, List, Dict
from app.vision import analyze_image
from app.rag.retriever import retrieve
from app.rag.indexer import index_document
from app.tracking import log_run
import time


# ── State Tanımı ──────────────────────────────────────────────
# LangGraph her adımda bu state objesini taşır ve günceller.

class AgentState(TypedDict):
    """Agent pipeline'ının tüm adımlarda taşıdığı durum."""
    image_bytes: bytes
    question: str
    vision_result: str
    context_docs: List[Dict]
    final_answer: str
    duration: float
    error: Optional[str]


# ── Graph Düğümleri (Nodes) ──────────────────────────────────

def analyze_node(state: AgentState) -> Dict[str, Any]:
    """
    Düğüm 1: Qwen2-VL ile görseli analiz et.
    Kullanıcının yüklediği görseli ve soruyu VLM'e gönderir.
    """
    try:
        result = analyze_image(state["image_bytes"], state["question"])
        return {"vision_result": result}
    except Exception as e:
        return {"vision_result": "", "error": f"VLM hatası: {e}"}


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Düğüm 2: Qdrant'tan ilgili geçmiş kayıtları getir (RAG).
    VLM çıktısını sorgu olarak kullanır.
    """
    if state.get("error"):
        return {"context_docs": []}

    try:
        docs = retrieve(state["vision_result"], top_k=3)
        return {"context_docs": docs}
    except Exception as e:
        print(f"[Agent] RAG retrieval hatası: {e}")
        return {"context_docs": []}


def synthesize_node(state: AgentState) -> Dict[str, Any]:
    """
    Düğüm 3: VLM çıktısı + RAG bağlamını birleştirerek
    zenginleştirilmiş final yanıtı oluştur.
    Ayrıca bu sonucu Qdrant'a kaydet (gelecek sorgular için).
    """
    vision_result = state["vision_result"]
    context_docs = state.get("context_docs", [])

    # Eğer hata varsa direkt VLM sonucunu dön
    if state.get("error"):
        return {"final_answer": state.get("error", "Bilinmeyen hata")}

    # Bağlam metinlerini birleştir
    context_text = "\n".join([d["text"] for d in context_docs if d.get("text")])

    # Zenginleştirilmiş yanıt
    if context_text:
        final = (
            f"{vision_result}\n\n"
            f"📚 Geçmiş Analizlerden Bağlam:\n{context_text}"
        )
    else:
        final = vision_result

    # Bu analizi de Qdrant'a kaydet (gelecek RAG sorguları için)
    try:
        index_document(
            text=vision_result,
            metadata={"question": state["question"], "timestamp": time.time()},
        )
    except Exception as e:
        print(f"[Agent] İndeksleme hatası (önemli değil): {e}")

    return {"final_answer": final}


# ── Graph'ı Kur ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """LangGraph agent graph'ını oluşturur."""
    graph = StateGraph(AgentState)

    # Düğümleri ekle
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    # Akışı tanımla: START → analyze → retrieve → synthesize → END
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph


# Derlenmiş graph (uygulama boyunca tek instance)
_compiled_graph = None


def get_graph():
    """Derlenmiş graph'ı singleton olarak döndürür."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


# ── Ana Çalıştırma Fonksiyonu ────────────────────────────────

async def run_agent(image_bytes: bytes, question: str) -> dict:
    """
    Agent pipeline'ını başlatır ve sonucu döndürür.
    
    Args:
        image_bytes: Kullanıcının yüklediği görselin byte dizisi
        question: Kullanıcının sorusu
    
    Returns:
        Analiz sonucu (vision_result, context, final_answer, süre vb.)
    """
    start = time.time()

    # Başlangıç state'ini oluştur
    initial_state: AgentState = {
        "image_bytes": image_bytes,
        "question": question,
        "vision_result": "",
        "context_docs": [],
        "final_answer": "",
        "duration": 0.0,
        "error": None,
    }

    # LangGraph pipeline'ını çalıştır
    graph = get_graph()
    final_state = graph.invoke(initial_state)

    duration = round(time.time() - start, 2)

    # MLflow'a logla
    try:
        log_run(
            question=question,
            answer=final_state.get("final_answer", ""),
            duration=duration,
            extra_metrics={"context_count": len(final_state.get("context_docs", []))},
        )
    except Exception as e:
        print(f"[Agent] MLflow log hatası (önemli değil): {e}")

    return {
        "question": question,
        "vision_analysis": final_state.get("vision_result", ""),
        "context_used": final_state.get("context_docs", []),
        "final_answer": final_state.get("final_answer", ""),
        "duration_seconds": duration,
        "error": final_state.get("error"),
    }