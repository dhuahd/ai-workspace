"""RAG 系统测试"""

import pytest
from unittest.mock import Mock, patch
from src.main import SimpleVectorStore, DocumentLoader, RAGSystem


class TestSimpleVectorStore:
    """测试向量存储"""
    
    def test_add_and_search(self):
        store = SimpleVectorStore()
        store.add("Python 是一种编程语言", [1.0, 0.0, 0.0])
        store.add("机器学习是 AI 的子集", [0.0, 1.0, 0.0])
        store.add("深度学习使用神经网络", [0.0, 0.0, 1.0])
        
        results = store.search([1.0, 0.1, 0.0], top_k=2)
        
        assert len(results) == 2
        assert "Python" in results[0]["text"]
        assert results[0]["score"] > results[1]["score"]
    
    def test_empty_store(self):
        store = SimpleVectorStore()
        results = store.search([1.0, 0.0])
        assert results == []
    
    def test_cosine_similarity(self):
        store = SimpleVectorStore()
        sim = store._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 0.001
        
        sim = store._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim - 0.0) < 0.001
    
    def test_len(self):
        store = SimpleVectorStore()
        assert len(store) == 0
        store.add("test", [0.1, 0.2])
        assert len(store) == 1


class TestDocumentLoader:
    """测试文档加载"""
    
    def test_chunk_text(self):
        text = "A" * 1000
        chunks = DocumentLoader.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 3  # 0-500, 450-950, 900-1000
        assert len(chunks[0]) == 500
    
    def test_chunk_short_text(self):
        text = "短文本"
        chunks = DocumentLoader.chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "短文本"


class TestRAGSystem:
    """测试 RAG 系统"""
    
    @patch("src.main.OpenAI")
    def test_search(self, mock_openai_cls):
        mock_client = Mock()
        mock_embed_response = Mock()
        mock_embed_response.data = [Mock()]
        mock_embed_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_embed_response
        mock_openai_cls.return_value = mock_client
        
        rag = RAGSystem(api_key="test-key")
        rag.client = mock_client
        rag.vector_store.add("测试文档", [0.1, 0.2, 0.3], {"source": "test.txt"})
        
        results = rag.search("测试")
        assert len(results) == 1
        assert results[0]["text"] == "测试文档"
    
    @patch("src.main.OpenAI")
    def test_ask_no_results(self, mock_openai_cls):
        mock_client = Mock()
        mock_embed_response = Mock()
        mock_embed_response.data = [Mock()]
        mock_embed_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_embed_response
        mock_openai_cls.return_value = mock_client
        
        rag = RAGSystem(api_key="test-key")
        rag.client = mock_client
        
        result = rag.ask("未知问题", stream=False)
        assert "没有找到" in result
    
    @patch("src.main.OpenAI")
    def test_get_stats(self, mock_openai_cls):
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        rag = RAGSystem(api_key="test-key")
        rag.client = mock_client
        stats = rag.get_stats()
        
        assert "total_chunks" in stats
        assert "model" in stats
        assert stats["total_chunks"] == 0
