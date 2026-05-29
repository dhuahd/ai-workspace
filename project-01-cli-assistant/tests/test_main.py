"""AI 个人助手 CLI 测试"""

import pytest
from unittest.mock import Mock, patch
from src.main import AIAssistant

class TestAIAssistant:
    """测试 AI 助手核心功能"""
    
    def _mock_chat_response(self, mock_openai_cls, content, stream=False):
        """Helper: 创建 mock OpenAI 客户端"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client
        return mock_client
    
    @patch("src.main.OpenAI")
    def test_chat_sync(self, mock_openai_cls):
        """测试同步对话"""
        mock_client = self._mock_chat_response(mock_openai_cls, "你好！")
        
        assistant = AIAssistant(api_key="test-key")
        assistant.client = mock_client
        
        result = assistant.chat("你好", stream=False)
        assert result == "你好！"
        assert assistant.get_history_length() == 1
    
    @patch("src.main.OpenAI")
    def test_chat_stream(self, mock_openai_cls):
        """测试流式对话"""
        mock_client = Mock()
        chunk1 = Mock()
        chunk1.choices = [Mock()]
        chunk1.choices[0].delta.content = "你"
        chunk2 = Mock()
        chunk2.choices = [Mock()]
        chunk2.choices[0].delta.content = "好"
        chunk3 = Mock()
        chunk3.choices = [Mock()]
        chunk3.choices[0].delta.content = None
        
        mock_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]
        mock_openai_cls.return_value = mock_client
        
        assistant = AIAssistant(api_key="test-key")
        assistant.client = mock_client
        
        result = assistant.chat("hello", stream=True)
        assert result == "你好"
    
    @patch("src.main.Path.exists", return_value=True)
    @patch("src.main.Path.read_text", return_value="print('hello')")
    @patch("src.main.OpenAI")
    def test_explain_code(self, mock_openai_cls, mock_read, mock_exists):
        """测试代码解释"""
        mock_client = self._mock_chat_response(mock_openai_cls, "打印 hello")
        
        assistant = AIAssistant(api_key="test-key")
        assistant.client = mock_client
        
        result = assistant.chat("explain test.py", stream=False)
        assert "打印" in result
    
    @patch("src.main.OpenAI")
    def test_clear_history(self, mock_openai_cls):
        """测试清空历史"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        assistant = AIAssistant(api_key="test-key")
        assistant.client = mock_client
        assistant.history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]
        assistant.clear_history()
        assert assistant.get_history_length() == 0
    
    @patch("src.main.OpenAI")
    def test_get_history_length(self, mock_openai_cls):
        """测试对话轮数统计"""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        
        assistant = AIAssistant(api_key="test-key")
        assistant.client = mock_client
        assistant.history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        assert assistant.get_history_length() == 2
    
    def test_read_file_not_found(self):
        """测试文件不存在"""
        assistant = AIAssistant(api_key="test-key")
        with pytest.raises(FileNotFoundError):
            assistant._read_file("nonexistent.txt")
