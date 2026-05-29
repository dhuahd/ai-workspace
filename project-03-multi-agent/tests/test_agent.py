"""多 Agent 系统测试"""

import pytest
from unittest.mock import Mock, patch
from src.main import (
    Task, AgentConfig, BaseAgent, PlannerAgent, 
    ExecutorAgent, ReviewerAgent, MultiAgentWorkflow
)


class TestTask:
    """测试任务数据结构"""
    
    def test_create_task(self):
        task = Task(id="task-1", description="测试任务")
        assert task.id == "task-1"
        assert task.status == "pending"
        assert task.result == ""


class TestBaseAgent:
    """测试 Agent 基类"""
    
    def test_think(self):
        config = AgentConfig(name="Test", role="测试", system_prompt="你是测试助手")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "测试回复"
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = BaseAgent(config, mock_client)
        result = agent.think("你好")
        assert result == "测试回复"
    
    def test_clear_history(self):
        config = AgentConfig(name="Test", role="测试", system_prompt="test")
        mock_client = Mock()
        agent = BaseAgent(config, mock_client)
        agent.history = [{"role": "user", "content": "hi"}]
        agent.clear_history()
        assert agent.history == []


class TestPlannerAgent:
    """测试规划 Agent"""
    
    def test_plan(self):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''```json
{
    "summary": "创建网站",
    "tasks": [
        {"id": "task-1", "description": "设计页面结构"},
        {"id": "task-2", "description": "编写HTML代码"},
        {"id": "task-3", "description": "部署上线"}
    ]
}
```'''
        mock_client.chat.completions.create.return_value = mock_response
        
        planner = PlannerAgent(mock_client)
        plan = planner.plan("帮我建一个网站")
        
        assert "summary" in plan
        assert len(plan["tasks"]) == 3


class TestReviewerAgent:
    """测试审查 Agent"""
    
    def test_review_pass(self):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"verdict": "pass", "feedback": "很好"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        reviewer = ReviewerAgent(mock_client)
        task = Task(id="t1", description="写代码", result="代码已写好")
        result = reviewer.review(task)
        
        assert result["verdict"] == "pass"


class TestMultiAgentWorkflow:
    """测试多 Agent 工作流"""
    
    @patch("src.main.OpenAI")
    def test_run_workflow(self, mock_openai_cls):
        mock_client = Mock()
        
        # Mock 规划响应
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = '''```json
{"summary": "测试任务", "tasks": [{"id": "t1", "description": "完成任务"}]}
```'''
        
        # Mock 执行响应
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message.content = "任务完成"
        
        # Mock 审查响应
        mock_review_response = Mock()
        mock_review_response.choices = [Mock()]
        mock_review_response.choices[0].message.content = '{"verdict": "pass", "feedback": "通过"}'
        
        mock_client.chat.completions.create.side_effect = [
            mock_plan_response,
            mock_exec_response,
            mock_review_response,
        ]
        mock_openai_cls.return_value = mock_client
        
        wf = MultiAgentWorkflow(api_key="test-key")
        tasks = wf.run("完成一个测试任务", verbose=False)
        
        assert len(tasks) == 1
        assert tasks[0].status == "completed"
    
    def test_get_log_empty(self):
        wf = MultiAgentWorkflow(api_key="test-key")
        log = wf.get_log()
        assert "执行日志" in log
        assert len(wf.tasks) == 0
