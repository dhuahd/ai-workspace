"""多 Agent 协作工作流引擎 —— 规划-执行-审查 三阶段协作"""

import os
import json
from typing import Any, Callable
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()
console = Console()


@dataclass
class Task:
    """任务数据结构"""
    id: str
    description: str
    status: str = "pending"  # pending, running, completed, failed
    result: str = ""
    error: str = ""


@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    role: str
    system_prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7


class BaseAgent:
    """Agent 基类"""
    
    def __init__(self, config: AgentConfig, client: OpenAI):
        self.config = config
        self.client = client
        self.history: list[dict] = []
    
    def think(self, prompt: str) -> str:
        """调用 LLM 获取回复"""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            *self.history,
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        
        reply = response.choices[0].message.content
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": reply})
        return reply
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []


class PlannerAgent(BaseAgent):
    """规划 Agent：将复杂任务拆分为子任务"""
    
    def __init__(self, client: OpenAI):
        config = AgentConfig(
            name="Planner",
            role="任务规划器",
            system_prompt="""你是一个专业的任务规划器。收到用户请求后，你需要：
1. 分析任务的复杂度和目标
2. 将任务拆分为清晰、可执行的子任务
3. 输出 JSON 格式的子任务列表

输出格式：
```json
{
    "summary": "任务概述",
    "tasks": [
        {"id": "task-1", "description": "子任务描述"},
        {"id": "task-2", "description": "子任务描述"}
    ]
}
```

注意：
- 每个子任务要具体、可独立执行
- 子任务数量控制在 3-7 个
- 子任务之间要有逻辑顺序
"""
        )
        super().__init__(config, client)
    
    def plan(self, request: str) -> dict:
        """规划任务，返回结构化的任务列表"""
        prompt = f"请为以下请求制定执行计划：\n\n{request}"
        response = self.think(prompt)
        
        # 提取 JSON
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            plan = json.loads(json_str.strip())
            return plan
        except (json.JSONDecodeError, IndexError):
            # 降级：手动解析
            return {
                "summary": request[:100],
                "tasks": [{"id": f"task-{i+1}", "description": line.strip("- ")}
                         for i, line in enumerate(response.split("\n")) if line.strip().startswith("-")]
            } or {"summary": request[:100], "tasks": [{"id": "task-1", "description": request}]}


class ExecutorAgent(BaseAgent):
    """执行 Agent：逐个执行子任务"""
    
    def __init__(self, client: OpenAI):
        config = AgentConfig(
            name="Executor",
            role="任务执行器",
            system_prompt="""你是一个专业的任务执行器。你会收到一个子任务描述，请：
1. 分析如何完成这个子任务
2. 给出具体的执行方案和结果
3. 如果子任务涉及代码，请给出代码示例

请用中文回答，结果要具体实用。"""
        )
        super().__init__(config, client)
    
    def execute(self, task_description: str, context: str = "") -> str:
        """执行单个子任务"""
        prompt = f"当前上下文：{context}\n\n请执行以下子任务：{task_description}"
        return self.think(prompt)


class ReviewerAgent(BaseAgent):
    """审查 Agent：检查执行结果，决定是否重试"""
    
    def __init__(self, client: OpenAI):
        config = AgentConfig(
            name="Reviewer",
            role="质量审查器",
            system_prompt="""你是一个严格的质量审查器。你需要：
1. 检查执行结果是否满足任务要求
2. 判断结果质量：pass（通过）或 retry（需要重试）
3. 如果 retry，给出具体的改进建议

输出 JSON 格式：
```json
{
    "verdict": "pass",
    "feedback": "审查意见"
}
```
或
```json
{
    "verdict": "retry",
    "feedback": "具体改进建议"
}
```
"""
        )
        super().__init__(config, client)
    
    def review(self, task: Task, context: str = "") -> dict:
        """审查任务执行结果"""
        prompt = f"""上下文：{context}

任务：{task.description}
执行结果：{task.result}

请审查以上执行结果。"""
        
        response = self.think(prompt)
        
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            return {"verdict": "pass", "feedback": response[:200]}


class MultiAgentWorkflow:
    """多 Agent 协作工作流引擎"""
    
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        self.planner = PlannerAgent(self.client)
        self.executor = ExecutorAgent(self.client)
        self.reviewer = ReviewerAgent(self.client)
        
        self.max_retries = 2
        self.tasks: list[Task] = []
    
    def run(self, request: str, verbose: bool = True) -> list[Task]:
        """运行完整工作流：规划 → 执行 → 审查"""
        self.tasks = []
        
        # 阶段1: 规划
        if verbose:
            console.print(Panel(f"📋 规划中: {request[:80]}...", title="[bold blue]阶段1: 规划[/bold blue]"))
        
        plan = self.planner.plan(request)
        
        if verbose:
            console.print(f"  [green]✓ {plan.get('summary', '')}[/green]")
            for t in plan.get("tasks", []):
                console.print(f"    - {t['description'][:60]}")
        
        # 阶段2: 执行+审查
        context = ""
        for task_data in plan.get("tasks", []):
            task = Task(id=task_data["id"], description=task_data["description"])
            self.tasks.append(task)
            
            if verbose:
                console.print(Panel(f"执行: {task.description[:60]}...", 
                                   title=f"[bold yellow]阶段2: {task.id}[/bold yellow]"))
            
            # 执行（带重试）
            for attempt in range(self.max_retries + 1):
                task.status = "running"
                
                if attempt > 0 and verbose:
                    console.print(f"  [yellow]🔄 重试第 {attempt} 次...[/yellow]")
                    self.executor.clear_history()
                
                task.result = self.executor.execute(task.description, context)
                task.status = "completed"
                
                if verbose:
                    console.print(f"  [dim]{task.result[:100]}...[/dim]")
                
                # 审查
                if verbose:
                    console.print(f"  [bold magenta]审查中...[/bold magenta]")
                
                review = self.reviewer.review(task, context)
                
                if verbose:
                    verdict_icon = "✅" if review.get("verdict") == "pass" else "❌"
                    console.print(f"  {verdict_icon} {review.get('feedback', '')[:80]}")
                
                if review.get("verdict") == "pass":
                    break
                elif attempt < self.max_retries:
                    context += f"\n改进建议: {review.get('feedback', '')}"
                else:
                    task.status = "failed"
                    task.error = review.get("feedback", "已达最大重试次数")
        
        # 阶段3: 总结
        if verbose:
            self._print_summary()
        
        return self.tasks
    
    def _print_summary(self):
        """打印执行总结"""
        table = Table(title="[bold]📊 工作流执行总结[/bold]")
        table.add_column("任务ID", style="cyan")
        table.add_column("描述", style="white")
        table.add_column("状态", style="green")
        table.add_column("结果预览", style="dim")
        
        for task in self.tasks:
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(task.status, "❓")
            
            table.add_row(
                task.id,
                task.description[:40],
                f"{status_icon} {task.status}",
                task.result[:50] + "..." if task.result else "-"
            )
        
        console.print(table)
    
    def get_log(self) -> str:
        """获取执行日志"""
        log_parts = ["# 多 Agent 工作流执行日志\n"]
        for task in self.tasks:
            log_parts.append(f"\n## {task.id}: {task.description}")
            log_parts.append(f"状态: {task.status}")
            log_parts.append(f"\n结果:\n{task.result}")
            if task.error:
                log_parts.append(f"\n错误:\n{task.error}")
        return "\n".join(log_parts)


def main():
    from typer import Typer
    
    app = Typer(help="🤖 多 Agent 协作工作流引擎")
    
    workflow_manager: MultiAgentWorkflow | None = None
    
    def get_workflow() -> MultiAgentWorkflow:
        nonlocal workflow_manager
        if workflow_manager is None:
            workflow_manager = MultiAgentWorkflow()
        return workflow_manager
    
    @app.command()
    def run(request: str):
        """运行多 Agent 工作流"""
        wf = get_workflow()
        tasks = wf.run(request, verbose=True)
        
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        console.print(f"\n[bold]总计: {len(tasks)} 个任务 | ✅ {completed} 完成 | ❌ {failed} 失败[/bold]")
    
    @app.command()
    def log():
        """查看执行日志"""
        wf = get_workflow()
        if not wf.tasks:
            console.print("[yellow]📭 暂无执行记录[/yellow]")
        else:
            console.print(Markdown(wf.get_log()))
    
    app()

if __name__ == "__main__":
    main()
