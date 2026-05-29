"""AI 个人助手 CLI —— 支持流式对话、文件总结、代码解释"""

import os
import json
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from dotenv import load_dotenv

load_dotenv()

console = Console()

CONFIG_PATH = Path.home() / ".ai-assistant" / "config.json"

class AIAssistant:
    """AI 个人助手核心类"""
    
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.history: list[dict] = []
        self.system_prompt = os.getenv("AI_ASSISTANT_SYSTEM_PROMPT", 
            "你是一个友好的AI助手。用中文回答，简洁清晰。")
    
    def chat(self, message: str, stream: bool = True) -> str:
        """发送消息并获取回复，支持流式输出"""
        self.history.append({"role": "user", "content": message})
        
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        
        if stream:
            return self._chat_stream(messages)
        else:
            return self._chat_sync(messages)
    
    def _chat_stream(self, messages: list[dict]) -> str:
        """流式对话，实时打印"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        
        full_reply = ""
        console.print()
        with Live(Panel("思考中...", title="🤖 AI 助手"), refresh_per_second=10, vertical_overflow="visible") as live:
            first_chunk = True
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_chunk:
                        full_reply = ""
                        first_chunk = False
                    full_reply += chunk.choices[0].delta.content
                    live.update(Panel(Markdown(full_reply), title="🤖 AI 助手"))
        
        self.history.append({"role": "assistant", "content": full_reply})
        return full_reply
    
    def _chat_sync(self, messages: list[dict]) -> str:
        """同步对话"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        console.print(Panel(Markdown(reply), title="🤖 AI 助手"))
        return reply
    
    def summarize_file(self, filepath: str) -> str:
        """总结文件内容"""
        content = self._read_file(filepath)
        prompt = f"请总结以下文件的内容，用中文回答，简洁明了。文件名：{filepath}\n\n```\n{content}\n```"
        return self.chat(prompt)
    
    def explain_code(self, filepath: str) -> str:
        """解释代码逻辑"""
        code = self._read_file(filepath)
        ext = Path(filepath).suffix
        prompt = f"请解释以下{ext}代码的功能和逻辑，用中文回答：\n\n```{ext}\n{code}\n```"
        return self.chat(prompt)
    
    def _read_file(self, filepath: str) -> str:
        """读取文件"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        return path.read_text(encoding="utf-8")
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.history = []
        console.print("[green]✅ 对话历史已清空[/green]")
    
    def show_history(self) -> None:
        """显示对话历史摘要"""
        if not self.history:
            console.print("[yellow]📭 暂无对话历史[/yellow]")
            return
        
        for i in range(0, len(self.history), 2):
            user_msg = self.history[i]["content"][:60]
            ai_msg = self.history[i+1]["content"][:60] if i+1 < len(self.history) else "..."
            console.print(f"[bold]第{i//2+1}轮[/bold]")
            console.print(f"  🧑 你: {user_msg}")
            console.print(f"  🤖 AI: {ai_msg}")
            console.print()
    
    def get_history_length(self) -> int:
        """获取对话轮数"""
        return len(self.history) // 2
    
    def save_config(self) -> None:
        """保存配置到本地"""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "model": self.model,
            "base_url": self.base_url,
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]✅ 配置已保存到 {CONFIG_PATH}[/green]")
    
    @classmethod
    def load_config(cls) -> dict:
        """加载本地配置"""
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return {}

def main():
    from typer import Typer
    
    app = Typer(help="🤖 AI 个人助手 CLI")
    assistant = None
    
    def get_assistant() -> AIAssistant:
        nonlocal assistant
        if assistant is None:
            config = AIAssistant.load_config()
            assistant = AIAssistant(
                model=config.get("model"),
                base_url=config.get("base_url"),
            )
        return assistant
    
    @app.command()
    def chat(message: str):
        """发送消息并获取流式回复"""
        get_assistant().chat(message)
    
    @app.command()
    def summarize(filepath: str):
        """总结文件内容"""
        get_assistant().summarize_file(filepath)
    
    @app.command()
    def explain(filepath: str):
        """解释代码逻辑"""
        get_assistant().explain_code(filepath)
    
    @app.command()
    def history():
        """查看对话历史"""
        get_assistant().show_history()
    
    @app.command()
    def clear():
        """清空对话历史"""
        get_assistant().clear_history()
    
    @app.command()
    def config(model: str | None = None):
        """查看或设置模型配置"""
        a = get_assistant()
        if model:
            a.model = model
            a.save_config()
            console.print(f"[green]✅ 模型已切换为: {model}[/green]")
        else:
            console.print(f"[bold]当前模型:[/bold] {a.model}")
            console.print(f"[bold]API 地址:[/bold] {a.base_url}")
    
    @app.command()
    def interactive():
        """进入交互式对话模式"""
        from rich.prompt import Prompt
        
        a = get_assistant()
        console.print(Panel("🤖 AI 助手已就绪 | 输入 'quit' 退出 | 'clear' 清空 | 'history' 查看历史", 
                           title="交互模式"))
        
        while True:
            try:
                user_input = Prompt.ask("\n🧑 你")
                if user_input.lower() == "quit":
                    console.print("[green]👋 再见！[/green]")
                    break
                elif user_input.lower() == "clear":
                    a.clear_history()
                    continue
                elif user_input.lower() == "history":
                    a.show_history()
                    continue
                a.chat(user_input)
            except KeyboardInterrupt:
                console.print("\n[green]👋 再见！[/green]")
                break
    
    app()

if __name__ == "__main__":
    main()
