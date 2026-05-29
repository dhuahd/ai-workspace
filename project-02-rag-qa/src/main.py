"""RAG 知识库问答系统 —— 支持文档索引、语义搜索、智能问答"""

import os
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class SimpleVectorStore:
    """简易向量存储（无需外部依赖，基于内存）"""
    
    def __init__(self):
        self.documents: list[dict] = []  # [{text, embedding, metadata}]
    
    def add(self, text: str, embedding: list[float], metadata: dict | None = None) -> None:
        """添加文档"""
        self.documents.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        })
    
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """余弦相似度搜索"""
        if not self.documents:
            return []
        
        results = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            results.append({**doc, "score": similarity})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def __len__(self) -> int:
        return len(self.documents)


class DocumentLoader:
    """文档加载器：支持 txt, md, pdf (text extraction), py"""
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}
    
    @staticmethod
    def load_file(filepath: str) -> list[dict]:
        """加载单个文件，返回 [{text, metadata}]"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        if path.suffix.lower() not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        content = path.read_text(encoding="utf-8")
        return [{"text": content, "metadata": {"source": str(path), "filename": path.name}}]
    
    @staticmethod
    def load_directory(directory: str, recursive: bool = True) -> list[dict]:
        """加载目录下所有支持的文件"""
        path = Path(directory)
        docs = []
        pattern = "**/*" if recursive else "*"
        
        for filepath in path.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                try:
                    docs.extend(DocumentLoader.load_file(str(filepath)))
                except Exception as e:
                    print(f"警告: 无法加载 {filepath}: {e}")
        
        return docs
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """将长文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


class RAGSystem:
    """RAG 问答系统核心"""
    
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.vector_store = SimpleVectorStore()
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    
    def get_embedding(self, text: str) -> list[float]:
        """获取文本的向量表示"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    
    def index_file(self, filepath: str) -> int:
        """索引单个文件"""
        docs = DocumentLoader.load_file(filepath)
        count = 0
        
        for doc in docs:
            chunks = DocumentLoader.chunk_text(doc["text"], self.chunk_size, self.chunk_overlap)
            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                metadata = {**doc["metadata"], "chunk_index": i, "total_chunks": len(chunks)}
                self.vector_store.add(chunk, embedding, metadata)
                count += 1
        
        return count
    
    def index_directory(self, directory: str) -> int:
        """索引整个目录"""
        docs = DocumentLoader.load_directory(directory)
        count = 0
        
        for doc in docs:
            chunks = DocumentLoader.chunk_text(doc["text"], self.chunk_size, self.chunk_overlap)
            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                metadata = {**doc["metadata"], "chunk_index": i, "total_chunks": len(chunks)}
                self.vector_store.add(chunk, embedding, metadata)
                count += 1
        
        return count
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义搜索"""
        query_embedding = self.get_embedding(query)
        return self.vector_store.search(query_embedding, top_k)
    
    def ask(self, question: str, top_k: int = 5, stream: bool = True) -> str:
        """基于知识库问答"""
        results = self.search(question, top_k)
        
        if not results:
            return "知识库中没有找到相关内容。"
        
        # 拼接上下文
        context_parts = []
        for i, r in enumerate(results):
            source = r["metadata"].get("source", "unknown")
            context_parts.append(f"[文档{i+1} 来源: {source}]\n{r['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""请基于以下知识库内容回答用户的问题。如果知识库中没有相关信息，请如实说明。

知识库内容：
{context}

用户问题：{question}

请用中文回答，引用来源时注明文档编号。"""
        
        if stream:
            return self._stream_chat(prompt)
        else:
            return self._sync_chat(prompt)
    
    def _sync_chat(self, prompt: str) -> str:
        """同步聊天"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个基于知识库的问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    def _stream_chat(self, prompt: str) -> str:
        """流式聊天"""
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.console import Console
        
        console = Console()
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个基于知识库的问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            stream=True,
        )
        
        full_reply = ""
        console.print()
        with Live(Panel("检索知识库中...", title="📚 RAG 问答"), refresh_per_second=10) as live:
            first_chunk = True
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_chunk:
                        full_reply = ""
                        first_chunk = False
                    full_reply += chunk.choices[0].delta.content
                    live.update(Panel(Markdown(full_reply), title="📚 RAG 问答"))
        
        return full_reply
    
    def get_stats(self) -> dict:
        """获取知识库统计"""
        return {
            "total_chunks": len(self.vector_store),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model": self.model,
        }


def main():
    from typer import Typer
    from rich.console import Console
    from rich.table import Table
    
    app = Typer(help="📚 RAG 知识库问答系统")
    console = Console()
    rag = RAGSystem()
    
    @app.command()
    def index(path: str):
        """索引文件或目录"""
        p = Path(path)
        if p.is_file():
            count = rag.index_file(str(p))
        elif p.is_dir():
            count = rag.index_directory(str(p))
        else:
            console.print(f"[red]路径不存在: {path}[/red]")
            return
        console.print(f"[green]✅ 已索引 {count} 个文本块[/green]")
    
    @app.command()
    def ask(question: str):
        """基于知识库提问"""
        answer = rag.ask(question)
        if answer != "知识库中没有找到相关内容。":
            console.print(answer)
        else:
            console.print("[yellow]📭 知识库中没有找到相关内容[/yellow]")
    
    @app.command()
    def search(query: str):
        """搜索知识库"""
        results = rag.search(query)
        if not results:
            console.print("[yellow]📭 未找到相关结果[/yellow]")
            return
        
        table = Table(title=f"搜索结果: {query}")
        table.add_column("#", style="cyan")
        table.add_column("来源", style="green")
        table.add_column("内容预览", style="white")
        table.add_column("相关度", style="magenta")
        
        for i, r in enumerate(results):
            table.add_row(
                str(i+1),
                r["metadata"].get("filename", "unknown"),
                r["text"][:80] + "...",
                f"{r['score']:.4f}"
            )
        
        console.print(table)
    
    @app.command()
    def stats():
        """查看知识库统计"""
        s = rag.get_stats()
        console.print(f"[bold]知识库统计[/bold]")
        console.print(f"  总块数: {s['total_chunks']}")
        console.print(f"  分块大小: {s['chunk_size']}")
        console.print(f"  重叠大小: {s['chunk_overlap']}")
        console.print(f"  模型: {s['model']}")
    
    app()

if __name__ == "__main__":
    main()
