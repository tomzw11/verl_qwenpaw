
#!/usr/bin/env python3
"""
QwenPaw 客户端 - 简化版

设计原则：
1. QwenPaw 只作为 UI - 负责获取用户输入
2. 真正的 Rollout 由 verl Gateway + vLLM 处理
3. 只提供纯文本获取接口，不处理 log_probs
"""

import asyncio
import json
import time
import random
from typing import Any

class QwenPawClient:
    """
    QwenPaw 客户端 - 只获取纯文本
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:52143",
        agent_id: str = "default",
        user_id: str = "default",
        channel: str = "console",
        stream: bool = True,
        max_retries: int = 3
    ):
        self.base_url = base_url
        self.agent_id = agent_id
        self.user_id = user_id
        self.channel = channel
        self.stream = stream
        self.old_chat_url = f"{base_url}/api/console/chat"
        self.new_chat_url = f"{base_url}/api/agents/{agent_id}/chats"
        self.max_retries = max_retries
        self._lock = asyncio.Lock()
    
    def _build_old_payload(self, text: str, session_id: str | None = None) -&gt; dict:
        if session_id is None:
            session_id = f"verl-{int(time.time()*1000)}-{random.randint(1000,9999)}"
        return {
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "status": "created"
                        }
                    ]
                }
            ],
            "session_id": session_id,
            "user_id": self.user_id,
            "channel": self.channel,
            "stream": self.stream
        }
    
    async def chat(self, text: str, session_id: str | None = None) -&gt; str:
        """
        从 QwenPaw 获取纯文本响应
        QwenPaw 只作为 UI，真正的 Rollout 由 verl Gateway + vLLM 处理
        """
        async with self._lock:
            return await self._chat_with_retry(text, session_id)
    
    async def _chat_with_retry(self, text: str, session_id: str | None = None) -&gt; str:
        import httpx
        
        for attempt in range(self.max_retries):
            try:
                return await self._chat_once(text, session_id)
            except Exception as e:
                if attempt &lt; self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[QwenPaw] 请求失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                    print(f"[QwenPaw] {wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"QwenPaw 请求失败，已重试 {self.max_retries} 次: {e}")
    
    async def _chat_once(self, text: str, session_id: str | None = None) -&gt; str:
        import httpx
        
        payload = self._build_old_payload(text, session_id)
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Id": self.agent_id
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            if self.stream:
                response_text = ""
                final_response = None
                found_final_content = False
                
                async with client.stream("POST", self.old_chat_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                obj_type = data.get("object", "")
                                status = data.get("status", "")
                                
                                if obj_type == "response" and status == "completed":
                                    final_response = data
                                
                                if obj_type == "content":
                                    text_chunk = data.get("text", "")
                                    if text_chunk and status != "completed":
                                        response_text += text_chunk
                                    elif text_chunk and status == "completed":
                                        response_text = text_chunk
                                        found_final_content = True
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                print(f"[QwenPaw] 解析响应行失败: {e}")
                
                if final_response:
                    output = final_response.get("output")
                    if output:
                        return self._extract_text_from_output(output)
                
                return response_text
            else:
                response = await client.post(self.old_chat_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return self._extract_text_from_output(data.get("output", data))
    
    def _extract_text_from_output(self, output) -&gt; str:
        if output is None:
            return ""
        
        if isinstance(output, str):
            return output
        
        if isinstance(output, dict):
            if output.get("object") == "message" and "content" in output:
                content_list = output["content"]
                if isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "")
            if output.get("object") == "content" and "text" in output:
                return output["text"]
            if "text" in output:
                return str(output["text"])
            if "content" in output:
                return str(output["content"])
        
        if isinstance(output, list):
            for item in output:
                text = self._extract_text_from_output(item)
                if text:
                    return text
        
        return str(output)

# 简单测试函数
async def test_client():
    print("测试 QwenPaw 客户端")
    client = QwenPawClient()
    
    try:
        response = await client.chat("Hello, say something")
        print(f"响应: {response}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_client())

