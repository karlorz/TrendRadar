# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import os
from typing import Any, Dict, List, Optional

from litellm import completion


class AIClient:
    """统一的 AI 客户端（基于 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（格式: provider/model_name）
                - API_KEY: API 密钥
                - API_BASE: API 基础 URL（可选）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
                - FALLBACK_MODELS: 备用模型列表（可选）
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        self.api_key = config.get("API_KEY") or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        调用 AI 模型进行对话

        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: 额外参数，会覆盖默认配置

        Returns:
            str: AI 响应内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        # 处理 Gemini 兼容端点的特殊配置
        model, api_base = self._process_gemini_config(self.model, self.api_base)

        # 构建请求参数
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }

        # 添加 API Key
        if self.api_key:
            params["api_key"] = self.api_key

        # 添加 API Base（如果配置了）
        if api_base:
            params["api_base"] = api_base

        # 添加 max_tokens（如果配置了且不为 0）
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens

        # 添加 fallback 模型（如果配置了）
        if self.fallback_models:
            params["fallbacks"] = self.fallback_models

        # 合并其他额外参数
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # 调用 LiteLLM
        response = completion(**params)

        # 提取响应内容
        return response.choices[0].message.content

    def _process_gemini_config(self, model: str, api_base: str) -> tuple[str, str]:
        """
        处理 Gemini 兼容端点的配置

        Args:
            model: 模型名称
            api_base: API 基础 URL

        Returns:
            tuple: (处理后的模型名, 处理后的 API 基础 URL)
        """
        # 如果不是 Gemini 模型，直接返回
        if not model.startswith("gemini/"):
            return model, api_base

        # 如果配置了自定义端点
        if api_base:
            # 确保端点格式正确
            normalized_base = self._normalize_gemini_endpoint(api_base)

            # 对于自定义 Gemini 兼容端点，使用 openai/ 前缀强制使用 OpenAI 协议
            # 这样可以兼容所有提供 OpenAI 兼容接口的 Gemini 服务
            if self._should_use_openai_protocol(model, api_base):
                openai_model = model.replace("gemini/", "openai/")
                return openai_model, normalized_base

            return model, normalized_base

        # 使用官方 Gemini 端点
        return model, api_base

    def _normalize_gemini_endpoint(self, endpoint: str) -> str:
        """
        标准化 Gemini API 端点格式

        Args:
            endpoint: 原始端点 URL

        Returns:
            str: 标准化后的端点 URL
        """
        if not endpoint:
            return endpoint

        # 移除末尾的斜杠
        endpoint = endpoint.rstrip("/")

        # 如果不包含 /v1，尝试添加
        if "/v1" not in endpoint and "/v1beta" not in endpoint:
            # 检查是否已经是完整的 API 路径
            if not any(
                endpoint.endswith(suffix)
                for suffix in ["/chat", "/completions", "/generate"]
            ):
                endpoint += "/v1"

        return endpoint

    def _should_use_openai_protocol(self, model: str, api_base: str) -> bool:
        """
        判断是否应该使用 OpenAI 协议

        Args:
            model: 模型名称
            api_base: API 基础 URL

        Returns:
            bool: 是否使用 OpenAI 协议
        """
        # 如果端点包含常见的 OpenAI 兼容路径，使用 OpenAI 协议
        openai_indicators = [
            "/v1/chat/completions",
            "/v1/completions",
            "/chat/completions",
            "/openai",
            "openai-",
        ]

        return any(indicator in api_base.lower() for indicator in openai_indicators)

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        if not self.api_key:
            return (
                False,
                "未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置",
            )

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return (
                False,
                f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）",
            )

        # 验证 Gemini 特定配置
        if self.model.startswith("gemini/"):
            # 检查自定义端点是否有效
            if self.api_base:
                if not self._validate_gemini_endpoint(self.api_base):
                    return (
                        False,
                        f"自定义 Gemini API 端点格式错误: {self.api_base}，应为有效的 HTTP/HTTPS URL",
                    )

        return True, ""

    def _validate_gemini_endpoint(self, endpoint: str) -> bool:
        """
        验证 Gemini API 端点格式

        Args:
            endpoint: API 端点 URL

        Returns:
            bool: 是否有效
        """
        if not endpoint:
            return True  # 空字符串表示使用默认端点

        # 基本格式验证
        if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
            return False

        # Gemini API 通常以 /v1/ 结尾
        if not endpoint.endswith("/"):
            endpoint += "/"

        return True
