# coding=utf-8
"""
AI 分析器模块

调用 AI 大模型对热点新闻进行深度分析
基于 LiteLLM 统一接口，支持 100+ AI 提供商
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from trendradar.ai.client import AIClient


@dataclass
class AIAnalysisResult:
    """AI 分析结果"""
    # 新版 5 核心板块
    core_trends: str = ""                # 核心热点与舆情态势
    sentiment_controversy: str = ""      # 舆论风向与争议
    signals: str = ""                    # 异动与弱信号
    rss_insights: str = ""               # RSS 深度洞察
    outlook_strategy: str = ""           # 研判与策略建议

    # 基础元数据
    raw_response: str = ""               # 原始响应
    success: bool = False                # 是否成功
    error: str = ""                      # 错误信息

    # 新闻数量统计
    total_news: int = 0                  # 总新闻数（热榜+RSS）
    analyzed_news: int = 0               # 实际分析的新闻数
    max_news_limit: int = 0              # 分析上限配置值
    hotlist_count: int = 0               # 热榜新闻数
    rss_count: int = 0                   # RSS 新闻数


class AIAnalyzer:
    """AI 分析器"""

    def __init__(
        self,
        ai_config: Dict[str, Any],
        analysis_config: Dict[str, Any],
        get_time_func: Callable,
        debug: bool = False,
    ):
        """
        初始化 AI 分析器

        Args:
            ai_config: AI 模型配置（LiteLLM 格式）
            analysis_config: AI 分析功能配置（language, prompt_file 等）
            get_time_func: 获取当前时间的函数
            debug: 是否开启调试模式
        """
        self.ai_config = ai_config
        self.analysis_config = analysis_config
        self.get_time_func = get_time_func
        self.debug = debug

        # 创建 AI 客户端（基于 LiteLLM）
        self.client = AIClient(ai_config)

        # 验证配置
        valid, error = self.client.validate_config()
        if not valid:
            print(f"[AI] 配置警告: {error}")

        # 从分析配置获取功能参数
        self.max_news = analysis_config.get("MAX_NEWS_FOR_ANALYSIS", 50)
        self.include_rss = analysis_config.get("INCLUDE_RSS", True)
        self.include_rank_timeline = analysis_config.get("INCLUDE_RANK_TIMELINE", False)
        self.language = analysis_config.get("LANGUAGE", "Chinese")

        # 加载提示词模板
        self.system_prompt, self.user_prompt_template = self._load_prompt_template(
            analysis_config.get("PROMPT_FILE", "ai_analysis_prompt.txt")
        )

    @staticmethod
    def _truncate_text(text: str, limit: int = 800) -> str:
        """截断用于日志输出的文本，避免刷屏。"""
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit] + "..."

    def _dump_failed_attempt(
        self,
        *,
        attempt: int,
        max_attempts: int,
        error: str,
        user_prompt: str,
        response: str,
    ) -> Optional[str]:
        """调试模式下落盘失败样本，便于复现与排查。"""
        if not self.debug:
            return None

        try:
            now = self.get_time_func()
            date_folder = now.strftime("%Y-%m-%d")
            time_part = now.strftime("%H-%M-%S")
            out_dir = Path("output") / "ai_debug" / date_folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"ai_analysis_{time_part}_attempt{attempt}_of_{max_attempts}.txt"

            meta_lines = [
                "==== TrendRadar AI 分析失败样本 ====",
                f"attempt: {attempt}/{max_attempts}",
                f"model: {self.ai_config.get('MODEL', '')}",
                f"api_base: {self.ai_config.get('API_BASE', '')}",
                f"error: {error}",
                "",
                "---- User Prompt ----",
                user_prompt or "",
                "",
                "---- Raw Response ----",
                response or "",
                "",
            ]
            out_path.write_text("\n".join(meta_lines), encoding="utf-8")
            return str(out_path)
        except Exception:
            return None

    def _load_prompt_template(self, prompt_file: str) -> tuple:
        """加载提示词模板"""
        config_dir = Path(__file__).parent.parent.parent / "config"
        prompt_path = config_dir / prompt_file

        if not prompt_path.exists():
            print(f"[AI] 提示词文件不存在: {prompt_path}")
            return "", ""

        content = prompt_path.read_text(encoding="utf-8")

        # 解析 [system] 和 [user] 部分
        system_prompt = ""
        user_prompt = ""

        if "[system]" in content and "[user]" in content:
            parts = content.split("[user]")
            system_part = parts[0]
            user_part = parts[1] if len(parts) > 1 else ""

            # 提取 system 内容
            if "[system]" in system_part:
                system_prompt = system_part.split("[system]")[1].strip()

            user_prompt = user_part.strip()
        else:
            # 整个文件作为 user prompt
            user_prompt = content

        return system_prompt, user_prompt

    def analyze(
        self,
        stats: List[Dict],
        rss_stats: Optional[List[Dict]] = None,
        report_mode: str = "daily",
        report_type: str = "当日汇总",
        platforms: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> AIAnalysisResult:
        """
        执行 AI 分析

        Args:
            stats: 热榜统计数据
            rss_stats: RSS 统计数据
            report_mode: 报告模式
            report_type: 报告类型
            platforms: 平台列表
            keywords: 关键词列表

        Returns:
            AIAnalysisResult: 分析结果
        """
        if not self.client.api_key:
            return AIAnalysisResult(
                success=False,
                error="未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"
            )

        # 准备新闻内容并获取统计数据
        news_content, rss_content, hotlist_total, rss_total, analyzed_count = self._prepare_news_content(stats, rss_stats)
        total_news = hotlist_total + rss_total

        if not news_content and not rss_content:
            return AIAnalysisResult(
                success=False,
                error="没有可分析的新闻内容",
                total_news=total_news,
                hotlist_count=hotlist_total,
                rss_count=rss_total,
                analyzed_news=0,
                max_news_limit=self.max_news
            )

        # 构建提示词
        current_time = self.get_time_func().strftime("%Y-%m-%d %H:%M:%S")

        # 提取关键词
        if not keywords:
            keywords = [s.get("word", "") for s in stats if s.get("word")] if stats else []

        # 使用安全的字符串替换，避免模板中其他花括号（如 JSON 示例）被误解析
        user_prompt = self.user_prompt_template
        user_prompt = user_prompt.replace("{report_mode}", report_mode)
        user_prompt = user_prompt.replace("{report_type}", report_type)
        user_prompt = user_prompt.replace("{current_time}", current_time)
        user_prompt = user_prompt.replace("{news_count}", str(hotlist_total))
        user_prompt = user_prompt.replace("{rss_count}", str(rss_total))
        user_prompt = user_prompt.replace("{platforms}", ", ".join(platforms) if platforms else "多平台")
        user_prompt = user_prompt.replace("{keywords}", ", ".join(keywords[:20]) if keywords else "无")
        user_prompt = user_prompt.replace("{news_content}", news_content)
        user_prompt = user_prompt.replace("{rss_content}", rss_content)
        user_prompt = user_prompt.replace("{language}", self.language)

        if self.debug:
            print("\n" + "=" * 80)
            print("[AI 调试] 发送给 AI 的完整提示词")
            print("=" * 80)
            if self.system_prompt:
                print("\n--- System Prompt ---")
                print(self.system_prompt)
            print("\n--- User Prompt ---")
            print(user_prompt)
            print("=" * 80 + "\n")

        # 调用 AI API：增加内容级校验与重试，避免“返回了内容但不符合预期”导致污染推送
        num_retries = int(self.ai_config.get("NUM_RETRIES", 0) or 0)
        max_attempts = max(1, 1 + num_retries)

        base_messages: List[Dict[str, str]] = []
        if self.system_prompt:
            base_messages.append({"role": "system", "content": self.system_prompt})
        base_messages.append({"role": "user", "content": user_prompt})

        last_response = ""
        last_error = ""
        final_result: Optional[AIAnalysisResult] = None

        for attempt in range(1, max_attempts + 1):
            messages = list(base_messages)
            if attempt > 1:
                retry_hint = (
                    "上一次输出未通过校验。请严格按以下要求重新输出：\n"
                    "1) 仅返回一个 JSON 对象，禁止包含任何解释性文字/Markdown/代码块标记\n"
                    "2) 必须包含 5 个字段：core_trends, sentiment_controversy, signals, rss_insights, outlook_strategy\n"
                    "3) 字段值必须是字符串\n"
                )
                if last_error:
                    retry_hint += f"\n校验失败原因：{last_error}\n"
                if last_response:
                    messages.append({"role": "assistant", "content": last_response})
                messages.append({"role": "user", "content": retry_hint})

            try:
                # 由外层循环统一控制重试次数，避免与 LiteLLM 的 num_retries 叠加放大
                response = self.client.chat(messages, num_retries=0)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                last_error = f"API 调用异常 ({error_type}): {error_msg}"
                if attempt < max_attempts:
                    print(f"[AI] 第 {attempt}/{max_attempts} 次请求失败，将重试: {last_error}")
                self._dump_failed_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=last_error,
                    user_prompt=user_prompt,
                    response="",
                )
                continue

            last_response = response
            parsed = self._parse_response(response)
            if parsed.success:
                final_result = parsed
                break

            last_error = parsed.error or "未知错误"
            if attempt < max_attempts:
                print(f"[AI] 第 {attempt}/{max_attempts} 次返回无效内容，将重试: {last_error}")
                if self.debug:
                    print(
                        "[AI 调试] 无效响应片段: "
                        + self._truncate_text((response or "").replace("\n", "\\n"), 500)
                    )
            self._dump_failed_attempt(
                attempt=attempt,
                max_attempts=max_attempts,
                error=last_error,
                user_prompt=user_prompt,
                response=response,
            )
            final_result = parsed

        if final_result is None or not final_result.success:
            # 明确失败：避免把跑偏内容当作成功推送
            friendly_msg = last_error or "AI 分析失败（未知错误）"
            friendly_msg = f"{friendly_msg}（已尝试 {max_attempts} 次）"
            final_result = AIAnalysisResult(success=False, error=friendly_msg, raw_response=last_response)

        # 如果配置未启用 RSS 分析，强制清空 AI 返回的 RSS 洞察
        if not self.include_rss:
            final_result.rss_insights = ""

        # 填充统计数据（无论成功与否都提供，便于排查）
        final_result.total_news = total_news
        final_result.hotlist_count = hotlist_total
        final_result.rss_count = rss_total
        final_result.analyzed_news = analyzed_count
        final_result.max_news_limit = self.max_news
        return final_result

    def _prepare_news_content(
        self,
        stats: List[Dict],
        rss_stats: Optional[List[Dict]] = None,
    ) -> tuple:
        """
        准备新闻内容文本（增强版）

        热榜新闻包含：来源、标题、排名范围、时间范围、出现次数
        RSS 包含：来源、标题、发布时间

        Returns:
            tuple: (news_content, rss_content, hotlist_total, rss_total, analyzed_count)
        """
        news_lines = []
        rss_lines = []
        news_count = 0
        rss_count = 0

        # 计算总新闻数
        hotlist_total = sum(len(s.get("titles", [])) for s in stats) if stats else 0
        rss_total = sum(len(s.get("titles", [])) for s in rss_stats) if rss_stats else 0

        # 热榜内容
        if stats:
            for stat in stats:
                word = stat.get("word", "")
                titles = stat.get("titles", [])
                if word and titles:
                    news_lines.append(f"\n**{word}** ({len(titles)}条)")
                    for t in titles:
                        if not isinstance(t, dict):
                            continue
                        title = t.get("title", "")
                        if not title:
                            continue

                        # 来源
                        source = t.get("source_name", t.get("source", ""))

                        # 构建行
                        if source:
                            line = f"- [{source}] {title}"
                        else:
                            line = f"- {title}"

                        # 始终显示简化格式：排名范围 + 时间范围 + 出现次数
                        ranks = t.get("ranks", [])
                        if ranks:
                            min_rank = min(ranks)
                            max_rank = max(ranks)
                            rank_str = f"{min_rank}" if min_rank == max_rank else f"{min_rank}-{max_rank}"
                        else:
                            rank_str = "-"

                        first_time = t.get("first_time", "")
                        last_time = t.get("last_time", "")
                        time_str = self._format_time_range(first_time, last_time)

                        appear_count = t.get("count", 1)

                        line += f" | 排名:{rank_str} | 时间:{time_str} | 出现:{appear_count}次"

                        # 开启完整时间线时，额外添加轨迹
                        if self.include_rank_timeline:
                            rank_timeline = t.get("rank_timeline", [])
                            timeline_str = self._format_rank_timeline(rank_timeline)
                            line += f" | 轨迹:{timeline_str}"

                        news_lines.append(line)

                        news_count += 1
                        if news_count >= self.max_news:
                            break
                if news_count >= self.max_news:
                    break

        # RSS 内容（仅在启用时构建）
        if self.include_rss and rss_stats:
            remaining = self.max_news - news_count
            for stat in rss_stats:
                if rss_count >= remaining:
                    break
                word = stat.get("word", "")
                titles = stat.get("titles", [])
                if word and titles:
                    rss_lines.append(f"\n**{word}** ({len(titles)}条)")
                    for t in titles:
                        if not isinstance(t, dict):
                            continue
                        title = t.get("title", "")
                        if not title:
                            continue

                        # 来源
                        source = t.get("source_name", t.get("feed_name", ""))

                        # 发布时间
                        time_display = t.get("time_display", "")

                        # 构建行：[来源] 标题 | 发布时间
                        if source:
                            line = f"- [{source}] {title}"
                        else:
                            line = f"- {title}"
                        if time_display:
                            line += f" | {time_display}"
                        rss_lines.append(line)

                        rss_count += 1
                        if rss_count >= remaining:
                            break

        news_content = "\n".join(news_lines) if news_lines else ""
        rss_content = "\n".join(rss_lines) if rss_lines else ""
        total_count = news_count + rss_count

        return news_content, rss_content, hotlist_total, rss_total, total_count

    @staticmethod
    def _extract_json_text(response: str) -> str:
        """
        从模型响应中提取 JSON 字符串

        兼容：
        - ```json ... ```
        - ``` ... ```
        - 夹杂解释文本的情况：截取最外层 { ... } 段落
        """
        if not response:
            return ""

        text = response.strip().lstrip("\ufeff")
        if not text:
            return ""

        # 1) 优先处理 ```json fenced block
        fence_patterns = ["```json", "```JSON", "```"]
        for fence in fence_patterns:
            if fence in text:
                parts = text.split(fence, 1)
                if len(parts) < 2:
                    continue
                code_block = parts[1]
                end_idx = code_block.find("```")
                extracted = code_block[:end_idx] if end_idx != -1 else code_block
                extracted = extracted.strip()
                if extracted:
                    return extracted

        # 2) 兜底：截取第一个 { 到最后一个 } 之间的内容
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()

        return text

    @staticmethod
    def _repair_common_json_issues(json_text: str) -> str:
        """修复少量常见 JSON 问题（例如尾随逗号），尽量避免误修复。"""
        if not json_text:
            return ""
        # 移除对象/数组尾随逗号：{ "a": 1, } / [1,2,]
        repaired = re.sub(r",\s*([}\]])", r"\1", json_text)
        return repaired.strip()

    @staticmethod
    def _looks_unrelated(text: str) -> bool:
        """基于强特征的内容级校验：识别明显跑偏的配置/模板类输出。"""
        if not text:
            return False
        lowered = text.lower()
        strong_markers = [
            "[word_groups]",
            "[include_words]",
            "[exclude_words]",
            "frequency_words",
            "frequency_words.txt",
            "config.yaml",
            "setup-windows",
            "setup-mac",
        ]
        return any(marker in lowered for marker in strong_markers)

    @staticmethod
    def _validate_analysis_payload(data: Any, raw_response: str) -> Tuple[bool, str, Dict[str, str]]:
        """校验 JSON 结构与内容，返回 (是否有效, 错误原因, 规范化后的字段字典)。"""
        if not isinstance(data, dict):
            return False, "JSON 顶层不是对象（dict）", {}

        expected_keys = [
            "core_trends",
            "sentiment_controversy",
            "signals",
            "rss_insights",
            "outlook_strategy",
        ]
        missing = [k for k in expected_keys if k not in data]
        if missing:
            return False, f"JSON 缺少字段: {', '.join(missing)}", {}

        normalized: Dict[str, str] = {}
        for key in expected_keys:
            value = data.get(key, "")
            if value is None:
                normalized[key] = ""
            elif isinstance(value, str):
                normalized[key] = value.strip()
            elif isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = str(value)

        combined_text = (raw_response or "") + "\n" + "\n".join(normalized.values())
        if AIAnalyzer._looks_unrelated(combined_text):
            return False, "内容疑似跑偏（出现配置/词表模板特征）", {}

        return True, "", normalized

    def _format_time_range(self, first_time: str, last_time: str) -> str:
        """格式化时间范围（简化显示，只保留时分）"""
        def extract_time(time_str: str) -> str:
            if not time_str:
                return "-"
            # 尝试提取 HH:MM 部分
            if " " in time_str:
                parts = time_str.split(" ")
                if len(parts) >= 2:
                    time_part = parts[1]
                    if ":" in time_part:
                        return time_part[:5]  # HH:MM
            elif ":" in time_str:
                return time_str[:5]
            # 处理 HH-MM 格式
            result = time_str[:5] if len(time_str) >= 5 else time_str
            if len(result) == 5 and result[2] == '-':
                result = result.replace('-', ':')
            return result

        first = extract_time(first_time)
        last = extract_time(last_time)

        if first == last or last == "-":
            return first
        return f"{first}~{last}"

    def _format_rank_timeline(self, rank_timeline: List[Dict]) -> str:
        """格式化排名时间线"""
        if not rank_timeline:
            return "-"

        parts = []
        for item in rank_timeline:
            time_str = item.get("time", "")
            if len(time_str) == 5 and time_str[2] == '-':
                time_str = time_str.replace('-', ':')
            rank = item.get("rank")
            if rank is None:
                parts.append(f"0({time_str})")
            else:
                parts.append(f"{rank}({time_str})")

        return "→".join(parts)

    def _parse_response(self, response: str) -> AIAnalysisResult:
        """解析 AI 响应"""
        result = AIAnalysisResult(raw_response=response)

        if not response or not response.strip():
            result.error = "AI 返回空响应"
            return result

        json_text = self._extract_json_text(response)
        json_text = self._repair_common_json_issues(json_text)

        if not json_text:
            result.error = "提取的 JSON 内容为空"
            return result

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            error_context = json_text[max(0, e.pos - 30):e.pos + 30] if e.pos is not None else ""
            result.error = f"JSON 解析错误 (位置 {e.pos}): {e.msg}"
            if error_context:
                result.error += f"，上下文: ...{error_context}..."
            return result
        except Exception as e:
            result.error = f"JSON 解析失败: {type(e).__name__}: {str(e)}"
            return result

        valid, reason, normalized = self._validate_analysis_payload(data, raw_response=response)
        if not valid:
            result.error = f"响应校验失败: {reason}"
            return result

        result.core_trends = normalized["core_trends"]
        result.sentiment_controversy = normalized["sentiment_controversy"]
        result.signals = normalized["signals"]
        result.rss_insights = normalized["rss_insights"]
        result.outlook_strategy = normalized["outlook_strategy"]
        result.success = True

        return result
