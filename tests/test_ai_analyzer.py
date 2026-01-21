import unittest
from datetime import datetime

from trendradar.ai.analyzer import AIAnalyzer


class TestAIAnalyzerResponseParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = AIAnalyzer(
            ai_config={
                "MODEL": "openai/gpt-4o",
                "API_KEY": "test",
                "NUM_RETRIES": 0,
            },
            analysis_config={
                "LANGUAGE": "Chinese",
                "PROMPT_FILE": "ai_analysis_prompt.txt",
                "MAX_NEWS_FOR_ANALYSIS": 50,
                "INCLUDE_RSS": True,
                "INCLUDE_RANK_TIMELINE": False,
            },
            get_time_func=lambda: datetime(2026, 1, 21, 19, 9, 0),
            debug=False,
        )

    def test_parse_valid_json_codeblock(self) -> None:
        response = """```json
{
  "core_trends": "a",
  "sentiment_controversy": "b",
  "signals": "c",
  "rss_insights": "d",
  "outlook_strategy": "e"
}
```"""
        result = self.analyzer._parse_response(response)
        self.assertTrue(result.success)
        self.assertEqual(result.core_trends, "a")
        self.assertEqual(result.outlook_strategy, "e")

    def test_parse_json_with_extra_text(self) -> None:
        response = """这里是一些解释文字（不推荐，但应尽量兼容）
{
  "core_trends": "a",
  "sentiment_controversy": "b",
  "signals": "c",
  "rss_insights": "d",
  "outlook_strategy": "e"
}
谢谢"""
        result = self.analyzer._parse_response(response)
        self.assertTrue(result.success)
        self.assertEqual(result.signals, "c")

    def test_repair_trailing_comma(self) -> None:
        response = """{
          "core_trends": "a",
          "sentiment_controversy": "b",
          "signals": "c",
          "rss_insights": "d",
          "outlook_strategy": "e",
        }"""
        result = self.analyzer._parse_response(response)
        self.assertTrue(result.success)
        self.assertEqual(result.rss_insights, "d")

    def test_missing_required_fields_should_fail(self) -> None:
        response = """{
          "core_trends": "a",
          "sentiment_controversy": "b",
          "rss_insights": "d",
          "outlook_strategy": "e"
        }"""
        result = self.analyzer._parse_response(response)
        self.assertFalse(result.success)
        self.assertIn("缺少字段", result.error)

    def test_unrelated_marker_in_payload_should_fail(self) -> None:
        response = """{
          "core_trends": "[WORD_GROUPS]\\n[CNC/数控机床求购]\\n+求购",
          "sentiment_controversy": "b",
          "signals": "c",
          "rss_insights": "d",
          "outlook_strategy": "e"
        }"""
        result = self.analyzer._parse_response(response)
        self.assertFalse(result.success)
        self.assertIn("响应校验失败", result.error)


if __name__ == "__main__":
    unittest.main()

