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

    def test_empty_response_should_fail(self) -> None:
        """空响应应返回失败"""
        result = self.analyzer._parse_response("")
        self.assertFalse(result.success)
        self.assertIn("空响应", result.error)

    def test_whitespace_only_response_should_fail(self) -> None:
        """纯空白响应应返回失败"""
        result = self.analyzer._parse_response("   \n\t  ")
        self.assertFalse(result.success)
        self.assertIn("空响应", result.error)

    def test_none_response_should_fail(self) -> None:
        """None 响应应返回失败"""
        result = self.analyzer._parse_response(None)  # type: ignore
        self.assertFalse(result.success)
        self.assertIn("空响应", result.error)

    def test_invalid_json_should_fail(self) -> None:
        """无效 JSON 应返回失败"""
        response = '{"core_trends": "a", "sentiment_controversy": '
        result = self.analyzer._parse_response(response)
        self.assertFalse(result.success)
        self.assertIn("JSON", result.error)

    def test_json_with_bom_should_parse(self) -> None:
        """带 BOM 的响应应正确解析"""
        response = '\ufeff{"core_trends": "a", "sentiment_controversy": "b", "signals": "c", "rss_insights": "d", "outlook_strategy": "e"}'
        result = self.analyzer._parse_response(response)
        self.assertTrue(result.success)
        self.assertEqual(result.core_trends, "a")

    def test_generic_codeblock_should_parse(self) -> None:
        """不带语言标识的代码块应正确解析"""
        response = """```
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

    def test_null_field_value_should_convert_to_empty_string(self) -> None:
        """null 字段值应转为空字符串"""
        response = """{
          "core_trends": null,
          "sentiment_controversy": "b",
          "signals": "c",
          "rss_insights": "d",
          "outlook_strategy": "e"
        }"""
        result = self.analyzer._parse_response(response)
        self.assertTrue(result.success)
        self.assertEqual(result.core_trends, "")

    def test_nested_object_field_should_convert_to_json_string(self) -> None:
        """嵌套对象字段应转为 JSON 字符串"""
        response = """{
          "core_trends": {"key": "value"},
          "sentiment_controversy": "b",
          "signals": "c",
          "rss_insights": "d",
          "outlook_strategy": "e"
        }"""
        result = self.analyzer._parse_response(response)
        self.assertTrue(result.success)
        self.assertIn("key", result.core_trends)
        self.assertIn("value", result.core_trends)

    def test_config_yaml_marker_should_fail(self) -> None:
        """包含 config.yaml 标记的响应应判为跑偏"""
        response = """{
          "core_trends": "请参考 config.yaml 进行配置",
          "sentiment_controversy": "b",
          "signals": "c",
          "rss_insights": "d",
          "outlook_strategy": "e"
        }"""
        result = self.analyzer._parse_response(response)
        self.assertFalse(result.success)
        self.assertIn("响应校验失败", result.error)

    def test_json_array_top_level_should_fail(self) -> None:
        """顶层为数组的 JSON 应返回失败"""
        response = '[{"core_trends": "a"}]'
        result = self.analyzer._parse_response(response)
        self.assertFalse(result.success)
        # 数组被解析后缺少字段，校验失败
        self.assertIn("响应校验失败", result.error)


class TestExtractJsonText(unittest.TestCase):
    """测试 JSON 提取函数"""

    def test_extract_from_empty_string(self) -> None:
        """空字符串应返回空"""
        result = AIAnalyzer._extract_json_text("")
        self.assertEqual(result, "")

    def test_extract_from_none(self) -> None:
        """None 应返回空（防御性）"""
        result = AIAnalyzer._extract_json_text(None)  # type: ignore
        self.assertEqual(result, "")

    def test_extract_plain_json(self) -> None:
        """纯 JSON 应原样返回"""
        json_str = '{"a": 1}'
        result = AIAnalyzer._extract_json_text(json_str)
        self.assertEqual(result, '{"a": 1}')

    def test_extract_json_from_markdown(self) -> None:
        """从 Markdown 代码块提取"""
        text = "解释文字\n```json\n{\"a\": 1}\n```\n后续文字"
        result = AIAnalyzer._extract_json_text(text)
        self.assertEqual(result, '{"a": 1}')


class TestRepairJsonIssues(unittest.TestCase):
    """测试 JSON 修复函数"""

    def test_repair_empty_string(self) -> None:
        """空字符串应返回空"""
        result = AIAnalyzer._repair_common_json_issues("")
        self.assertEqual(result, "")

    def test_repair_trailing_comma_in_object(self) -> None:
        """修复对象尾随逗号"""
        result = AIAnalyzer._repair_common_json_issues('{"a": 1, }')
        # 正则移除逗号和空白，结果为 {"a": 1}
        self.assertEqual(result, '{"a": 1}')

    def test_repair_trailing_comma_in_array(self) -> None:
        """修复数组尾随逗号"""
        result = AIAnalyzer._repair_common_json_issues('[1, 2, ]')
        # 正则移除逗号和空白，结果为 [1, 2]
        self.assertEqual(result, '[1, 2]')

    def test_no_change_for_valid_json(self) -> None:
        """有效 JSON 不应改变"""
        valid = '{"a": 1, "b": 2}'
        result = AIAnalyzer._repair_common_json_issues(valid)
        self.assertEqual(result, valid)


class TestLooksUnrelated(unittest.TestCase):
    """测试内容跑偏检测函数"""

    def test_empty_string_not_unrelated(self) -> None:
        """空字符串不算跑偏"""
        self.assertFalse(AIAnalyzer._looks_unrelated(""))

    def test_normal_content_not_unrelated(self) -> None:
        """正常内容不算跑偏"""
        self.assertFalse(AIAnalyzer._looks_unrelated("今日热点分析"))

    def test_word_groups_marker_is_unrelated(self) -> None:
        """[WORD_GROUPS] 标记算跑偏"""
        self.assertTrue(AIAnalyzer._looks_unrelated("[WORD_GROUPS]\n+关键词"))

    def test_frequency_words_is_unrelated(self) -> None:
        """frequency_words.txt 标记算跑偏"""
        self.assertTrue(AIAnalyzer._looks_unrelated("请编辑 frequency_words.txt 文件"))

    def test_case_insensitive_detection(self) -> None:
        """大小写不敏感检测"""
        self.assertTrue(AIAnalyzer._looks_unrelated("[INCLUDE_WORDS]"))
        self.assertTrue(AIAnalyzer._looks_unrelated("[include_words]"))


if __name__ == "__main__":
    unittest.main()

