"""
Tests for GoogleProvider with Gemini 3.8 Flash
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from providers.google_ai import GoogleProvider
from providers import create_provider


class TestGoogleProvider(unittest.TestCase):
    """Test GoogleProvider configuration and Gemini 3.8 Flash integration"""

    def setUp(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "dummy-api-key")

    @patch("providers.google_ai.genai.Client")
    def test_default_model_is_gemini_3_8_flash(self, mock_client):
        """Test default LLM model is gemini-3.8-flash"""
        provider = GoogleProvider(api_key=self.api_key)
        self.assertEqual(provider.llm_model, "gemini-3.8-flash")
        self.assertEqual(provider.embedding_model, "gemini-embedding-001")

    @patch("providers.google_ai.genai.Client")
    def test_provider_factory_defaults_to_gemini_3_8_flash(self, mock_client):
        """Test create_provider creates GoogleProvider with gemini-3.8-flash by default"""
        with patch.dict(os.environ, {"PROVIDER": "google", "GOOGLE_API_KEY": "dummy-key"}):
            # Ensure GOOGLE_LLM_MODEL is unset for fallback test
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GOOGLE_LLM_MODEL", None)
                provider = create_provider()
                self.assertIsInstance(provider, GoogleProvider)
                self.assertEqual(provider.llm_model, "gemini-3.8-flash")

    @patch("providers.google_ai.genai.Client")
    def test_thinking_budget_zero_for_short_tokens(self, mock_client_cls):
        """Test thinking_budget=0 is injected when max_tokens < 100 and no thinking config is set"""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.thought = False
        mock_part.text = "Yes"
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key")
        result = provider.generate_text(prompt="test", max_tokens=10)

        # Check call config
        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs.get("config")
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.thinking_config)
        self.assertEqual(config.thinking_config.thinking_budget, 0)
        self.assertEqual(result["response"], "Yes")

    @patch("providers.google_ai.genai.Client")
    def test_thought_parts_separated_from_response(self, mock_client_cls):
        """Test thinking parts (thought=True) are returned under thinking and not response"""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_candidate = MagicMock()
        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.text = "Thinking about invoice..."

        text_part = MagicMock()
        text_part.thought = False
        text_part.text = "10"

        mock_candidate.content.parts = [thought_part, text_part]
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key")
        result = provider.generate_text(prompt="score relevance", max_tokens=500)

        self.assertEqual(result["response"], "10")
        self.assertEqual(result["thinking"], "Thinking about invoice...")

    @patch("providers.google_ai.genai.Client")
    def test_custom_thinking_level_config(self, mock_client_cls):
        """Test custom thinking_level is passed to GenerateContentConfig"""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.thought = False
        mock_part.text = "result"
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key", thinking_level="low")
        result = provider.generate_text(prompt="test", max_tokens=500)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs.get("config")
        self.assertIsNotNone(config.thinking_config)
        self.assertIn("LOW", str(config.thinking_config.thinking_level).upper())

    @patch("providers.google_ai.genai.Client")
    def test_custom_thinking_budget_config(self, mock_client_cls):
        """Test custom thinking_budget is passed to GenerateContentConfig"""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.thought = False
        mock_part.text = "result"
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key", thinking_budget=25)
        result = provider.generate_text(prompt="test", max_tokens=500)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs.get("config")
        self.assertIsNotNone(config.thinking_config)
        self.assertEqual(config.thinking_config.thinking_budget, 25)


if __name__ == "__main__":
    unittest.main()
