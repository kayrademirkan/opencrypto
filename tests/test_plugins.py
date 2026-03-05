"""Tests for optional plugins — no network calls, just contract verification."""


def test_telegram_module_import():
    from opencrypto.plugins.telegram import send_signal_message, send_telegram_message

    assert callable(send_telegram_message)
    assert callable(send_signal_message)


def test_charts_module_import():
    from opencrypto.plugins.charts import generate_chart

    assert callable(generate_chart)


def test_llm_module_import():
    from opencrypto.plugins.llm import ai_comment

    assert callable(ai_comment)


def test_llm_disabled_returns_empty():
    """When USE_LLM is False, ai_comment should return empty without calling API."""
    from opencrypto.plugins.llm import ai_comment

    result = ai_comment({"symbol": "BTCUSDT", "direction": "LONG", "confidence": 70})
    assert result["comment"] == ""
    assert result["tokens_used"] == 0


def test_telegram_returns_bool():
    """send_telegram_message always returns a bool."""
    import asyncio

    from opencrypto.plugins.telegram import send_telegram_message

    result = asyncio.run(send_telegram_message("", chat_id=""))
    assert isinstance(result, bool)
