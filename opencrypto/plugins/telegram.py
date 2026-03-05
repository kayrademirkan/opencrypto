"""
OpenCrypto — Telegram Plugin (Optional)

Send trading signals and notifications via Telegram Bot API.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
"""

from __future__ import annotations

import logging

import httpx

from opencrypto.core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, USE_TELEGRAM

logger = logging.getLogger(__name__)


async def send_telegram_message(text: str, chat_id: str = "") -> bool:
    """Send a message to Telegram. Returns True on success."""
    if not USE_TELEGRAM:
        return False
    target = chat_id or TELEGRAM_CHAT_ID
    if not target:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                json={
                    "chat_id": target,
                    "text": text,
                    "parse_mode": "HTML",
                },
            )
            return resp.status_code == 200
    except Exception as exc:
        logger.error("Telegram message failed: %s", exc)
        return False


async def send_signal_message(signal: dict, ai_comment: str = "") -> bool:
    """Format and send a trading signal to Telegram."""
    sym = signal.get("display_symbol", signal.get("symbol", "?"))
    direction = signal.get("direction", "?")
    dir_emoji = "📈" if direction == "LONG" else "📉"

    entry = signal.get("entry", 0)
    sl = signal.get("sl", 0)
    tp = signal.get("tp1", signal.get("tp", 0))
    conf = signal.get("confidence", 0)
    lev = signal.get("leverage", 1)
    rr = signal.get("rr_ratio", 0)
    sig_type = signal.get("signal_type_label", signal.get("signal_type", ""))
    reasons = signal.get("reasons", [])

    reasons_text = "\n".join(f"  • {r}" for r in reasons[:8])

    msg = (
        f"{dir_emoji} <b>{sym} — {direction}</b>\n\n"
        f"📊 Type: {sig_type}\n"
        f"🎯 Confidence: {conf:.0f}% | R:R {rr}\n"
        f"⚡ Leverage: {lev}x\n\n"
        f"📍 Entry: <code>{entry:.6g}</code>\n"
        f"🛑 SL: <code>{sl:.6g}</code>\n"
        f"✅ TP: <code>{tp:.6g}</code>\n\n"
        f"📋 Reasons:\n{reasons_text}\n"
    )

    if ai_comment:
        msg += f"\n💬 AI: {ai_comment}\n"

    msg += "\n🔔 OpenCrypto Framework"

    return await send_telegram_message(msg)


async def send_photo(photo_path: str, caption: str = "", chat_id: str = "") -> bool:
    """Send a photo to Telegram."""
    if not USE_TELEGRAM:
        return False
    target = chat_id or TELEGRAM_CHAT_ID
    if not target:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            with open(photo_path, "rb") as f:
                resp = await client.post(
                    url,
                    data={
                        "chat_id": target,
                        "caption": caption,
                        "parse_mode": "HTML",
                    },
                    files={"photo": f},
                )
                return resp.status_code == 200
    except Exception as exc:
        logger.error("Telegram photo send failed: %s", exc)
        return False
