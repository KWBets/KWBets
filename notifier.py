import logging

import httpx
import pandas as pd

import config
from alerts.alert_engine import format_alert

logger = logging.getLogger(__name__)


def send_console_alert(message: str) -> None:
    print(f"\n{'=' * 50}\n{message}\n{'=' * 50}\n")


def send_slack_alert(message: str) -> None:
    if not config.SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not configured")
        return
    with httpx.Client() as client:
        client.post(config.SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)


def send_discord_alert(message: str) -> None:
    if not config.DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not configured")
        return
    with httpx.Client() as client:
        client.post(config.DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)


CHANNELS = {
    "console": send_console_alert,
    "slack": send_slack_alert,
    "discord": send_discord_alert,
}


def notify_value_bets(value_bets: pd.DataFrame) -> int:
    if value_bets.empty:
        return 0

    sent = 0
    for _, bet in value_bets.iterrows():
        message = format_alert(bet)
        for channel in config.ALERT_CHANNELS:
            channel = channel.strip()
            handler = CHANNELS.get(channel)
            if handler:
                try:
                    handler(message)
                    sent += 1
                except Exception as exc:
                    logger.error("Failed to send alert via %s: %s", channel, exc)
    return sent
