"""Plugin settings reader.

Settings live under ``plugins.entries.hermes-habitica.settings`` in
config.yaml; the same keys are rendered by ``config_schema`` in plugin.yaml
and written by ``hermes config set plugins.entries...settings.<key>``.

Values are read through Hermes's own config loader
(``load_config_readonly``) so profile resolution and locking stay with
Hermes; this module never parses config.yaml or .env directly and never
reads ``os.environ``.
"""

from __future__ import annotations

from typing import Any

PLUGIN_ID = "hermes-habitica"

# Schema defaults, mirrored from plugin.yaml. load_settings() merges user
# values over these, so callers always see every knob.
DEFAULTS: dict[str, Any] = {
    "request_timeout_seconds": 30,
    "chat_fetch_limit": 20,
}


def load_settings() -> dict[str, Any]:
    """User settings merged over DEFAULTS.

    Fails closed to the defaults: settings are optional behaviour tuning,
    and the reader runs in CLI and cron processes where the full config
    stack may not be initialised.
    """
    merged: dict[str, Any] = dict(DEFAULTS)
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        entries = (config.get("plugins") or {}).get("entries") or {}
        user = ((entries.get(PLUGIN_ID) or {}).get("settings")) or {}
        for key in DEFAULTS:
            if key in user:
                merged[key] = user[key]
    except Exception:
        pass
    return merged
