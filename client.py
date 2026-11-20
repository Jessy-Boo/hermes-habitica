"""Habitica API v3 client: stdlib-only HTTP over scoped credentials.

Every request carries the three headers Habitica requires:

- ``x-api-user``: the account's user ID
- ``x-api-key``: the account's API token
- ``x-client``: the calling application; omitting it is a hard
  ``BadRequest: Missing x-client headers`` error

Credentials come exclusively from ``agent.secret_scope.get_secret`` (the
values the installer stores for ``HABITICA_USER_ID`` /
``HABITICA_API_TOKEN``). This module never reads environment variables,
never reads ``.env`` files and never logs the token.

Behaviour verified against the live API (2026-11):

- one task type per request: ``/tasks/user?type=`` accepts exactly one
  value; comma-combined types return a response with no ``data`` key
- party detail: bare ``GET /group`` is a 404; use
  ``GET /groups?type=party`` or ``GET /group/<id>``
- party chat posting: the message goes in an URL-encoded form body, not
  the query string (the query form returns a non-JSON error)
- own chat messages are deletable via ``DELETE /groups/<id>/chat/<mid>``
- ``balance.gems`` on ``/user`` is an integer gem count, not an object
- writes are irreversible server-side state changes: scoring a task
  changes streaks, stats and gold with no undo endpoint
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

API_ROOT = "https://habitica.com/api/v3"
CLIENT_ID = "hermes-habitica/1.0.0"

RETRY_ATTEMPTS = 3
BACKOFF_CAP_SECONDS = 30.0

RETRYABLE_STATUS = (429, 503)

# GET /tasks/user takes the plural type names (verified live: the singular
# form returns HTTP 400 "Invalid request parameters").
_LIST_TYPE = {
    "habit": "habits",
    "daily": "dailys",
    "todo": "todos",
    "reward": "rewards",
}


class HabiticaError(RuntimeError):
    """A Habitica API error, with the vendor's error code and message."""


class HabiticaCredentialsError(HabiticaError):
    """The scoped secrets for the Habitica credentials are missing."""


def _credentials() -> tuple[str, str]:
    """Resolve the credential pair through Hermes's scoped secret route.

    There is deliberately no environment-variable fallback: if the scoped
    secrets are absent the call fails, even when ``HABITICA_USER_ID`` /
    ``HABITICA_API_TOKEN`` are set in the process environment. (A unit test
    proves env values are ignored.)
    """
    missing: list[str] = []
    user_id: str | None = None
    api_token: str | None = None
    try:
        user_id = get_secret("HABITICA_USER_ID")
    except Exception:
        missing.append("HABITICA_USER_ID")
    try:
        api_token = get_secret("HABITICA_API_TOKEN")
    except Exception:
        missing.append("HABITICA_API_TOKEN")
    if missing:
        raise HabiticaCredentialsError(
            "Habitica credentials are not configured in Hermes secrets "
            "(missing: " + ", ".join(missing) + "). Set them with "
            "`hermes config set plugins.entries.hermes-habitica."
            "requires_env.<name> <value>`."
        )
    return str(user_id), str(api_token)


class HabiticaClient:
    """Minimal, dependency-free Habitica v3 client."""

    def __init__(
        self,
        user_id: str,
        api_token: str,
        timeout: float = 30.0,
        sleeper=time.sleep,
        urlopen=urllib.request.urlopen,
    ) -> None:
        self.user_id = user_id
        self.api_token = api_token
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._urlopen = urlopen

    @classmethod
    def resolve(cls, timeout: float = 30.0) -> "HabiticaClient":
        """Build a client from the scoped secrets."""
        user_id, api_token = _credentials()
        return cls(user_id=user_id, api_token=api_token, timeout=timeout)

    # ------------------------------------------------------------------
    # request core
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-user": self.user_id,
            "x-api-key": self.api_token,
            "x-client": CLIENT_ID,
            "Accept": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        query: dict | None = None,
        json_body: dict | None = None,
        form_body: dict | None = None,
    ):
        """One API call with 429/503 backoff. Returns the response ``data``."""
        url = API_ROOT + "/" + path.lstrip("/")
        if query:
            url = url + "?" + urllib.parse.urlencode(query)
        data = None
        headers = self._headers()
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif form_body is not None:
            data = urllib.parse.urlencode(form_body).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        last_error: Exception | None = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            req = urllib.request.Request(
                url, data=data, headers=headers, method=method
            )
            try:
                with self._urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if not payload.get("success"):
                    raise HabiticaError(
                        "Habitica API error ("
                        + str(payload.get("error"))
                        + "): "
                        + str(payload.get("message"))
                    )
                return payload.get("data")
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempt < RETRY_ATTEMPTS:
                    delay = self._retry_delay(exc, attempt)
                    logger.info(
                        "Habitica returned %s; retrying in %.1fs (attempt %d)",
                        exc.code,
                        delay,
                        attempt,
                    )
                    self._sleep(delay)
                    last_error = exc
                    continue
                raise HabiticaError(
                    "Habitica API returned HTTP "
                    + str(exc.code)
                    + ": "
                    + self._error_body(exc)
                ) from exc
            except HabiticaError:
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise HabiticaError(
                    "Habitica API request failed: " + str(exc)
                ) from exc
        raise HabiticaError(
            "Habitica API stayed busy after "
            + str(RETRY_ATTEMPTS)
            + " attempts"
        ) from last_error

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                return min(float(header), BACKOFF_CAP_SECONDS)
            except ValueError:
                pass
        return min(2.0 * attempt, BACKOFF_CAP_SECONDS)

    @staticmethod
    def _error_body(exc: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            return str(payload.get("message") or payload.get("error") or exc)
        except Exception:
            return str(exc)

    # ------------------------------------------------------------------
    # typed helpers (thin, so handlers read clearly)
    # ------------------------------------------------------------------

    def get(self, path: str, query: dict | None = None):
        return self.request("GET", path, query=query)

    def post(self, path: str, query: dict | None = None, form_body: dict | None = None, json_body: dict | None = None):
        return self.request(
            "POST", path, query=query, json_body=json_body, form_body=form_body
        )

    def put(self, path: str, json_body: dict | None = None):
        return self.request("PUT", path, json_body=json_body)

    def delete(self, path: str):
        return self.request("DELETE", path)

    # ------------------------------------------------------------------
    # API surfaces (all verified against the live API)
    # ------------------------------------------------------------------

    def user(self) -> dict:
        return self.get("user")

    def tasks(self, task_type: str) -> list:
        """One type per request.

        The tool-facing vocabulary is singular (habit, daily, todo,
        reward); the list endpoint wants the plural forms the API uses
        (habits, dailys, todos, rewards). Task creation bodies, by
        contrast, take the singular form.
        """
        return self.get("tasks/user", query={"type": _LIST_TYPE[task_type]})

    def create_task(self, body: dict) -> dict:
        return self.post("tasks/user", json_body=body)

    def update_task(self, task_id: str, body: dict) -> dict:
        return self.put("tasks/" + task_id, json_body=body)

    def delete_task(self, task_id: str) -> dict:
        return self.delete("tasks/" + task_id)

    def score_task(self, task_id: str, direction: str) -> dict:
        """Score a task. ``direction`` is ``up`` or ``down``."""
        return self.post("tasks/" + task_id + "/score/" + direction)

    def tags(self) -> list:
        return self.get("tags")

    def party(self) -> dict | None:
        """The user's party group, or None when they are in no party."""
        groups = self.get("groups", query={"type": "party"})
        if not groups:
            return None
        return groups[0]

    def party_chat(self, group_id: str, limit: int = 20) -> list:
        return self.get(
            "groups/" + group_id + "/chat", query={"limit": int(limit)}
        )

    def post_party_chat(self, group_id: str, message: str) -> dict:
        """Post to party chat. The message rides in the form body."""
        return self.post(
            "groups/" + group_id + "/chat", form_body={"message": message}
        )

    def delete_party_chat(self, group_id: str, message_id: str) -> dict:
        return self.delete("groups/" + group_id + "/chat/" + message_id)
