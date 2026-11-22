"""Habitica agent tools: status, tasks CRUD, scoring, tags and party.

Registered in every Hermes process via ``provides_tools`` in plugin.yaml.

Handler contract (``tools/registry.py``): the dispatcher calls a plugin
tool as ``handler(args, **kwargs)`` where ``args`` is the model's argument
dict and ``**kwargs`` carries dispatcher context fields (``task_id``,
``session_id``, ``user_task``, ...) filtered by the handler's signature.
A handler must therefore take ``args`` as its only meaningful parameter,
read the model's arguments out of it, and return a string (JSON here).
A handler that declares a name the dispatcher also injects receives two
values for it and crashes; the public handlers below never declare one.

Irreversible and socially visible actions are additionally gated through
Hermes' human-approval system: ``_on_pre_tool_call`` returns an
``approve`` directive for ``habitica_task_delete`` and for the party
``chat_post`` / ``chat_delete`` actions, so Hermes asks the human before
any such call runs. Scoring stays description-guarded (a tick is a
routine, usually-positive action and the bread and butter of the plugin);
deletes and chat writes require a human tap.

Handlers build a lazily-resolved ``HabiticaClient`` unless the caller
injects one, which is how the unit tests drive everything through a fake
transport with no live network.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Optional

from .client import HabiticaClient, HabiticaError
from .settings import load_settings

logger = logging.getLogger(__name__)

TOOLSET = "habitica"

TASK_TYPES = ("habit", "daily", "todo", "reward")
DIFFICULTIES = {"trivial": 0.1, "easy": 1.0, "medium": 1.5, "hard": 2.0}
WEEKDAYS = ("su", "m", "t", "w", "th", "f", "s")

TICK_SAFETY = (
    "Scores a Habitica task: irreversible, affects streaks permanently "
    "(a missed daily kills its streak, a tick cannot be undone). Only call "
    "when the user names the task or explicitly asks to tick it."
)


# ----------------------------------------------------------------------
# output shaping
# ----------------------------------------------------------------------


def _task_summary(task: dict) -> dict:
    """Compact task view: enough to act on, small enough to read."""
    out: dict[str, Any] = {
        "id": task.get("id"),
        "type": task.get("type"),
        "text": task.get("text"),
    }
    # Habitica stores difficulty internally as "priority" and both GET and
    # PUT responses report "priority", not "difficulty".
    difficulty = task.get("difficulty")
    if difficulty is None:
        difficulty = task.get("priority")
    out["difficulty"] = difficulty
    for key in ("notes", "value", "streak", "completed"):
        if task.get(key) not in (None, "", False) or key == "value":
            out[key] = task.get(key)
    checklist = task.get("checklist") or []
    if checklist:
        out["checklist"] = [
            {"text": item.get("text"), "completed": item.get("completed")}
            for item in checklist
        ]
    if task.get("type") == "daily":
        out["frequency"] = task.get("frequency")
        out["everyX"] = task.get("everyX")
        if task.get("repeat"):
            out["repeat_days"] = [
                day for day, on in (task["repeat"] or {}).items() if on
            ]
    if task.get("type") == "habit":
        out["up"] = task.get("up")
        out["down"] = task.get("down")
    tag_ids = task.get("tags") or []
    if tag_ids:
        out["tags"] = tag_ids
    return out


def _clean_task_body(body: dict, task_type: str | None = None) -> dict:
    """Validate and normalise a create/update body."""
    out: dict[str, Any] = {}
    if body.get("text"):
        out["text"] = str(body["text"])
    if body.get("notes") is not None:
        out["notes"] = str(body["notes"])
    if body.get("difficulty"):
        word = str(body["difficulty"]).lower()
        if word not in DIFFICULTIES:
            raise HabiticaError(
                "difficulty must be one of: " + ", ".join(DIFFICULTIES)
            )
        # The API field is "priority" (verified live: POST /tasks/user
        # silently ignores "difficulty" and keeps the default).
        out["priority"] = DIFFICULTIES[word]
    if body.get("tags"):
        out["tags"] = list(body["tags"])
    checklist = body.get("checklist")
    if checklist:
        out["checklist"] = [{"text": str(item)} for item in checklist]

    if task_type in (None, "habit"):
        for key in ("up", "down"):
            if body.get(key) is not None:
                out[key] = bool(body[key])
    if task_type in (None, "daily"):
        if body.get("frequency"):
            freq = str(body["frequency"]).lower()
            if freq not in ("daily", "weekly", "monthly", "yearly"):
                raise HabiticaError(
                    "frequency must be daily, weekly, monthly or yearly"
                )
            out["frequency"] = freq
        if body.get("every_x") is not None:
            out["everyX"] = int(body["every_x"])
        days = body.get("repeat_days")
        if days:
            lowered = [str(d).lower() for d in days]
            bad = [d for d in lowered if d not in WEEKDAYS]
            if bad:
                raise HabiticaError(
                    "repeat_days uses keys like su,m,t,w,th,f,s; unknown: "
                    + ",".join(bad)
                )
            out["repeat"] = {day: (day in lowered) for day in WEEKDAYS}
    if task_type in (None, "reward") and body.get("value") is not None:
        out["value"] = float(body["value"])
    return out


def _require_uuid(value: str, field: str) -> str:
    """Model-supplied ids go into URL paths; validate them as UUIDs."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        raise HabiticaError(field + " must be a valid UUID") from None


# ----------------------------------------------------------------------
# tool handlers (client injectable for tests)
#
# Public signature is the Hermes tool contract: ``handler(args, **kwargs)
# -> str``. The ``_impl_*`` functions hold the behaviour and stay simple
# to test; the public wrapper validates, delegates, and JSON-encodes.
# ----------------------------------------------------------------------


def _make_client(client: Optional[HabiticaClient], timeout: float) -> HabiticaClient:
    if client is not None:
        return client
    return HabiticaClient.resolve(timeout=timeout)


def _settings() -> dict[str, Any]:
    return load_settings()


def _impl_status(c: HabiticaClient) -> dict:
    user = c.user()
    data = user.get("stats") or {}
    balance = user.get("balance")
    if isinstance(balance, dict):
        gems = balance.get("gems")
        hourglasses = balance.get("hourglasses")
    else:
        # some accounts return the gem count as a bare number
        gems = balance
        hourglasses = 0
    if isinstance(gems, dict):
        gems = gems.get("gems")
    streaks = sorted(
        (
            {"text": t.get("text"), "streak": t.get("streak") or 0}
            for t in c.tasks("daily")
            if t.get("streak")
        ),
        key=lambda item: item["streak"],
        reverse=True,
    )
    return {
        "username": user.get("profile", {}).get("name"),
        "level": data.get("lvl"),
        "class": data.get("class"),
        "hp": data.get("hp"),
        "maxHealth": data.get("maxHealth"),
        "mp": data.get("mp"),
        "maxMP": data.get("maxMP"),
        "exp": data.get("exp"),
        "toNextLevel": data.get("toNextLevel"),
        "gold": data.get("gp"),
        "gems": gems,
        "hourglasses": hourglasses,
        "top_daily_streaks": streaks[:5],
    }


def habitica_status(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """Character stats, balances and a streaks digest."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        return json.dumps(_impl_status(c), ensure_ascii=False)
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})


def _impl_tasks(c: HabiticaClient, task_type: str, include_completed: bool, limit: int) -> dict:
    if task_type not in TASK_TYPES:
        raise HabiticaError("task_type must be one of: " + ", ".join(TASK_TYPES))
    tasks = c.tasks(task_type)
    if not include_completed:
        tasks = [t for t in tasks if not t.get("completed")]
    tasks = tasks[: max(1, min(int(limit), 200))]
    return {"count": len(tasks), "tasks": [_task_summary(t) for t in tasks]}


def habitica_tasks(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """List tasks of one type (one type per API request)."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        return json.dumps(
            _impl_tasks(
                c,
                str(args.get("task_type") or ""),
                bool(args.get("include_completed")),
                int(args.get("limit") or 50),
            ),
            ensure_ascii=False,
        )
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": "bad argument: " + str(exc)})


def _impl_task_create(c: HabiticaClient, body: dict, task_type: str) -> dict:
    if task_type not in TASK_TYPES:
        raise HabiticaError("task_type must be one of: " + ", ".join(TASK_TYPES))
    if not body.get("text") or not str(body["text"]).strip():
        raise HabiticaError("text is required")
    clean = _clean_task_body(body, task_type)
    clean["type"] = task_type
    created = c.create_task(clean)
    return {"created": True, "task": _task_summary(created)}


def habitica_task_create(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """Create a habit, daily, todo or reward."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        body = dict(args or {})
        body.pop("task_id", None)
        task_type = str(body.pop("task_type", ""))
        body.setdefault("difficulty", "easy")
        return json.dumps(
            _impl_task_create(c, body, task_type), ensure_ascii=False
        )
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})


def _impl_task_update(c: HabiticaClient, task_id: str, body: dict) -> dict:
    if not task_id:
        raise HabiticaError("task_id is required")
    clean = _clean_task_body(body)
    if not clean:
        raise HabiticaError("nothing to update: pass at least one field")
    clean.pop("task_id", None)
    updated = c.update_task(_require_uuid(task_id, "task_id"), clean)
    return {"updated": True, "task": _task_summary(updated)}


def habitica_task_update(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """Update an existing task (partial: only the fields you pass)."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        body = dict(args or {})
        task_id = str(body.pop("task_id", ""))
        return json.dumps(
            _impl_task_update(c, task_id, body), ensure_ascii=False
        )
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})


def _impl_task_delete(c: HabiticaClient, task_id: str) -> dict:
    if not task_id:
        raise HabiticaError("task_id is required")
    c.delete_task(_require_uuid(task_id, "task_id"))
    return {"deleted": True, "id": task_id}


def habitica_task_delete(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """Delete a Habitica task. Irreversible: deleted tasks cannot be restored."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        task_id = str(args.get("task_id") or "")
        return json.dumps(
            _impl_task_delete(c, task_id), ensure_ascii=False
        )
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})


def _impl_tick(c: HabiticaClient, task_id: str, direction: str) -> dict:
    if direction not in ("up", "down"):
        raise HabiticaError("direction must be up or down")
    if not task_id:
        raise HabiticaError("task_id is required")
    result = c.score_task(_require_uuid(task_id, "task_id"), direction) or {}
    # Verified live: stat changes arrive as a top-level "delta", new totals
    # as top-level hp/mp/gp/exp/lvl; "_tmp" only carries drops and quest
    # progress.
    drop = ((result.get("_tmp") or {}).get("drop")) or {}
    out = {
        "scored": direction,
        "id": task_id,
        "streak": result.get("streak"),
        "level": result.get("lvl"),
        "totals": {
            "hp": result.get("hp"),
            "mp": result.get("mp"),
            "gp": result.get("gp"),
            "exp": result.get("exp"),
        },
        "delta": result.get("delta") or {},
    }
    if drop:
        out["drop"] = {
            "type": drop.get("type"),
            "key": drop.get("key"),
            "dialog": drop.get("dialog"),
        }
    return out


def habitica_tick(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """Score a task up or down. Irreversible; affects streaks and stats."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        task_id = str(args.get("task_id") or "")
        direction = str(args.get("direction") or "up")
        return json.dumps(
            _impl_tick(c, task_id, direction), ensure_ascii=False
        )
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})


def _impl_tags(c: HabiticaClient) -> dict:
    tags = c.tags() or []
    return {
        "count": len(tags),
        "tags": [{"id": t.get("id"), "name": t.get("name")} for t in tags],
    }


def habitica_tags(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """List the account's tags with their ids (ids are what the task tools take)."""
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        return json.dumps(_impl_tags(c), ensure_ascii=False)
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})


def _impl_party(c: HabiticaClient, action: str, message: str, message_id: str, limit: int) -> dict:
    if action not in ("summary", "chat_list", "chat_post", "chat_delete"):
        raise HabiticaError(
            "action must be summary, chat_list, chat_post or chat_delete"
        )
    if action == "chat_delete":
        if not message_id:
            raise HabiticaError("message_id is required for chat_delete")
        _require_uuid(message_id, "message_id")
    party = c.party()
    if party is None:
        return {"party": None, "note": "This account is not in a party."}
    group_id = str(party.get("id") or party.get("_id") or "")
    out: dict[str, Any] = {
        "party": {
            "id": group_id,
            "name": party.get("name"),
            "memberCount": party.get("memberCount"),
            "privacy": party.get("privacy"),
        }
    }
    quest = party.get("quest") or {}
    if quest:
        out["quest"] = {
            "key": quest.get("key"),
            "active": quest.get("active"),
            "RSVPNeeded": quest.get("RSVPNeeded"),
            "progress": (quest.get("progress") or {}).get("hp"),
            "completed": quest.get("completed"),
        }
    if action == "summary":
        return out
    if action == "chat_list":
        limit = int(limit) if limit else int(load_settings()["chat_fetch_limit"] or 20)
        messages = c.party_chat(group_id, limit=limit)
        return {
            **out,
            "chat": [
                {
                    "id": m.get("id"),
                    "username": m.get("username"),
                    "text": m.get("text"),
                    "timestamp": m.get("timestamp"),
                    "system": (m.get("uuid") == "system"),
                }
                for m in messages
            ],
        }
    if action == "chat_post":
        if not message or not str(message).strip():
            raise HabiticaError("message is required for chat_post")
        posted = c.post_party_chat(group_id, str(message))
        posted_message = (posted or {}).get("message") or {}
        return {
            **out,
            "posted": True,
            "message_id": posted_message.get("id"),
        }
    # chat_delete: validated above, before any network call
    c.delete_party_chat(group_id, str(message_id))
    return {**out, "deleted": True, "message_id": message_id}


def habitica_party(args: dict, client: Optional[HabiticaClient] = None, **_ctx) -> str:
    """Party and quest status, with optional party chat posting.

    Actions: summary (party + quest state, default), chat_list (recent
    messages), chat_post (visible to party members; messages cannot be
    unsent except by deleting your own), chat_delete (removes one of your
    own messages; irreversible).
    """
    try:
        c = _make_client(client, float(_settings()["request_timeout_seconds"]))
        return json.dumps(
            _impl_party(
                c,
                str(args.get("action") or "summary"),
                str(args.get("message") or ""),
                str(args.get("message_id") or ""),
                int(args.get("limit") or 0),
            ),
            ensure_ascii=False,
        )
    except HabiticaError as exc:
        return json.dumps({"error": str(exc)})
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": "bad argument: " + str(exc)})


# ----------------------------------------------------------------------
# human-approval gate for irreversible / socially visible calls
# ----------------------------------------------------------------------


def _on_pre_tool_call(tool_name: str = "", args: Optional[dict] = None, **_ctx):
    """Escalate the destructive actions to Hermes' human-approval gate.

    Returning ``{"action": "approve", ...}`` routes the call through the
    approval flow (fail-closed: a denied, timed-out or errored gate blocks
    the call). Everything else returns None and runs normally.
    """
    a = args if isinstance(args, dict) else {}
    if tool_name == "habitica_task_delete":
        return {
            "action": "approve",
            "message": "hermes-habitica: permanently delete the Habitica task "
            + str(a.get("task_id") or "(unknown id)")
            + "? This cannot be undone.",
        }
    if tool_name == "habitica_party":
        action = a.get("action")
        if action == "chat_post":
            return {
                "action": "approve",
                "message": "hermes-habitica: post this message to your party "
                "chat (visible to party members)?",
            }
        if action == "chat_delete":
            return {
                "action": "approve",
                "message": "hermes-habitica: permanently delete this party "
                "chat message? This cannot be undone.",
            }
    return None


def register_hooks(ctx) -> None:
    """Entry point for provides_hooks: called in every Hermes process."""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)


# ----------------------------------------------------------------------
# schemas (LLM-facing)
# ----------------------------------------------------------------------


def _params(properties: dict, required: list | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


STATUS_SCHEMA = {
    "name": "habitica_status",
    "description": (
        "Show the user's Habitica character: level, class, HP, MP, XP, gold, "
        "gems and the longest daily streaks. Read-only."
    ),
    "parameters": _params({}),
}

TASKS_SCHEMA = {
    "name": "habitica_tasks",
    "description": (
        "List Habitica tasks of one type (habit, daily, todo or reward) with "
        "streaks, values and checklists. Read-only. One type per call, "
        "matching the Habitica API."
    ),
    "parameters": _params(
        {
            "task_type": {
                "type": "string",
                "enum": list(TASK_TYPES),
                "description": "Which task list to show.",
            },
            "include_completed": {
                "type": "boolean",
                "description": "Include completed todos too (default false).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum tasks to return (default 50, max 200).",
            },
        },
        ["task_type"],
    ),
}

CREATE_SCHEMA = {
    "name": "habitica_task_create",
    "description": (
        "Create a Habitica task: habit, daily, todo or reward. Dailies take "
        "frequency/every_x/repeat_days; rewards take a gold value; any type "
        "takes a checklist and tag ids (habitica_tags lists them)."
    ),
    "parameters": _params(
        {
            "task_type": {
                "type": "string",
                "enum": list(TASK_TYPES),
                "description": "Task type to create.",
            },
            "text": {"type": "string", "description": "Task title."},
            "difficulty": {
                "type": "string",
                "enum": list(DIFFICULTIES),
                "description": "trivial, easy (default), medium or hard.",
            },
            "notes": {"type": "string", "description": "Optional notes."},
            "checklist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional checklist items.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tag ids to attach (ids, not names).",
            },
            "up": {"type": "boolean", "description": "Habit only: enable plus."},
            "down": {"type": "boolean", "description": "Habit only: enable minus."},
            "frequency": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "yearly"],
                "description": "Daily only.",
            },
            "every_x": {
                "type": "integer",
                "description": "Daily only: every X days/weeks/months.",
            },
            "repeat_days": {
                "type": "array",
                "items": {"type": "string", "enum": list(WEEKDAYS)},
                "description": "Daily only, weekly frequency: active weekday keys (su,m,t,w,th,f,s).",
            },
            "value": {
                "type": "number",
                "description": "Reward only: cost in gold.",
            },
        },
        ["task_type", "text"],
    ),
}

UPDATE_SCHEMA = {
    "name": "habitica_task_update",
    "description": (
        "Update an existing Habitica task. Partial: only the fields you pass "
        "change. Find ids with habitica_tasks first; deleted or renamed "
        "tasks are a common source of failed updates."
    ),
    "parameters": _params(
        {
            "task_id": {"type": "string", "description": "Task id to update."},
            "text": {"type": "string"},
            "difficulty": {"type": "string", "enum": list(DIFFICULTIES)},
            "notes": {"type": "string"},
            "checklist": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "up": {"type": "boolean"},
            "down": {"type": "boolean"},
            "frequency": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "yearly"],
            },
            "every_x": {"type": "integer"},
            "repeat_days": {
                "type": "array",
                "items": {"type": "string", "enum": list(WEEKDAYS)},
            },
            "value": {"type": "number"},
        },
        ["task_id"],
    ),
}

DELETE_SCHEMA = {
    "name": "habitica_task_delete",
    "description": (
        "Delete a Habitica task by id. Irreversible: deleted tasks and "
        "their streak history cannot be restored. Confirm the id with "
        "habitica_tasks first. Requires human approval before it runs."
    ),
    "parameters": _params(
        {"task_id": {"type": "string", "description": "Task id to delete."}},
        ["task_id"],
    ),
}

TICK_SCHEMA = {
    "name": "habitica_tick",
    "description": TICK_SAFETY,
    "parameters": _params(
        {
            "task_id": {"type": "string", "description": "Task id to score."},
            "direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "up to complete or score a plus, down for a minus (default up).",
            },
        },
        ["task_id"],
    ),
}

TAGS_SCHEMA = {
    "name": "habitica_tags",
    "description": "List the account's Habitica tags with their ids. Read-only.",
    "parameters": _params({}),
}

PARTY_SCHEMA = {
    "name": "habitica_party",
    "description": (
        "Habitica party: summary (party and quest state, default), chat_list "
        "(recent messages), chat_post (sends a message visible to party "
        "members; it cannot be unsent except by deleting your own message) "
        "or chat_delete (removes one of your own messages, irreversible). "
        "chat_post and chat_delete require human approval before they run."
    ),
    "parameters": _params(
        {
            "action": {
                "type": "string",
                "enum": ["summary", "chat_list", "chat_post", "chat_delete"],
                "description": "What to do (default summary).",
            },
            "message": {"type": "string", "description": "chat_post only."},
            "message_id": {
                "type": "string",
                "description": "chat_delete only: id of your own message.",
            },
            "limit": {
                "type": "integer",
                "description": "chat_list only: how many messages (default from settings).",
            },
        },
    ),
}

TOOLS = (
    (STATUS_SCHEMA, habitica_status),
    (TASKS_SCHEMA, habitica_tasks),
    (CREATE_SCHEMA, habitica_task_create),
    (UPDATE_SCHEMA, habitica_task_update),
    (DELETE_SCHEMA, habitica_task_delete),
    (TICK_SCHEMA, habitica_tick),
    (TAGS_SCHEMA, habitica_tags),
    (PARTY_SCHEMA, habitica_party),
)


def register_tools(ctx) -> None:
    """Entry point for provides_tools: called in every Hermes process."""
    for schema, handler in TOOLS:
        ctx.register_tool(
            name=schema["name"],
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            description=schema["description"],
            emoji="🐉",
        )
