"""Habitica agent tools: status, tasks CRUD, scoring, tags and party.

Registered in every Hermes process via ``provides_tools`` in plugin.yaml.
Handlers build a lazily-resolved ``HabiticaClient`` unless the caller
injects one, which is how the unit tests drive everything through a fake
transport with no live network.

Safety posture, stated in the tool descriptions because that is the only
steering the model sees:

- scoring a task is an irreversible write that changes streaks, stats and
  gold, so ``habitica_tick`` says to use it only when the user names the
  task or explicitly asks to tick it
- deletes are irreversible, so ``habitica_task_delete`` and the chat
  delete action say so plainly
- party chat posts are visible to party members
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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


# ----------------------------------------------------------------------
# tool handlers (client injectable for tests)
# ----------------------------------------------------------------------


def _make_client(client: Optional[HabiticaClient], timeout: float) -> HabiticaClient:
    if client is not None:
        return client
    return HabiticaClient.resolve(timeout=timeout)


def habitica_status(client: Optional[HabiticaClient] = None, **_kwargs) -> dict:
    """Character stats, balances and a streaks digest."""
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
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


def habitica_tasks(
    task_type: str,
    include_completed: bool = False,
    limit: int = 50,
    client: Optional[HabiticaClient] = None,
    **_kwargs,
) -> dict:
    """List tasks of one type (one type per API request)."""
    if task_type not in TASK_TYPES:
        raise HabiticaError("task_type must be one of: " + ", ".join(TASK_TYPES))
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
    tasks = c.tasks(task_type)
    if not include_completed:
        tasks = [t for t in tasks if not t.get("completed")]
    tasks = tasks[: max(1, min(int(limit), 200))]
    return {"count": len(tasks), "tasks": [_task_summary(t) for t in tasks]}


def habitica_task_create(
    task_type: str,
    text: str,
    difficulty: str = "easy",
    notes: str = "",
    checklist: Optional[list] = None,
    tags: Optional[list] = None,
    up: Optional[bool] = None,
    down: Optional[bool] = None,
    frequency: Optional[str] = None,
    every_x: Optional[int] = None,
    repeat_days: Optional[list] = None,
    value: Optional[float] = None,
    client: Optional[HabiticaClient] = None,
    **_kwargs,
) -> dict:
    """Create a habit, daily, todo or reward."""
    if task_type not in TASK_TYPES:
        raise HabiticaError("task_type must be one of: " + ", ".join(TASK_TYPES))
    if not text or not str(text).strip():
        raise HabiticaError("text is required")
    body = _clean_task_body(
        {
            "text": text,
            "notes": notes,
            "difficulty": difficulty,
            "tags": tags,
            "checklist": checklist,
            "up": up,
            "down": down,
            "frequency": frequency,
            "every_x": every_x,
            "repeat_days": repeat_days,
            "value": value,
        },
        task_type,
    )
    body["type"] = task_type
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
    created = c.create_task(body)
    return {"created": True, "task": _task_summary(created)}


def habitica_task_update(
    task_id: str,
    text: Optional[str] = None,
    difficulty: Optional[str] = None,
    notes: Optional[str] = None,
    checklist: Optional[list] = None,
    tags: Optional[list] = None,
    up: Optional[bool] = None,
    down: Optional[bool] = None,
    frequency: Optional[str] = None,
    every_x: Optional[int] = None,
    repeat_days: Optional[list] = None,
    value: Optional[float] = None,
    client: Optional[HabiticaClient] = None,
    **_kwargs,
) -> dict:
    """Update an existing task (partial: only the fields you pass)."""
    if not task_id:
        raise HabiticaError("task_id is required")
    body = _clean_task_body(
        {
            "text": text,
            "notes": notes,
            "difficulty": difficulty,
            "tags": tags,
            "checklist": checklist,
            "up": up,
            "down": down,
            "frequency": frequency,
            "every_x": every_x,
            "repeat_days": repeat_days,
            "value": value,
        }
    )
    if not body:
        raise HabiticaError("nothing to update: pass at least one field")
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
    updated = c.update_task(task_id, body)
    return {"updated": True, "task": _task_summary(updated)}


def habitica_task_delete(
    task_id: str, client: Optional[HabiticaClient] = None, **_kwargs
) -> dict:
    """Delete a Habitica task. Irreversible: deleted tasks cannot be restored."""
    if not task_id:
        raise HabiticaError("task_id is required")
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
    c.delete_task(task_id)
    return {"deleted": True, "id": task_id}


def habitica_tick(
    task_id: str,
    direction: str = "up",
    client: Optional[HabiticaClient] = None,
    **_kwargs,
) -> dict:
    """Score a task up or down. Irreversible; affects streaks and stats."""
    if direction not in ("up", "down"):
        raise HabiticaError("direction must be up or down")
    if not task_id:
        raise HabiticaError("task_id is required")
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
    result = c.score_task(task_id, direction) or {}
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


def habitica_tags(client: Optional[HabiticaClient] = None, **_kwargs) -> dict:
    """List the account's tags with their ids (ids are what the task tools take)."""
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
    tags = c.tags() or []
    return {
        "count": len(tags),
        "tags": [{"id": t.get("id"), "name": t.get("name")} for t in tags],
    }


def habitica_party(
    action: str = "summary",
    message: str = "",
    message_id: str = "",
    limit: int = 0,
    client: Optional[HabiticaClient] = None,
    **_kwargs,
) -> dict:
    """Party and quest status, with optional party chat posting.

    Actions: summary (party + quest state, default), chat_list (recent
    messages), chat_post (visible to party members; messages cannot be
    unsent except by deleting your own), chat_delete (removes one of your
    own messages; irreversible).
    """
    if action not in ("summary", "chat_list", "chat_post", "chat_delete"):
        raise HabiticaError(
            "action must be summary, chat_list, chat_post or chat_delete"
        )
    settings = load_settings()
    c = _make_client(client, float(settings["request_timeout_seconds"]))
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
        limit = int(limit) if limit else int(settings["chat_fetch_limit"] or 20)
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
    # chat_delete
    if not message_id:
        raise HabiticaError("message_id is required for chat_delete")
    c.delete_party_chat(group_id, str(message_id))
    return {**out, "deleted": True, "message_id": message_id}


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
        "habitica_tasks first."
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
        "or chat_delete (removes one of your own messages, irreversible)."
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
