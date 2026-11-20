"""Tests for hermes-habitica. No live network: urllib is faked.

Every test drives the plugin through a fake ``urlopen`` that records the
requests it receives and replays canned responses, so the gate runs fast,
deterministically and without touching anyone's Habitica account.
"""

import importlib.util
import io
import json
import os
import pathlib
import sys
import unittest
import urllib.error
from unittest import mock

PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load_package(name: str):
    """Import the plugin directory as a package despite the dashed repo name."""
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pkg = _load_package("hhc_test_pkg")
client_mod = importlib.import_module("hhc_test_pkg.client")
tools = importlib.import_module("hhc_test_pkg.tools")
HabiticaClient = client_mod.HabiticaClient
HabiticaCredentialsError = client_mod.HabiticaCredentialsError
HabiticaError = client_mod.HabiticaError


def api_response(payload, status=200):
    """A fake http.client.HTTPResponse-like object."""
    body = json.dumps(payload).encode("utf-8")

    class FakeResponse(io.BytesIO):
        code = status

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            self.close()

        def getcode(self):
            return status

    return FakeResponse(body)


def http_error(status, headers=None, body=b'{"error":"x","message":"boom"}'):
    hdrs = mock.Mock()
    hdrs.get.side_effect = lambda name, default=None: (headers or {}).get(
        name, default
    )
    return urllib.error.HTTPError(
        "https://habitica.com/api/v3", status, "err", hdrs, io.BytesIO(body)
    )


class FakeUrlopen:
    """Records requests; replies from a script of payloads/errors."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        item = self.script.pop(0) if self.script else {}
        if isinstance(item, Exception):
            raise item
        return api_response(item.get("payload", {"success": True, "data": {}}))

    def last_url(self):
        return self.requests[-1].full_url

    def last_body(self):
        data = self.requests[-1].data
        return data.decode("utf-8") if data else ""


def make_client(script):
    fake = FakeUrlopen(script)
    c = HabiticaClient(
        user_id="uid-test",
        api_token="tok-test",
        timeout=5,
        sleeper=lambda _s: None,
        urlopen=fake,
    )
    return c, fake


USER_PAYLOAD = {
    "success": True,
    "data": {
        "profile": {"name": "Tester"},
        "stats": {
            "lvl": 5,
            "class": "warrior",
            "hp": 30,
            "maxHealth": 50,
            "mp": 10,
            "maxMP": 38,
            "exp": 100,
            "toNextLevel": 120,
            "gp": 77.5,
        },
        "balance": {"gems": 12, "hourglasses": 0},
    },
}


class ClientTests(unittest.TestCase):
    def test_required_headers_are_sent(self):
        c, fake = make_client([{"payload": USER_PAYLOAD}])
        c.user()
        req = fake.requests[0]
        self.assertEqual(req.get_header("X-api-user"), "uid-test")
        self.assertEqual(req.get_header("X-api-key"), "tok-test")
        self.assertTrue(req.get_header("X-client"))

    def test_success_false_raises_with_vendor_message(self):
        c, _ = make_client(
            [{"payload": {"success": False, "error": "NotFound", "message": "Not found."}}]
        )
        with self.assertRaises(HabiticaError) as ctx:
            c.get("group")
        self.assertIn("Not found.", str(ctx.exception))

    def test_one_type_per_request_uses_plural_form(self):
        c, fake = make_client([{"payload": {"success": True, "data": []}}])
        c.tasks("daily")
        url = fake.last_url()
        self.assertIn("type=dailys", url)
        self.assertNotIn("%2C", url)  # no comma-combined types

    def test_429_retries_with_backoff_then_succeeds(self):
        sleeps: list[float] = []
        fake = FakeUrlopen(
            [
                http_error(429, headers={"Retry-After": "1"}),
                http_error(429),
                {"payload": {"success": True, "data": {"ok": True}}},
            ]
        )
        c = HabiticaClient(
            user_id="u", api_token="t", timeout=5,
            sleeper=sleeps.append, urlopen=fake,
        )
        data = c.get("tags")
        self.assertEqual(data, {"ok": True})
        self.assertEqual(len(fake.requests), 3)
        self.assertTrue(sleeps)  # backoff actually happened

    def test_429_gives_up_after_three_attempts(self):
        fake = FakeUrlopen(
            [http_error(429), http_error(429), http_error(429)]
        )
        c = HabiticaClient(
            user_id="u", api_token="t", timeout=5,
            sleeper=lambda _s: None, urlopen=fake,
        )
        with self.assertRaises(HabiticaError):
            c.get("tags")

    def test_http_error_surfaces_vendor_message(self):
        c, _ = make_client(
            [http_error(401, body=b'{"error":"invalid_credentials","message":"bad"}')]
        )
        with self.assertRaises(HabiticaError) as ctx:
            c.get("user")
        self.assertIn("bad", str(ctx.exception))


class CredentialRouteTests(unittest.TestCase):
    def test_env_values_are_ignored_even_when_set(self):
        env = {
            "HABITICA_USER_ID": "env-user",
            "HABITICA_API_TOKEN": "env-token",
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch.object(
                client_mod, "get_secret", side_effect=KeyError("nope")
            ):
                with self.assertRaises(HabiticaCredentialsError):
                    client_mod._credentials()
        # and the env route is never consulted anywhere in the client module
        with open(os.path.join(PLUGIN_DIR, "client.py")) as fh:
            source = fh.read()
        self.assertNotIn("os.environ", source)
        # built by concatenation so the cleanliness linter's dotenv rule
        # does not match this negative assertion itself
        self.assertNotIn("load_" + "dotenv", source)
        self.assertNotIn("getenv", source)


class ChatBodyTests(unittest.TestCase):
    def test_chat_post_uses_urlencoded_body_not_query(self):
        c, fake = make_client(
            [
                {"payload": {"success": True, "data": [{"id": "g1", "name": "P"}]}},
                {
                    "payload": {
                        "success": True,
                        "data": {"message": {"id": "m1", "text": "hello"}},
                    }
                },
            ]
        )
        out = tools.habitica_party(
            action="chat_post", message="hello world", client=c
        )
        self.assertTrue(out["posted"])
        url = fake.last_url()
        self.assertNotIn("hello", url)  # message never rides the query string
        self.assertIn("message=hello+world", fake.last_body())


class HandlerTests(unittest.TestCase):
    def test_status_shapes_stats_and_streaks(self):
        c, fake = make_client(
            [
                {"payload": USER_PAYLOAD},
                {
                    "payload": {
                        "success": True,
                        "data": [
                            {"id": "t1", "text": "A", "streak": 302},
                            {"id": "t2", "text": "B", "streak": 78},
                            {"id": "t3", "text": "C", "streak": 0},
                        ],
                    }
                },
            ]
        )
        out = tools.habitica_status(client=c)
        self.assertEqual(out["level"], 5)
        self.assertEqual(out["gems"], 12)
        self.assertEqual(out["top_daily_streaks"][0]["streak"], 302)

    def test_status_handles_bare_number_balance(self):
        bare = json.loads(json.dumps(USER_PAYLOAD))
        bare["data"]["balance"] = 0
        c, _ = make_client(
            [{"payload": bare}, {"payload": {"success": True, "data": []}}]
        )
        out = tools.habitica_status(client=c)
        self.assertEqual(out["gems"], 0)

    def test_tasks_filters_completed_and_validates_type(self):
        c, fake = make_client(
            [
                {
                    "payload": {
                        "success": True,
                        "data": [
                            {"id": "t1", "type": "todo", "text": "open", "completed": False},
                            {"id": "t2", "type": "todo", "text": "done", "completed": True},
                        ],
                    }
                }
            ]
        )
        out = tools.habitica_tasks("todo", client=c)
        self.assertEqual(out["count"], 1)
        with self.assertRaises(HabiticaError):
            tools.habitica_tasks("bogus", client=c)

    def test_create_sends_typed_body(self):
        c, fake = make_client(
            [{"payload": {"success": True, "data": {"id": "new1", "type": "daily", "text": "X"}}}]
        )
        out = tools.habitica_task_create(
            "daily", "X", frequency="weekly", repeat_days=["m", "f"], client=c
        )
        self.assertTrue(out["created"])
        body = json.loads(fake.last_body())
        self.assertEqual(body["type"], "daily")
        self.assertEqual(body["priority"], 1.0)  # API field, not "difficulty"
        self.assertEqual(body["repeat"], {"su": False, "m": True, "t": False,
                                          "w": False, "th": False, "f": True, "s": False})

    def test_create_rejects_bad_difficulty_and_repeat_days(self):
        c, _ = make_client([])
        with self.assertRaises(HabiticaError):
            tools.habitica_task_create("habit", "X", difficulty="impossible", client=c)
        with self.assertRaises(HabiticaError):
            tools.habitica_task_create("daily", "X", repeat_days=["monday"], client=c)

    def test_update_partial_body(self):
        c, fake = make_client(
            [{"payload": {"success": True, "data": {"id": "t1", "text": "New"}}}]
        )
        tools.habitica_task_update("t1", text="New", client=c)
        body = json.loads(fake.last_body())
        self.assertEqual(body, {"text": "New"})

    def test_update_with_nothing_raises(self):
        c, _ = make_client([])
        with self.assertRaises(HabiticaError):
            tools.habitica_task_update("t1", client=c)

    def test_tick_returns_delta_totals_and_streak(self):
        c, fake = make_client(
            [
                {
                    "payload": {
                        "success": True,
                        "data": {
                            "streak": 303,
                            "lvl": 6,
                            "hp": 46,
                            "mp": 30,
                            "gp": 80,
                            "exp": 105,
                            "delta": {"hp": 0, "mp": 2.5, "gp": 1.2, "exp": 5},
                            "_tmp": {"drop": {"type": "HatchingPotion", "key": "Base"}},
                        },
                    }
                }
            ]
        )
        out = tools.habitica_tick("t1", "up", client=c)
        self.assertEqual(out["scored"], "up")
        self.assertEqual(out["streak"], 303)
        self.assertEqual(out["delta"]["exp"], 5)
        self.assertEqual(out["totals"]["exp"], 105)
        self.assertEqual(out["drop"]["key"], "Base")
        self.assertIn("/score/up", fake.last_url())

    def test_update_response_reports_priority_as_difficulty(self):
        c, _ = make_client(
            [{"payload": {"success": True, "data": {"id": "t1", "text": "New", "priority": 1.5}}}]
        )
        out = tools.habitica_task_update("t1", text="New", client=c)
        self.assertEqual(out["task"]["difficulty"], 1.5)

    def test_party_summary_and_no_party(self):
        c, _ = make_client(
            [{"payload": {"success": True, "data": []}}]
        )
        out = tools.habitica_party("summary", client=c)
        self.assertIsNone(out["party"])

        c2, _ = make_client(
            [
                {
                    "payload": {
                        "success": True,
                        "data": [
                            {
                                "id": "g1",
                                "name": "Solo",
                                "memberCount": 1,
                                "privacy": "private",
                                "quest": {"key": "dustbunnies", "active": True},
                            }
                        ],
                    }
                }
            ]
        )
        out = tools.habitica_party("summary", client=c2)
        self.assertEqual(out["quest"]["key"], "dustbunnies")

    def test_party_chat_list_maps_fields(self):
        c, _ = make_client(
            [
                {"payload": {"success": True, "data": [{"id": "g1"}]}},
                {
                    "payload": {
                        "success": True,
                        "data": [
                            {"id": "m1", "username": "jessy-boo", "text": "hi",
                             "timestamp": "2026-11-20T00:00:00Z", "uuid": "u1"},
                            {"id": "m2", "username": "system", "text": "boss fight",
                             "timestamp": "2026-11-20T00:01:00Z", "uuid": "system"},
                        ],
                    }
                },
            ]
        )
        out = tools.habitica_party("chat_list", client=c)
        self.assertFalse(out["chat"][0]["system"])
        self.assertTrue(out["chat"][1]["system"])


class SchemaTests(unittest.TestCase):
    def test_all_tools_have_valid_schemas(self):
        names = []
        for schema, handler in tools.TOOLS:
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertIn("parameters", schema)
            self.assertEqual(schema["parameters"]["type"], "object")
            self.assertTrue(callable(handler))
            names.append(schema["name"])
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names),
            {
                "habitica_status",
                "habitica_tasks",
                "habitica_task_create",
                "habitica_task_update",
                "habitica_task_delete",
                "habitica_tick",
                "habitica_tags",
                "habitica_party",
            },
        )

    def test_provides_tools_matches_manifest(self):
        import re

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "plugin.yaml")) as fh:
            manifest = fh.read()
        for schema, _handler in tools.TOOLS:
            self.assertIn(schema["name"], manifest)

    def test_tick_schema_carries_safety_phrasing(self):
        self.assertIn(
            "Only call when the user names the task or explicitly asks",
            tools.TICK_SCHEMA["description"],
        )
        self.assertIn("irreversible", tools.TICK_SCHEMA["description"])

    def test_delete_schema_says_irreversible(self):
        self.assertIn("Irreversible", tools.DELETE_SCHEMA["description"])

    def test_no_em_dashes_anywhere(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("plugin.yaml", "README.md", "tools.py", "client.py",
                     "settings.py", "__init__.py"):
            with open(os.path.join(root, name)) as fh:
                text = fh.read()
            self.assertNotIn("\u2014", text, name)  # em dash
            self.assertNotIn("\u2013", text, name)  # en dash


if __name__ == "__main__":
    unittest.main()
