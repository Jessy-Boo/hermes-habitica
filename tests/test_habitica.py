"""Tests for hermes-habitica. No live network: urllib is faked.

Two layers are covered:

- client-level: headers, one-type-per-request, backoff, credential
  resolution, redirect safety, id quoting
- tool-handler level: the public handlers follow the Hermes plugin tool
  contract (``handler(args, **kwargs) -> str`` with dispatcher context
  kwargs injected by signature), so every test calls them exactly the
  way the dispatcher does, including the context kwargs.
"""

import importlib.util
import io
import inspect
import json
import os
import pathlib
import sys
import unittest
import urllib.error
import urllib.request
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

# a valid UUID every id-based test can use
TASK_ID = "123e4567-e89b-12d3-a456-426614174000"
MSG_ID = "00000000-1111-2222-3333-444444444444"

# the context kwargs the Hermes dispatcher injects by signature
INJECTED = dict(task_id="hermes-task", session_id="hermes-session",
                user_task="hermes-user-task", parent_agent="hermes-parent",
                tool_call_id="hermes-call", turn_id="hermes-turn")


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


def jout(result):
    """Parse a handler's JSON string result."""
    return json.loads(result)


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

    def test_ids_are_quoted_into_path_segments(self):
        c, fake = make_client([{"payload": {"success": True, "data": {}}}])
        c.delete_task("../other")
        self.assertIn("tasks/..%2Fother", fake.last_url())

    def test_redirect_off_habitica_is_refused(self):
        handler = client_mod._SafeRedirectHandler()
        req = urllib.request.Request(
            "https://habitica.com/api/v3/user", headers={"X-api-key": "secret"}
        )
        result = handler.redirect_request(
            req, fp=None, code=302, msg="found", headers=None,
            newurl="https://evil.example/api",
        )
        self.assertIsNone(result)

    def test_same_host_redirect_drops_auth_headers(self):
        handler = client_mod._SafeRedirectHandler()
        req = urllib.request.Request(
            "https://habitica.com/api/v3/old",
            headers={"X-api-key": "secret", "X-api-user": "uid"},
        )
        result = handler.redirect_request(
            req, fp=None, code=302, msg="found", headers=None,
            newurl="https://habitica.com/api/v3/new",
        )
        self.assertIsNotNone(result)
        self.assertIsNone(result.get_header("X-api-key"))
        self.assertIsNone(result.get_header("X-api-user"))


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

    def test_none_and_empty_secret_results_are_missing(self):
        # get_secret returns default (None) on a scoped miss instead of
        # raising; that must read as missing, not as the literal "None"
        with mock.patch.object(client_mod, "get_secret", return_value=None):
            with self.assertRaises(HabiticaCredentialsError):
                client_mod._credentials()
        with mock.patch.object(client_mod, "get_secret", return_value="   "):
            with self.assertRaises(HabiticaCredentialsError):
                client_mod._credentials()


class ChatBodyTests(unittest.TestCase):
    def test_chat_post_uses_urlencoded_body_not_query(self):
        c, fake = make_client(
            [
                {"payload": {"success": True, "data": [{"id": "g1", "name": "P"}]}},
                {
                    "payload": {
                        "success": True,
                        "data": {"message": {"id": MSG_ID, "text": "hello"}},
                    }
                },
            ]
        )
        out = jout(tools.habitica_party(
            {"action": "chat_post", "message": "hello world"}, client=c
        ))
        self.assertTrue(out["posted"])
        url = fake.last_url()
        self.assertNotIn("hello", url)  # message never rides the query string
        self.assertIn("message=hello+world", fake.last_body())


class HandlerContractTests(unittest.TestCase):
    """The Hermes plugin tool contract: handler(args, **kwargs) -> str."""

    def test_every_handler_accepts_injected_context_kwargs(self):
        for schema, handler in tools.TOOLS:
            with self.subTest(tool=schema["name"]):
                result = handler({}, **INJECTED)  # no client: fails closed
                self.assertIsInstance(result, str, schema["name"])
                parsed = json.loads(result)
                self.assertIn("error", parsed, schema["name"])
                self.assertIn("credentials are not configured", parsed["error"])

    def test_handler_signatures_do_not_redeclare_context_names(self):
        for schema, handler in tools.TOOLS:
            params = set(inspect.signature(handler).parameters)
            self.assertNotIn(
                "session_id", params, schema["name"]
            )  # args is the only meaningful param; task_id must not be declared

    def test_handlers_return_strings_not_dicts(self):
        c, _ = make_client(
            [{"payload": USER_PAYLOAD}, {"payload": {"success": True, "data": []}},
             {"payload": {"success": True, "data": [{"id": "g1"}]}}]
        )
        for handler, args in (
            (tools.habitica_status, {}),
            (tools.habitica_tags, {}),
            (tools.habitica_party, {"action": "summary"}),
        ):
            result = handler(args, client=c, **INJECTED)
            self.assertIsInstance(result, str)


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
        out = jout(tools.habitica_status({}, client=c))
        self.assertEqual(out["level"], 5)
        self.assertEqual(out["gems"], 12)
        self.assertEqual(out["top_daily_streaks"][0]["streak"], 302)

    def test_status_handles_bare_number_balance(self):
        bare = json.loads(json.dumps(USER_PAYLOAD))
        bare["data"]["balance"] = 0
        c, _ = make_client(
            [{"payload": bare}, {"payload": {"success": True, "data": []}}]
        )
        out = jout(tools.habitica_status({}, client=c))
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
        out = jout(tools.habitica_tasks({"task_type": "todo"}, client=c))
        self.assertEqual(out["count"], 1)
        out = jout(tools.habitica_tasks({"task_type": "bogus"}, client=c))
        self.assertIn("error", out)

    def test_create_sends_typed_body(self):
        c, fake = make_client(
            [{"payload": {"success": True, "data": {"id": "new1", "type": "daily", "text": "X"}}}]
        )
        out = jout(tools.habitica_task_create(
            {
                "task_type": "daily",
                "text": "X",
                "frequency": "weekly",
                "repeat_days": ["m", "f"],
            },
            client=c,
        ))
        self.assertTrue(out["created"])
        body = json.loads(fake.last_body())
        self.assertEqual(body["type"], "daily")
        self.assertEqual(body["priority"], 1.0)  # API field, not "difficulty"
        self.assertEqual(body["repeat"], {"su": False, "m": True, "t": False,
                                          "w": False, "th": False, "f": True, "s": False})

    def test_create_rejects_bad_difficulty_and_repeat_days(self):
        c, _ = make_client([])
        out = jout(tools.habitica_task_create(
            {"task_type": "habit", "text": "X", "difficulty": "impossible"}, client=c
        ))
        self.assertIn("error", out)
        out = jout(tools.habitica_task_create(
            {"task_type": "daily", "text": "X", "repeat_days": ["monday"]}, client=c
        ))
        self.assertIn("error", out)

    def test_update_partial_body(self):
        c, fake = make_client(
            [{"payload": {"success": True, "data": {"id": TASK_ID, "text": "New"}}}]
        )
        out = jout(tools.habitica_task_update(
            {"task_id": TASK_ID, "text": "New"}, client=c
        ))
        self.assertTrue(out["updated"])
        body = json.loads(fake.last_body())
        self.assertEqual(body, {"text": "New"})

    def test_update_with_nothing_returns_error(self):
        c, _ = make_client([])
        out = jout(tools.habitica_task_update({"task_id": TASK_ID}, client=c))
        self.assertIn("error", out)

    def test_non_uuid_ids_are_rejected_before_any_network_call(self):
        c, fake = make_client([])
        for handler, args in (
            (tools.habitica_tick, {"task_id": "../other", "direction": "up"}),
            (tools.habitica_task_update, {"task_id": "x?query=1", "text": "hi"}),
            (tools.habitica_task_delete, {"task_id": "../../admin"}),
        ):
            out = jout(handler(args, client=c))
            self.assertIn("error", out)
            self.assertIn("valid UUID", out["error"])
        self.assertEqual(fake.requests, [])  # never reached the network

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
        out = jout(tools.habitica_tick(
            {"task_id": TASK_ID, "direction": "up"}, client=c
        ))
        self.assertEqual(out["scored"], "up")
        self.assertEqual(out["streak"], 303)
        self.assertEqual(out["delta"]["exp"], 5)
        self.assertEqual(out["totals"]["exp"], 105)
        self.assertEqual(out["drop"]["key"], "Base")
        self.assertIn("/score/up", fake.last_url())

    def test_update_response_reports_priority_as_difficulty(self):
        c, _ = make_client(
            [{"payload": {"success": True, "data": {"id": TASK_ID, "text": "New", "priority": 1.5}}}]
        )
        out = jout(tools.habitica_task_update(
            {"task_id": TASK_ID, "text": "New"}, client=c
        ))
        self.assertEqual(out["task"]["difficulty"], 1.5)

    def test_party_summary_and_no_party(self):
        c, _ = make_client([{"payload": {"success": True, "data": []}}])
        out = jout(tools.habitica_party({"action": "summary"}, client=c))
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
        out = jout(tools.habitica_party({"action": "summary"}, client=c2))
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
        out = jout(tools.habitica_party({"action": "chat_list"}, client=c))
        self.assertFalse(out["chat"][0]["system"])
        self.assertTrue(out["chat"][1]["system"])

    def test_party_chat_delete_requires_uuid_message_id(self):
        c, fake = make_client(
            [{"payload": {"success": True, "data": [{"id": "g1"}]}},
             {"payload": {"success": True, "data": {}}}]
        )
        out = jout(tools.habitica_party(
            {"action": "chat_delete", "message_id": MSG_ID}, client=c
        ))
        self.assertTrue(out["deleted"])
        out = jout(tools.habitica_party(
            {"action": "chat_delete", "message_id": "mid-1"}, client=c
        ))
        self.assertIn("error", out)


class ApprovalGateTests(unittest.TestCase):
    def test_destructive_actions_are_gated(self):
        self.assertEqual(
            tools._on_pre_tool_call(
                "habitica_task_delete", args={"task_id": TASK_ID}, **INJECTED
            )["action"],
            "approve",
        )
        for action in ("chat_post", "chat_delete"):
            result = tools._on_pre_tool_call(
                "habitica_party", args={"action": action}, **INJECTED
            )
            self.assertEqual(result["action"], "approve", action)

    def test_read_only_and_tick_actions_pass_ungated(self):
        for tool_name, args in (
            ("habitica_status", {}),
            ("habitica_tasks", {"task_type": "todo"}),
            ("habitica_tick", {"task_id": TASK_ID}),
            ("habitica_party", {"action": "summary"}),
            ("habitica_party", {"action": "chat_list"}),
            ("habitica_task_create", {"task_type": "todo", "text": "hi"}),
        ):
            self.assertIsNone(
                tools._on_pre_tool_call(tool_name, args=args, **INJECTED), tool_name
            )

    def test_gate_message_names_the_action(self):
        msg = tools._on_pre_tool_call(
            "habitica_task_delete", args={"task_id": TASK_ID}, **INJECTED
        )["message"]
        self.assertIn(TASK_ID, msg)
        self.assertIn("delete", msg.lower())

    def test_manifest_declares_the_hook(self):
        with open(os.path.join(PLUGIN_DIR, "plugin.yaml")) as fh:
            manifest = fh.read()
        self.assertIn("pre_tool_call", manifest)


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
        with open(os.path.join(PLUGIN_DIR, "plugin.yaml")) as fh:
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
        for name in ("plugin.yaml", "README.md", "tools.py", "client.py",
                     "settings.py", "__init__.py"):
            with open(os.path.join(PLUGIN_DIR, name)) as fh:
                text = fh.read()
            self.assertNotIn("\u2014", text, name)  # em dash
            self.assertNotIn("\u2013", text, name)  # en dash


if __name__ == "__main__":
    unittest.main()
