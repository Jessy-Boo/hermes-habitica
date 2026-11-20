# hermes-habitica

Manage a [Habitica](https://habitica.com) character from
[Hermes Agent](https://hermes-agent.nousresearch.com): view stats and
streaks, tick habits and dailies, create and edit tasks and rewards, and
follow party and quest progress, with optional party chat posting.

Not affiliated with or endorsed by Habitica. An independent implementation
of the public [Habitica API v3](https://habitica.com/api/v3).

## Install

```bash
hermes plugins install hermes-habitica
hermes plugins enable hermes-habitica
```

## Credentials

Two secrets are required, both from the Habitica web app's
**Settings, Site, and Mobile Data** page (API column):

- `HABITICA_USER_ID`: the user ID shown in the API column
- `HABITICA_API_TOKEN`: the API token (Show API Token)

The installer links to that page and stores both values in Hermes' scoped
secret store. The plugin reads them only through
`agent.secret_scope.get_secret`; it never reads environment variables or
`.env` files (a unit test proves env values are ignored).

```bash
hermes config set plugins.entries.hermes-habitica.requires_env.HABITICA_USER_ID <value>
hermes config set plugins.entries.hermes-habitica.requires_env.HABITICA_API_TOKEN <value>
```

## Tools

| Tool | What it does |
| --- | --- |
| `habitica_status` | Level, class, HP, MP, XP, gold, gems and the longest daily streaks |
| `habitica_tasks` | List one task type with streaks, values and checklists |
| `habitica_task_create` | Create habits, dailies, todos and rewards (checklists, tags, schedules, gold values) |
| `habitica_task_update` | Partial update of any task field |
| `habitica_task_delete` | Delete a task (irreversible) |
| `habitica_tick` | Score a task up or down (irreversible; affects streaks) |
| `habitica_tags` | List tags with their ids |
| `habitica_party` | Party and quest status; list, post to and clean up party chat |

## What this plugin does on your machine

- **Network**: every call goes to `habitica.com/api/v3` with your stored
  credentials. Nothing else is contacted, and no telemetry is sent.
- **Writes are real state changes**: ticks, edits and deletes are
  irreversible on Habitica's side; a tick permanently affects streaks,
  stats and gold. The `habitica_tick` tool description instructs the model
  to tick only when the user names the task or explicitly asks.
- **Party chat**: `chat_post` sends a message visible to party members; it
  cannot be unsent except by deleting your own message (`chat_delete`).
- **Credentials**: stored in Hermes' scoped secret store, sent only to
  `habitica.com` over HTTPS, never logged.
- **No files are read or written** on the host; no background processes.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `request_timeout_seconds` | `30` | HTTP timeout for habitica.com calls |
| `chat_fetch_limit` | `20` | Recent messages returned by the `chat_list` action |

HTTP 429/503 responses are retried up to three times with an exponential
backoff capped at 30 seconds (the server's `Retry-After` header wins).

## Developing

```bash
python3 -m unittest discover -s tests -t .
```

runs the unit tests (stdlib `unittest`, no live network: urllib is faked).

Layout:

- `client.py`: stdlib-only API v3 client (headers, backoff, verified
  endpoint behaviour)
- `tools.py`: tool handlers and LLM-facing schemas
- `settings.py`: config glue
- `tests/`: mocked-transport unit tests

## License

MIT
