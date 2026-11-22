"""hermes-habitica: full-suite Habitica tools for Hermes Agent.

Tools-only plugin: everything lives in ``tools.py`` and is registered
eagerly in every Hermes process via ``provides_tools`` in plugin.yaml.
There is no gateway platform code, no hooks and no middleware.

The shared HTTP client (``client.py``) is stdlib-only: no third-party
runtime dependency is declared, so ``--install-deps`` installs nothing and
the supply chain is Python's own.
"""

__version__ = "1.0.1"


def register(ctx) -> None:
    """Plugin entry point: registers the Habitica agent tools and the
    human-approval gate for the irreversible actions.

    The same registrations run through ``provides_tools`` /
    ``provides_hooks`` in every Hermes process; ``ctx.register_tool``
    replaces by name, so both paths together stay idempotent.
    """
    from .tools import register_hooks, register_tools

    register_tools(ctx)
    register_hooks(ctx)
