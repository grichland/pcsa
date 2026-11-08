"""The four ways things go wrong, and what each one means for the loop.

Transient   -> retry in place (a screencap failed, a tap did not land).
Recoverable -> the session restarts from whatever screen the phone shows (stuck, unknown screen too long).
SafetyStop  -> the run ends. Raised only when the next action could leave Ranked Singles or lose the account
               something (forfeit, quit, edit team, purchase, rating dialog). Never caught by handlers.
ConfigError -> refused before the phone is touched.
"""


class Recoverable(Exception):
    """The session should be restarted; the run continues."""


class Transient(Recoverable):
    """The operation can be retried in place; escalates to Recoverable after the configured attempts."""


class SafetyStop(Exception):
    """Fatal by design: the run ends and the reason is recorded."""


class ConfigError(Exception):
    """Configuration or tooling is wrong; nothing was done."""
