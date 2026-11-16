"""The four ways things go wrong, and what each means for the loop.

Transient: retry in place. Recoverable: restart the session. SafetyStop: end the run (the next action could leave
Ranked
Singles or cost the account something; never caught by handlers). ConfigError: refused before the phone is touched.
"""


class Recoverable(Exception):
    """The session should be restarted; the run continues."""


class Transient(Recoverable):
    """The operation can be retried in place; escalates to Recoverable after the configured attempts."""


class SafetyStop(Exception):
    """Fatal by design: the run ends and the reason is recorded."""


class ConfigError(Exception):
    """Configuration or tooling is wrong; nothing was done."""
