"""CG's own Session for the real-CG tests, watched: after each message it keeps the reply, the open request's legal
orders, the anomalies and the corrections, as plain copies in `records` (one per Session made). The Session itself
stays on CG's thread: its Rust core may not be touched or dropped elsewhere."""

from __future__ import annotations

from typing import ClassVar

from champions_gym.live import Session


class Watched(Session):
    records: ClassVar[list[dict]] = []

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.record: dict = {"seen": [], "anomalies": [], "corrections": []}
        Watched.records.append(self.record)

    def handle(self, message):
        reply = super().handle(message)
        kind = message.get("type") if isinstance(message, dict) else None
        legal = self.legal_orders() if kind == "decision_request" else None
        self.record["seen"].append({"message": message, "reply": reply, "legal": legal})
        self.record["anomalies"], self.record["corrections"] = self.anomalies, self.corrections
        return reply
