import pytest

from pcsa.common.clock import Deadline, FakeClock, SystemClock


def test_fake_clock_only_moves_when_told() -> None:
    clock = FakeClock(start=5.0)
    assert clock.now() == 5.0
    clock.sleep(2.5)
    assert clock.now() == 7.5
    assert clock.sleeps == [2.5]
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_deadline_math(fake_clock: FakeClock) -> None:
    deadline = Deadline.after(fake_clock, 10.0)
    assert deadline.remaining() == 10.0
    assert not deadline.expired()
    fake_clock.advance(4.0)
    assert deadline.minus(6.0).expired()
    assert deadline.minus(5.0).remaining() == pytest.approx(1.0)
    earlier = Deadline.after(fake_clock, 1.0)
    assert deadline.earliest(earlier) is earlier


def test_system_clock_is_monotonic() -> None:
    clock = SystemClock()
    a = clock.now()
    clock.sleep(0)
    assert clock.now() >= a
