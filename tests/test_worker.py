from datetime import datetime

from tl.worker import in_active_hours


def at(h):
    return datetime(2026, 10, 4, h, 30)


def test_active_hours_window(settings):
    s = settings.__class__(**{**settings.__dict__, "scrape_active_hours": (8, 22)})
    assert in_active_hours(s, at(8)) and in_active_hours(s, at(21))
    assert not in_active_hours(s, at(22)) and not in_active_hours(s, at(3))


def test_active_hours_wrapping_midnight(settings):
    s = settings.__class__(**{**settings.__dict__, "scrape_active_hours": (20, 2)})
    assert in_active_hours(s, at(23)) and in_active_hours(s, at(1))
    assert not in_active_hours(s, at(12))
