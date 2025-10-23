import json
from datetime import timedelta

from .transcript import format_timedelta


def json_default(obj):
    if isinstance(obj, timedelta):
        return format_timedelta(obj)

    return json.JSONEncoder().default(obj)
