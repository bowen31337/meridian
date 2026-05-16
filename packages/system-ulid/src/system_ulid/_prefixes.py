from __future__ import annotations

from enum import Enum


class IdPrefix(str, Enum):
    AGENT = "agent"
    SESS = "sess"
    THR = "thr"
    MSG = "msg"
    TC = "tc"
    SKILL = "skill"
    SKILLVER = "skillver"
    ENV = "env"
    MEM = "mem"
    VAULT = "vault"
    USR = "usr"
    CHAN = "chan"
    FILE = "file"
    WH = "wh"
