from __future__ import annotations
import hashlib

TASKS = {
    "t1_y1_d0_to_y1_d7": {"source_day": 0, "target_day": 7},
    "t2_d7_to_d90": {"source_day": 7, "target_day": 90},
    "t3_y2_d0_to_y2_d90": {"source_day": 0, "target_day": 90},
}


def _stable_seed(*values: object, base_seed: int = 42) -> int:
    payload = "|".join([str(base_seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (
        2**32
    )


