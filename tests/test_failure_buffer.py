from __future__ import annotations

from fasb.buffers.failure_buffer import FailureBuffer


def test_failure_buffer_save_load(tmp_path) -> None:
    buffer = FailureBuffer(max_size=2)
    buffer.add({"seed": 1, "risk_score": 0.2, "failure_mode": "collision"})
    buffer.add({"seed": 2, "risk_score": 0.9, "failure_mode": "offroad"})
    buffer.add({"seed": 3, "risk_score": 0.1, "failure_mode": "solved"})
    assert len(buffer) == 2
    path = tmp_path / "buffer.jsonl"
    buffer.save(path)
    loaded = FailureBuffer.load(path)
    assert len(loaded) == 2
    assert loaded.sample(1)[0]["seed"] in {2, 3}


def test_priority_sample_stress() -> None:
    buffer = FailureBuffer(max_size=2000)
    for seed in range(1000):
        buffer.add({"seed": seed, "risk_score": seed / 999, "learnability": 1.0})
    for _ in range(100):
        assert 0 <= buffer.sample_priority()["seed"] < 1000
