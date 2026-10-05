"""未互证日志缺口诊断（复核员按演练查看）测试。

覆盖四类归因：
  candidate_occupied      —— 候选已占用（完全符合条件的对方已入选别的配对）；
  time_diff_exceeded      —— 时差超限（互指且波段模式相同但时差超允许值）；
  band_or_mode_mismatch   —— 波段或模式不符（有互指但波段/模式不同）；
  counterpart_log_missing —— 缺少对方日志；
以及“时差最小再按记录号最小”的择对方规则、按记录号排序、只读、
锁定占用、封存可查、未知演练报错。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("QSO_DB_PATH", tmp.name)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def _log(no, station, counterpart, t, band="40m", mode="SSB"):
    return {
        "record_no": no,
        "station": station,
        "counterpart": counterpart,
        "time": t,
        "band": band,
        "mode": mode,
    }


def _items_by_record(body):
    return {item["record_no"]: item for item in body["items"]}


T0 = "2026-10-04T00:00:00+00:00"


# --------------------------------------------------------------------------
# 纯算法层单测
# --------------------------------------------------------------------------
def _rec(no, station, counterpart, secs, band="40M", mode="SSB"):
    from app.matching import Record

    return Record(
        record_no=no,
        station=station,
        counterpart=counterpart,
        ts_micros=secs * 1_000_000,
        band=band,
        mode=mode,
    )


def _diagnose(records, unpaired, allowed=10):
    from app.matching import diagnose_unpaired_gaps

    return {
        g.record_no: g
        for g in diagnose_unpaired_gaps(records, allowed, unpaired)
    }


def test_algorithm_four_categories():
    records = [
        # X：完全合格的对方 C1 已与别的记录配对 -> 候选已占用
        _rec("X", "A", "B", 0),
        _rec("D", "A", "B", 2),
        _rec("C1", "B", "A", 2),
        # T：互指且波段模式相同，但时差 60s 超限 -> 时差超限
        _rec("T", "D", "E", 0),
        _rec("T1", "E", "D", 60),
        # B：有互指但波段/模式不符 -> 波段或模式不符
        _rec("B", "F", "G", 0),
        _rec("B1", "G", "F", 0, band="20M", mode="SSB"),
        _rec("B2", "G", "F", 0, band="40M", mode="CW"),
        # M：无人互指 -> 缺少对方日志
        _rec("M", "H", "I", 0),
    ]
    gaps = _diagnose(records, ["X", "T", "B", "M"])

    assert gaps["X"].reason == "candidate_occupied"
    assert gaps["X"].counterpart_record_no == "C1"
    assert gaps["X"].time_diff_seconds == 2

    assert gaps["T"].reason == "time_diff_exceeded"
    assert gaps["T"].counterpart_record_no == "T1"
    assert gaps["T"].time_diff_seconds == 60

    assert gaps["B"].reason == "band_or_mode_mismatch"
    # B1/B2 时差同为 0，取记录号最小的 B1
    assert gaps["B"].counterpart_record_no == "B1"
    assert gaps["B"].time_diff_seconds == 0

    assert gaps["M"].reason == "counterpart_log_missing"
    assert gaps["M"].counterpart_record_no is None
    assert gaps["M"].time_diff_seconds is None


def test_algorithm_closest_is_min_diff_then_min_record_no():
    # 两个完全合格候选：取时差最小；时差相同取记录号最小
    records = [
        _rec("U", "A", "B", 0),
        _rec("Q1", "B", "A", 5),
        _rec("Q2", "B", "A", 1),
    ]
    gaps = _diagnose(records, ["U"])
    assert gaps["U"].reason == "candidate_occupied"
    assert gaps["U"].counterpart_record_no == "Q2"
    assert gaps["U"].time_diff_seconds == 1

    records = [
        _rec("U", "A", "B", 0),
        _rec("Q1", "B", "A", 3),
        _rec("Q2", "B", "A", 3),
    ]
    gaps = _diagnose(records, ["U"])
    assert gaps["U"].counterpart_record_no == "Q1"
    assert gaps["U"].time_diff_seconds == 3


def test_algorithm_category_order_is_strict():
    # 同时存在合格候选、超限候选、波段模式不符候选时，优先归“候选已占用”
    records = [
        _rec("U", "A", "B", 0),
        _rec("OK", "B", "A", 1),                      # 合格
        _rec("FAR", "B", "A", 50),                    # 互指同波段模式但超限
        _rec("OTHER", "B", "A", 0, mode="CW"),        # 互指但模式不符
    ]
    gaps = _diagnose(records, ["U"])
    assert gaps["U"].reason == "candidate_occupied"
    assert gaps["U"].counterpart_record_no == "OK"


def test_algorithm_result_sorted_by_record_no():
    from app.matching import diagnose_unpaired_gaps

    records = [_rec("R3", "A", "X", 0), _rec("R1", "B", "Y", 0),
               _rec("R2", "C", "Z", 0)]
    gaps = diagnose_unpaired_gaps(records, 10, ["R3", "R1", "R2"])
    assert [g.record_no for g in gaps] == ["R1", "R2", "R3"]


# --------------------------------------------------------------------------
# HTTP 集成测试
# --------------------------------------------------------------------------
def test_diagnosis_four_categories_over_http(client):
    client.post(
        "/drills",
        json={"drill_id": "GAP", "allowed_time_diff_seconds": 10},
    )
    payload = {"logs": [
        # X、D 均为 A->B，C1 为 B->A；最优配对 (C1, D) 时差 0，
        # X 落选 -> 候选已占用（对方 C1，时差 2）
        _log("X", "A", "B", "2026-10-04T00:00:00+00:00"),
        _log("D", "A", "B", "2026-10-04T00:00:02+00:00"),
        _log("C1", "B", "A", "2026-10-04T00:00:02+00:00"),
        # T/T1 互指同波段模式，时差 60s 超限 -> 时差超限
        _log("T", "D", "E", "2026-10-04T00:00:00+00:00"),
        _log("T1", "E", "D", "2026-10-04T00:01:00+00:00"),
        # BM 与 BM1 互指但模式不同 -> 波段或模式不符
        _log("BM", "F", "G", "2026-10-04T00:00:00+00:00", mode="SSB"),
        _log("BM1", "G", "F", "2026-10-04T00:00:00+00:00", mode="CW"),
        # M 无人互指 -> 缺少对方日志
        _log("M", "H", "I", "2026-10-04T00:00:00+00:00"),
    ]}
    r = client.post("/drills/GAP/logs", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["unpaired"] == ["X", "T", "T1", "BM", "BM1", "M"]

    r = client.get("/drills/GAP/unpaired-diagnosis")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["drill_id"] == "GAP"
    assert body["version"] == 1
    assert body["sealed"] is False
    # 按记录号升序
    assert [it["record_no"] for it in body["items"]] == [
        "BM", "BM1", "M", "T", "T1", "X"
    ]

    by = _items_by_record(body)
    assert by["X"]["reason"] == "candidate_occupied"
    assert by["X"]["reason_label"] == "候选已占用"
    assert by["X"]["counterpart_record_no"] == "C1"
    assert by["X"]["time_diff_seconds"] == 2

    assert by["T"]["reason"] == "time_diff_exceeded"
    assert by["T"]["reason_label"] == "时差超限"
    assert by["T"]["counterpart_record_no"] == "T1"
    assert by["T"]["time_diff_seconds"] == 60
    assert by["T1"]["reason"] == "time_diff_exceeded"
    assert by["T1"]["counterpart_record_no"] == "T"
    assert by["T1"]["time_diff_seconds"] == 60

    assert by["BM"]["reason"] == "band_or_mode_mismatch"
    assert by["BM"]["reason_label"] == "波段或模式不符"
    assert by["BM"]["counterpart_record_no"] == "BM1"
    assert by["BM"]["time_diff_seconds"] == 0
    assert by["BM1"]["reason"] == "band_or_mode_mismatch"
    assert by["BM1"]["counterpart_record_no"] == "BM"

    assert by["M"]["reason"] == "counterpart_log_missing"
    assert by["M"]["reason_label"] == "缺少对方日志"
    assert by["M"]["counterpart_record_no"] is None
    assert by["M"]["time_diff_seconds"] is None


def test_diagnosis_closest_counterpart_tie_break(client):
    # 同站两条 A->B 竞争唯一对方 B->A：落选者按“时差最小”指认入选对方
    client.post(
        "/drills",
        json={"drill_id": "TB", "allowed_time_diff_seconds": 10},
    )
    client.post("/drills/TB/logs", json={"logs": [
        _log("X", "A", "B", "2026-10-04T00:00:00+00:00"),
        _log("Y", "A", "B", "2026-10-04T00:00:01+00:00"),
        _log("K", "B", "A", "2026-10-04T00:00:01+00:00"),
    ]})
    result = client.get("/drills/TB/result").json()
    assert [(p["record_a"], p["record_b"]) for p in result["pairs"]] == [
        ("K", "Y")
    ]
    body = client.get("/drills/TB/unpaired-diagnosis").json()
    by = _items_by_record(body)
    assert by["X"]["reason"] == "candidate_occupied"
    assert by["X"]["counterpart_record_no"] == "K"
    assert by["X"]["time_diff_seconds"] == 1

    # 时差并列：两个合格候选时差相同，取记录号最小（两条同站竞争记录
    # 分别占用两个对方，使 X 落选）
    client.post(
        "/drills",
        json={"drill_id": "TB2", "allowed_time_diff_seconds": 10},
    )
    client.post("/drills/TB2/logs", json={"logs": [
        _log("X", "A", "B", "2026-10-04T00:00:00+00:00"),
        _log("Y1", "A", "B", "2026-10-04T00:00:03+00:00"),
        _log("Y2", "A", "B", "2026-10-04T00:00:03+00:00"),
        _log("K1", "B", "A", "2026-10-04T00:00:03+00:00"),
        _log("K2", "B", "A", "2026-10-04T00:00:03+00:00"),
    ]})
    body = client.get("/drills/TB2/unpaired-diagnosis").json()
    by = _items_by_record(body)
    # 最优：(K1,Y1)、(K2,Y2)，X 落选；X 的合格候选 K1/K2 时差均为 3，取 K1
    assert list(by) == ["X"]
    assert by["X"]["reason"] == "candidate_occupied"
    assert by["X"]["counterpart_record_no"] == "K1"
    assert by["X"]["time_diff_seconds"] == 3


def test_diagnosis_empty_drill(client):
    client.post(
        "/drills",
        json={"drill_id": "EMPTY", "allowed_time_diff_seconds": 10},
    )
    body = client.get("/drills/EMPTY/unpaired-diagnosis").json()
    assert body == {
        "drill_id": "EMPTY",
        "version": 0,
        "sealed": False,
        "items": [],
    }


def test_diagnosis_unknown_drill_404(client):
    r = client.get("/drills/NOPE/unpaired-diagnosis")
    assert r.status_code == 404
    assert r.json()["error"] == "drill_not_found"


def test_diagnosis_is_read_only(client):
    client.post(
        "/drills",
        json={"drill_id": "RO", "allowed_time_diff_seconds": 10},
    )
    client.post("/drills/RO/logs", json={"logs": [
        _log("M", "H", "I", T0),
    ]})
    before_result = client.get("/drills/RO/result").json()
    before_snaps = client.get("/drills/RO/snapshots").json()
    for _ in range(3):
        r = client.get("/drills/RO/unpaired-diagnosis")
        assert r.status_code == 200
        assert r.json()["version"] == 1
    after_result = client.get("/drills/RO/result").json()
    after_snaps = client.get("/drills/RO/snapshots").json()
    assert after_result == before_result
    assert after_snaps == before_snaps


def test_diagnosis_reflects_writes_corrections_locks_and_sealed(client):
    client.post(
        "/drills",
        json={"drill_id": "LIVE", "allowed_time_diff_seconds": 10},
    )
    client.post("/drills/LIVE/logs", json={"logs": [
        _log("M", "H", "I", T0),
    ]})
    by = _items_by_record(client.get("/drills/LIVE/unpaired-diagnosis").json())
    assert by["M"]["reason"] == "counterpart_log_missing"

    # 补写对方但时差超限 -> 归因变为“时差超限”，版本推进
    client.put(
        "/drills/LIVE/logs/N",
        json=_log("N", "I", "H", "2026-10-04T00:01:00+00:00"),
    )
    body = client.get("/drills/LIVE/unpaired-diagnosis").json()
    assert body["version"] == 2
    by = _items_by_record(body)
    assert by["M"]["reason"] == "time_diff_exceeded"
    assert by["M"]["counterpart_record_no"] == "N"

    # 更正到时差内 -> M/N 配对，缺口清单为空，版本再推进
    client.put(
        "/drills/LIVE/logs/N",
        json=_log("N", "I", "H", "2026-10-04T00:00:04+00:00"),
    )
    body = client.get("/drills/LIVE/unpaired-diagnosis").json()
    assert body["version"] == 3 and body["items"] == []

    # 锁定占用造成的未配对也按“候选已占用”归因（另起演练便于控制时差）：
    # L1(t0)/L2(t100)、R1(t0)/R2(t50)，允许时差 50。锁定 (L1, R2)
    # 强制入选（时差 50）；L2 与 R1 时差 100 超限互不为合格候选，但
    # L2 对 R2（时差 50，已被锁定占用）、R1 对 L1（时差 0，已被锁定占用）
    # 都有完全合格的对方却未入选 -> 均为“候选已占用”。
    client.post(
        "/drills",
        json={"drill_id": "LK", "allowed_time_diff_seconds": 50},
    )
    client.post("/drills/LK/logs", json={"logs": [
        _log("L1", "A", "B", "2026-10-04T00:00:00+00:00"),
        _log("L2", "A", "B", "2026-10-04T00:01:40+00:00"),
        _log("R1", "B", "A", "2026-10-04T00:00:00+00:00"),
        _log("R2", "B", "A", "2026-10-04T00:00:50+00:00"),
    ]})
    client.put("/drills/LK/locks", json={"record_a": "L1", "record_b": "R2"})
    body = client.get("/drills/LK/unpaired-diagnosis").json()
    by = _items_by_record(body)
    assert sorted(by) == ["L2", "R1"]
    assert by["R1"]["reason"] == "candidate_occupied"
    assert by["R1"]["counterpart_record_no"] == "L1"
    assert by["R1"]["time_diff_seconds"] == 0
    assert by["L2"]["reason"] == "candidate_occupied"
    assert by["L2"]["counterpart_record_no"] == "R2"
    assert by["L2"]["time_diff_seconds"] == 50

    # 封存：版本不变、仍可查询且内容为最终态
    sealed_version = body["version"]
    client.post("/drills/LK/seal")
    r = client.get("/drills/LK/unpaired-diagnosis")
    assert r.status_code == 200
    sealed_body = r.json()
    assert sealed_body["sealed"] is True
    assert sealed_body["version"] == sealed_version
    assert [it["record_no"] for it in sealed_body["items"]] == ["L2", "R1"]
