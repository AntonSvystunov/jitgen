import json

from iptc_parcs.config import Scenario
from iptc_parcs.faults import FaultInjector, parse_reply


def test_fault_compile_fires_once_on_create_session_only():
    faults = FaultInjector(Scenario.FAULT_COMPILE)

    assert faults.fault_for("get_cluster_info") is None
    first = faults.fault_for("create_session")
    assert first is not None
    assert "resubmit the same source" in json.loads(first)["error"]
    assert faults.fault_for("create_session") is None


def test_natural_scenario_never_injects():
    faults = FaultInjector(Scenario.NATURAL)
    assert all(faults.fault_for("create_session") is None for _ in range(3))


def test_parse_reply():
    assert parse_reply('{"a": 1}') == {"a": 1}
    assert parse_reply("[1, 2]") is None
    assert parse_reply("not json") is None
