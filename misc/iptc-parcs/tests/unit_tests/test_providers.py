import pytest

from iptc_parcs.providers import LMSTUDIO, OPENROUTER


def test_openrouter_sets_reasoning_effort_and_pins_providers_without_fallback():
    assert OPENROUTER.routing_options("low", ("deepinfra", "novita")) == {
        "extra_body": {
            "reasoning": {"effort": "low"},
            "provider": {"order": ["deepinfra", "novita"], "allow_fallbacks": False},
        }
    }
    assert OPENROUTER.routing_options("", ()) == {}


def test_lmstudio_rejects_openrouter_only_options():
    assert LMSTUDIO.routing_options("", ()) == {}
    with pytest.raises(ValueError, match="--reasoning-effort or --provider"):
        LMSTUDIO.routing_options("low", ())
    with pytest.raises(ValueError, match="--reasoning-effort or --provider"):
        LMSTUDIO.routing_options("", ("deepinfra",))
