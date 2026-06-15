"""Tests for the colour -> French mission mapping (pure logic, no camera/audio)."""

import json
from pathlib import Path

from vision.missions import (
    DEFAULT_ACTIONS_FR,
    build_mission,
    load_action_map,
)


def test_build_mission_preserves_order_and_repeats():
    mission = build_mission(
        ["blue", "green", "orange", "blue"], DEFAULT_ACTIONS_FR
    )
    assert mission.colors == ["blue", "green", "orange", "blue"]
    assert [s.index for s in mission.steps] == [1, 2, 3, 4]
    # Repeated colour appears twice with the same action.
    assert mission.steps[0].action == mission.steps[3].action
    assert mission.actions[1] == DEFAULT_ACTIONS_FR["green"]


def test_build_mission_drops_separators_and_unknown():
    mission = build_mission(
        ["black", "green", "unknown", "blue", "black"], DEFAULT_ACTIONS_FR
    )
    assert mission.colors == ["green", "blue"]


def test_unmapped_colour_becomes_unknown_step():
    mission = build_mission(["green", "teal"], DEFAULT_ACTIONS_FR)
    assert mission.steps[1].action is None
    assert not mission.steps[1].known
    assert "inconnue" in mission.steps[1].phrase().lower()
    assert mission.has_unknown


def test_empty_sequence_yields_empty_mission():
    mission = build_mission([], DEFAULT_ACTIONS_FR)
    assert mission.is_empty
    assert mission.narration() == "Aucune mission détectée."


def test_narration_numbers_each_step():
    mission = build_mission(["green", "blue"], DEFAULT_ACTIONS_FR)
    text = mission.narration()
    assert "Étape 1" in text and "Étape 2" in text
    assert DEFAULT_ACTIONS_FR["green"] in text


def test_mission_to_dict_roundtrips_colour_code():
    mission = build_mission(["green", "orange"], DEFAULT_ACTIONS_FR)
    data = mission.to_dict()
    assert data["color_code"] == "green-orange"
    assert len(data["steps"]) == 2


def test_load_action_map_override_file_wins(tmp_path: Path):
    override = tmp_path / "actions.json"
    override.write_text(json.dumps({"actions": {"green": "Saute"}}))
    actions = load_action_map(None, override)
    assert actions["green"] == "Saute"
    # Non-overridden defaults remain.
    assert actions["blue"] == DEFAULT_ACTIONS_FR["blue"]


def test_load_action_map_reads_calibration_actions_block(tmp_path: Path):
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"colors": {}, "actions": {"blue": "Recule"}}))
    actions = load_action_map(cal, None)
    assert actions["blue"] == "Recule"


def test_default_actions_cover_controller_colours():
    # The four colours the controller drives must all have French phrasing.
    for color in ("green", "blue", "yellow", "orange"):
        assert color in DEFAULT_ACTIONS_FR
