from _fakes.tools import FilesWidget, HumanOnlyWidget

from swarm_arena import Ref
from swarm_arena._model.goal import FileSet


def test_files_widget_owns_snapshot_config() -> None:
    assert set(FilesWidget.Config.model_fields) == {"files", "members", "label"}
    config = FilesWidget.Config.model_validate(
        {"files": ["data/*.csv"], "members": [{"path": "_fakes.tools:Widget"}], "label": "x"}
    )
    assert config.files == FileSet(globs=["data/*.csv"])
    assert config.members == [Ref(path="_fakes.tools:Widget")]
    assert config.label == "x"


def test_human_only_widget_owns_command_config() -> None:
    assert set(HumanOnlyWidget.Config.model_fields) == {"command", "timeout_s"}
    assert HumanOnlyWidget.Config.model_validate(
        {"command": ["true"], "timeout_s": 1}
    ).model_dump() == {
        "command": ["true"],
        "timeout_s": 1.0,
    }
    assert HumanOnlyWidget._human_only == frozenset({"command"})
