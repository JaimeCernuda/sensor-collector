"""Tests for Google Colab checkpoint persistence and training log utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from tick2.utils.colab import (
    load_checkpoint_from_drive,
    save_checkpoint_to_drive,
    setup_training_log,
)

# Patch targets for mocking Colab environment
_PATCH_IS_COLAB = "tick2.utils.colab.is_colab"
_PATCH_DRIVE_MOUNTED = "tick2.utils.colab._drive_is_mounted"


@pytest.fixture()
def _mock_colab_env():
    """Mock both is_colab and _drive_is_mounted to simulate Colab."""
    with (
        patch(_PATCH_IS_COLAB, return_value=True),
        patch(_PATCH_DRIVE_MOUNTED, return_value=True),
    ):
        yield


@pytest.fixture()
def tmp_checkpoint(tmp_path: Path) -> Path:
    """Create a fake checkpoint directory with files."""
    ckpt = tmp_path / "model_ft" / "best"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"model": "test"}')
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 64)
    return tmp_path / "model_ft"


@pytest.fixture()
def tmp_single_file(tmp_path: Path) -> Path:
    """Create a single checkpoint file (e.g. best_model.pt)."""
    pt = tmp_path / "best_model.pt"
    pt.write_bytes(b"\x00" * 128)
    return pt


class TestSaveCheckpointToDrive:
    """Tests for save_checkpoint_to_drive."""

    def test_returns_none_outside_colab(self, tmp_checkpoint: Path) -> None:
        with patch(_PATCH_IS_COLAB, return_value=False):
            result = save_checkpoint_to_drive(
                local_path=tmp_checkpoint,
                model_name="test_model",
            )
        assert result is None

    def test_returns_none_drive_not_mounted(self, tmp_checkpoint: Path) -> None:
        with (
            patch(_PATCH_IS_COLAB, return_value=True),
            patch(_PATCH_DRIVE_MOUNTED, return_value=False),
        ):
            result = save_checkpoint_to_drive(
                local_path=tmp_checkpoint,
                drive_base="/nonexistent/drive",
                model_name="test_model",
            )
        assert result is None

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_copies_directory_tree(self, tmp_checkpoint: Path, tmp_path: Path) -> None:
        drive_base = str(tmp_path / "drive_checkpoints")
        result = save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            notebook="03",
            model_name="granite_ft",
        )

        assert result is not None
        assert result == Path(drive_base) / "03" / "granite_ft"
        assert (result / "best" / "config.json").exists()
        assert (result / "best" / "model.safetensors").exists()

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_copies_single_file(self, tmp_single_file: Path, tmp_path: Path) -> None:
        drive_base = str(tmp_path / "drive_checkpoints")
        result = save_checkpoint_to_drive(
            local_path=tmp_single_file,
            drive_base=drive_base,
            notebook="03",
            model_name="timesfm_ft/best_model.pt",
        )

        assert result is not None
        assert result.exists()

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_overwrites_existing_drive_checkpoint(
        self, tmp_checkpoint: Path, tmp_path: Path
    ) -> None:
        drive_base = str(tmp_path / "drive_checkpoints")

        # First save
        save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            model_name="granite_ft",
        )

        # Modify local checkpoint
        (tmp_checkpoint / "best" / "config.json").write_text('{"model": "v2"}')

        # Second save should overwrite
        result = save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            model_name="granite_ft",
        )

        assert result is not None
        content = (result / "best" / "config.json").read_text()
        assert '"v2"' in content

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_returns_none_for_nonexistent_local_path(self, tmp_path: Path) -> None:
        result = save_checkpoint_to_drive(
            local_path=tmp_path / "does_not_exist",
            drive_base=str(tmp_path / "drive"),
            model_name="test",
        )
        assert result is None


class TestLoadCheckpointFromDrive:
    """Tests for load_checkpoint_from_drive."""

    def test_returns_none_outside_colab(self) -> None:
        with patch(_PATCH_IS_COLAB, return_value=False):
            result = load_checkpoint_from_drive(model_name="test")
        assert result is None

    def test_returns_none_drive_not_mounted(self) -> None:
        with (
            patch(_PATCH_IS_COLAB, return_value=True),
            patch(_PATCH_DRIVE_MOUNTED, return_value=False),
        ):
            result = load_checkpoint_from_drive(model_name="test")
        assert result is None

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_returns_none_when_checkpoint_missing(self, tmp_path: Path) -> None:
        result = load_checkpoint_from_drive(
            drive_base=str(tmp_path / "empty"),
            model_name="nonexistent",
        )
        assert result is None

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_returns_drive_path_without_local(
        self, tmp_checkpoint: Path, tmp_path: Path
    ) -> None:
        drive_base = str(tmp_path / "drive_ckpts")
        save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            notebook="03",
            model_name="granite_ft",
        )

        result = load_checkpoint_from_drive(
            drive_base=drive_base,
            notebook="03",
            model_name="granite_ft",
        )

        assert result is not None
        assert result == Path(drive_base) / "03" / "granite_ft"

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_copies_to_local_path(self, tmp_checkpoint: Path, tmp_path: Path) -> None:
        drive_base = str(tmp_path / "drive_ckpts")
        local_dest = tmp_path / "restored_model"

        save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            notebook="03",
            model_name="moirai_ft",
        )

        result = load_checkpoint_from_drive(
            drive_base=drive_base,
            notebook="03",
            model_name="moirai_ft",
            local_path=str(local_dest),
        )

        assert result is not None
        assert result == local_dest
        assert (local_dest / "best" / "config.json").exists()

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_skips_copy_if_local_exists(
        self, tmp_checkpoint: Path, tmp_path: Path
    ) -> None:
        drive_base = str(tmp_path / "drive_ckpts")
        local_dest = tmp_path / "existing_model"
        local_dest.mkdir(parents=True)
        (local_dest / "marker.txt").write_text("pre-existing")

        save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            model_name="test",
        )

        result = load_checkpoint_from_drive(
            drive_base=drive_base,
            model_name="test",
            local_path=str(local_dest),
        )

        assert result == local_dest
        # The pre-existing marker should still be there (not overwritten)
        assert (local_dest / "marker.txt").exists()


class TestRoundTrip:
    """End-to-end save and load cycle."""

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_directory_checkpoint_round_trip(
        self, tmp_checkpoint: Path, tmp_path: Path
    ) -> None:
        drive_base = str(tmp_path / "drive")
        restore_dest = tmp_path / "restored"

        saved = save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            notebook="03",
            model_name="granite_ft/combined/E1",
        )
        assert saved is not None

        loaded = load_checkpoint_from_drive(
            drive_base=drive_base,
            notebook="03",
            model_name="granite_ft/combined/E1",
            local_path=str(restore_dest),
        )

        assert loaded is not None
        assert (loaded / "best" / "config.json").exists()
        original = (tmp_checkpoint / "best" / "config.json").read_text()
        restored = (loaded / "best" / "config.json").read_text()
        assert original == restored

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_file_checkpoint_round_trip(
        self, tmp_single_file: Path, tmp_path: Path
    ) -> None:
        drive_base = str(tmp_path / "drive")
        restore_dest = tmp_path / "restored_model.pt"

        saved = save_checkpoint_to_drive(
            local_path=tmp_single_file,
            drive_base=drive_base,
            model_name="timesfm_ft/best_model.pt",
        )
        assert saved is not None

        loaded = load_checkpoint_from_drive(
            drive_base=drive_base,
            model_name="timesfm_ft/best_model.pt",
            local_path=str(restore_dest),
        )

        assert loaded is not None
        assert loaded.exists()
        assert loaded.stat().st_size == tmp_single_file.stat().st_size


class TestPathConstruction:
    """Verify correct Drive path layout."""

    def test_default_drive_base_without_drive(self, tmp_checkpoint: Path) -> None:
        """When is_colab=True but Drive isn't mounted, returns None."""
        with (
            patch(_PATCH_IS_COLAB, return_value=True),
            patch(_PATCH_DRIVE_MOUNTED, return_value=False),
        ):
            result = save_checkpoint_to_drive(
                local_path=tmp_checkpoint,
                model_name="granite_ft",
            )
        assert result is None

    @pytest.mark.usefixtures("_mock_colab_env")
    def test_nested_model_name(self, tmp_checkpoint: Path, tmp_path: Path) -> None:
        drive_base = str(tmp_path / "drive")
        result = save_checkpoint_to_drive(
            local_path=tmp_checkpoint,
            drive_base=drive_base,
            notebook="03",
            model_name="granite_ttm_ft/combined/E2_mix10",
        )

        assert result is not None
        expected = Path(drive_base) / "03" / "granite_ttm_ft" / "combined" / "E2_mix10"
        assert result == expected


class TestSetupTrainingLog:
    """Tests for setup_training_log."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_path = setup_training_log(tmp_path, name="training")
        assert log_path == tmp_path / "training.log"
        assert log_path.exists()

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        log_path = setup_training_log(nested)
        assert nested.exists()
        assert log_path.parent == nested

    def test_attaches_handler_to_finetuning_logger(self, tmp_path: Path) -> None:
        setup_training_log(tmp_path, name="test_attach")
        ft_logger = logging.getLogger("tick2.finetuning")
        file_handlers = [
            h for h in ft_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert any(
            Path(h.baseFilename).name == "test_attach.log" for h in file_handlers
        )

    def test_captures_finetuning_messages(self, tmp_path: Path) -> None:
        log_path = setup_training_log(tmp_path, name="capture")
        child = logging.getLogger("tick2.finetuning.moirai_ft")
        child.info("Epoch 1/10: train_loss=0.500000, val_loss=0.600000")
        # Flush handlers
        for h in logging.getLogger("tick2.finetuning").handlers:
            h.flush()
        content = log_path.read_text(encoding="utf-8")
        assert "Epoch 1/10" in content
        assert "val_loss=0.600000" in content

    def test_no_duplicate_handlers_on_rerun(self, tmp_path: Path) -> None:
        setup_training_log(tmp_path, name="dedup")
        setup_training_log(tmp_path, name="dedup")
        ft_logger = logging.getLogger("tick2.finetuning")
        matching = [
            h
            for h in ft_logger.handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).name == "dedup.log"
        ]
        assert len(matching) == 1

    def test_custom_name(self, tmp_path: Path) -> None:
        log_path = setup_training_log(tmp_path, name="granite_E1")
        assert log_path.name == "granite_E1.log"

    def teardown_method(self) -> None:
        """Remove any file handlers we added during tests."""
        ft_logger = logging.getLogger("tick2.finetuning")
        for h in ft_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                ft_logger.removeHandler(h)
