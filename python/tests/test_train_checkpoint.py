from __future__ import annotations

from pathlib import Path

from gimbur_nn.train import _prune_epoch_checkpoints


def test_prune_epoch_checkpoints_keeps_latest_files(tmp_path: Path) -> None:
    for epoch in (1, 2, 3, 4):
        (tmp_path / f"epoch_{epoch}.pt").write_text(str(epoch))
    (tmp_path / "epoch_5.pt.tmp").write_text("partial")

    _prune_epoch_checkpoints(tmp_path, retention=2)

    assert sorted(path.name for path in tmp_path.glob("epoch_*.pt")) == [
        "epoch_3.pt",
        "epoch_4.pt",
    ]
    assert (tmp_path / "epoch_5.pt.tmp").is_file()
