from pathlib import Path
from typing import Optional, Union

import torch


PathLike = Union[str, Path]


def resolve_checkpoint_path(
        checkpoint: Optional[PathLike],
        default_dir: PathLike,
        extension: str,
) -> Optional[Path]:
    """Resolve a checkpoint argument.

    The argument may be:
    - None/empty: no checkpoint.
    - a full or relative path: used as provided.
    - a bare checkpoint stem: resolved under default_dir with extension appended.
    """
    if checkpoint is None:
        return None
    checkpoint = str(checkpoint).strip()
    if not checkpoint:
        return None

    path = Path(checkpoint)
    if path.suffix:
        return path
    if len(path.parts) > 1:
        return path.with_suffix(extension)
    return Path(default_dir) / f"{checkpoint}{extension}"


def load_state_dict(
        checkpoint: PathLike,
        device: Union[str, torch.device],
) -> dict:
    """Load a torch checkpoint onto the selected runtime device."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    map_location = torch.device(device)
    if map_location.type == "cuda" and not torch.cuda.is_available():
        map_location = torch.device("cpu")
    return torch.load(path, map_location=map_location)


def strip_compile_prefix(state_dict: dict) -> dict:
    """Normalize checkpoints saved from torch.compile-wrapped modules."""
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def load_model_state(
        model: torch.nn.Module,
        checkpoint: PathLike,
        device: Union[str, torch.device],
        strict: bool = True,
        allow_partial: bool = False,
) -> None:
    """Load a checkpoint into a model with explicit partial-load behavior."""
    state_dict = strip_compile_prefix(load_state_dict(checkpoint, device))
    if allow_partial:
        model_state_dict = model.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if key in model_state_dict}
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected}")
