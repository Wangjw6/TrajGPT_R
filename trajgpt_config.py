from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    action_dim: int
    spatial_vocab_size: int
    user_vocab_size: int
    departure_bins: int = 144
    speed_bins: Optional[int] = None


DATASET_SPECS = {
    "toyota": DatasetSpec(
        name="toyota",
        action_dim=9,
        spatial_vocab_size=262144,
        user_vocab_size=40000,
        speed_bins=120,
    ),
    "tdrive": DatasetSpec(
        name="tdrive",
        action_dim=10,
        spatial_vocab_size=16384,
        user_vocab_size=10000,
    ),
    "porto": DatasetSpec(
        name="porto",
        action_dim=10,
        spatial_vocab_size=5524,
        user_vocab_size=512,
    ),
    "geolife": DatasetSpec(
        name="geolife",
        action_dim=10,
        spatial_vocab_size=121649,
        user_vocab_size=10000,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: {valid}") from exc
