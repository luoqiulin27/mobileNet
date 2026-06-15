from dataclasses import dataclass


@dataclass(frozen=True)
class SanpoProfile:
    name: str
    output_dir_name: str
    classes: tuple[str, ...]
    mask_to_class: dict[int, int]


SANPO_PROFILES: dict[str, SanpoProfile] = {
    "phase1_5class": SanpoProfile(
        name="phase1_5class",
        output_dir_name="phase1_sanpo_5class",
        classes=("person", "vehicle", "pole", "stairs", "obstacle"),
        mask_to_class={
            12: 0,  # pedestrian -> person
            21: 1,  # vehicle -> vehicle
            24: 2,  # pole -> pole
            15: 3,  # stairs -> stairs
            20: 4,  # obstacle -> obstacle
            26: 4,  # bike rack -> obstacle
            14: 4,  # animal -> obstacle
            13: 4,  # rider -> obstacle
        },
    ),
    "sanpo_obstacle_8class": SanpoProfile(
        name="sanpo_obstacle_8class",
        output_dir_name="sanpo_obstacle_8class",
        classes=("person", "vehicle", "rider", "animal", "stairs", "pole", "bike_rack", "obstacle"),
        mask_to_class={
            12: 0,  # pedestrian
            21: 1,  # vehicle
            13: 2,  # rider
            14: 3,  # animal
            15: 4,  # stairs
            24: 5,  # pole
            26: 6,  # bike rack
            20: 7,  # obstacle
        },
    ),
}


def get_sanpo_profile(name: str) -> SanpoProfile:
    if name not in SANPO_PROFILES:
        valid = ", ".join(sorted(SANPO_PROFILES))
        raise KeyError(f"Unknown SANPO profile: {name}. Valid profiles: {valid}")
    return SANPO_PROFILES[name]
