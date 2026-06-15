from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SunrgbdProfile:
    name: str
    output_dir_name: str
    classes: tuple[str, ...]
    aliases: dict[str, str]
    ignored_names: tuple[str, ...]

    @property
    def class_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.classes)}


def canonicalize_name(name: str) -> str:
    name = name.strip().lower()
    for char in ("-", " ", "/"):
        name = name.replace(char, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


SUNRGBD_INDOOR_12_CLASSES = (
    "person",
    "seat",
    "table",
    "sofa",
    "bed",
    "storage",
    "door",
    "sanitary",
    "appliance",
    "plant",
    "box_bag",
    "indoor_obstacle",
)


SUNRGBD_INDOOR_12_ALIASES = {
    "person": "person",
    "chair": "seat",
    "chairs": "seat",
    "stool": "seat",
    "bench": "seat",
    "auditoriumseat": "seat",
    "seat": "seat",
    "table": "table",
    "desk": "table",
    "counter": "table",
    "coffee_table": "table",
    "dining_table": "table",
    "sofa": "sofa",
    "couch": "sofa",
    "bed": "bed",
    "mattress": "bed",
    "cabinet": "storage",
    "cupboard": "storage",
    "dresser": "storage",
    "drawer": "storage",
    "shelf": "storage",
    "shelves": "storage",
    "bookshelf": "storage",
    "bookcase": "storage",
    "nightstand": "storage",
    "wardrobe": "storage",
    "door": "door",
    "doorway": "door",
    "sink": "sanitary",
    "toilet": "sanitary",
    "faucet": "sanitary",
    "bathtub": "sanitary",
    "monitor": "appliance",
    "moniter": "appliance",
    "computermonitor": "appliance",
    "computer_monitor": "appliance",
    "computer": "appliance",
    "desktop": "appliance",
    "cpu": "appliance",
    "laptop": "appliance",
    "printer": "appliance",
    "keyboard": "appliance",
    "mouse": "appliance",
    "television": "appliance",
    "tv": "appliance",
    "lamp": "appliance",
    "plant": "plant",
    "box": "box_bag",
    "bag": "box_bag",
    "backpack": "box_bag",
    "container": "box_bag",
    "basket": "box_bag",
    "trashcan": "box_bag",
    "garbage_bin": "box_bag",
    "column": "indoor_obstacle",
    "pillar": "indoor_obstacle",
    "partition": "indoor_obstacle",
    "divider": "indoor_obstacle",
    "stand": "indoor_obstacle",
    "pipe": "indoor_obstacle",
    "board": "indoor_obstacle",
    "blackboard": "indoor_obstacle",
    "whiteboard": "indoor_obstacle",
    "mirror": "indoor_obstacle",
    "rug": "indoor_obstacle",
    "carpet": "indoor_obstacle",
}


SUNRGBD_IGNORED_NAMES = (
    "",
    "unknown",
    "idk",
    "wall",
    "floor",
    "ceiling",
    "window",
    "blinds",
    "curtain",
    "picture",
    "pictureframe",
    "poster",
    "paper",
    "papers",
    "book",
    "books",
    "bottle",
    "cup",
    "bowl",
    "plate",
    "pen",
    "outlet",
    "electrical_outlet",
    "electricaloutlet",
    "switch",
    "handle",
    "baseboard",
    "wire",
    "cord",
    "cable",
    "tag",
    "pricetag",
    "sign",
)


SUNRGBD_PROFILES: dict[str, SunrgbdProfile] = {
    "sunrgbd_indoor_12class": SunrgbdProfile(
        name="sunrgbd_indoor_12class",
        output_dir_name="sunrgbd_indoor_12class",
        classes=SUNRGBD_INDOOR_12_CLASSES,
        aliases={canonicalize_name(k): v for k, v in SUNRGBD_INDOOR_12_ALIASES.items()},
        ignored_names=tuple(canonicalize_name(name) for name in SUNRGBD_IGNORED_NAMES),
    ),
}


def get_sunrgbd_profile(name: str) -> SunrgbdProfile:
    if name not in SUNRGBD_PROFILES:
        valid = ", ".join(sorted(SUNRGBD_PROFILES))
        raise KeyError(f"Unknown SUNRGBD profile: {name}. Valid profiles: {valid}")
    return SUNRGBD_PROFILES[name]
