"""CARLA 0.9.16's semantic tags and their colours (standard library only).

The tag numbers are carla.CityObjectLabel's in the 0.9.16 wheel (checked by tests against it when it is installed);
the colours are CARLA's CityScapes palette (LibCarla/source/carla/image/CityScapesPalette.h), so an image Falcon
colours matches what CARLA's own converter would draw.

Instance images (sensor.camera.instance_segmentation): R = the semantic tag, G + 256·B = the actor id for actors.
Map meshes (parked cars, signs, buildings) carry ids that are not actor ids; traffic-light heads and signs are better
claimed by their boxes than by their pixels.
"""
from __future__ import annotations

# tag: (name, (r, g, b), group)
TAGS = {
    0: ('unlabeled', (0, 0, 0), 'void'),
    1: ('road', (128, 64, 128), 'flat'),
    2: ('sidewalk', (244, 35, 232), 'flat'),
    3: ('building', (70, 70, 70), 'construction'),
    4: ('wall', (102, 102, 156), 'construction'),
    5: ('fence', (190, 153, 153), 'construction'),
    6: ('pole', (153, 153, 153), 'object'),
    7: ('traffic light', (250, 170, 30), 'object'),
    8: ('traffic sign', (220, 220, 0), 'object'),
    9: ('vegetation', (107, 142, 35), 'nature'),
    10: ('terrain', (152, 251, 152), 'nature'),
    11: ('sky', (70, 130, 180), 'sky'),
    12: ('pedestrian', (220, 20, 60), 'human'),
    13: ('rider', (255, 0, 0), 'human'),
    14: ('car', (0, 0, 142), 'vehicle'),
    15: ('truck', (0, 0, 70), 'vehicle'),
    16: ('bus', (0, 60, 100), 'vehicle'),
    17: ('train', (0, 80, 100), 'vehicle'),
    18: ('motorcycle', (0, 0, 230), 'vehicle'),
    19: ('bicycle', (119, 11, 32), 'vehicle'),
    20: ('static', (110, 190, 160), 'object'),
    21: ('dynamic', (170, 120, 50), 'object'),
    22: ('other', (55, 90, 80), 'void'),
    23: ('water', (45, 60, 150), 'nature'),
    24: ('road line', (157, 234, 50), 'flat'),
    25: ('ground', (81, 0, 81), 'flat'),
    26: ('bridge', (150, 100, 100), 'construction'),
    27: ('rail track', (230, 150, 140), 'flat'),
    28: ('guard rail', (180, 165, 180), 'construction'),
}
# carla.CityObjectLabel names for the same numbers (the wheel's spelling)
CARLA_NAMES = {0: 'NONE', 1: 'Roads', 2: 'Sidewalks', 3: 'Buildings', 4: 'Walls', 5: 'Fences', 6: 'Poles',
               7: 'TrafficLight', 8: 'TrafficSigns', 9: 'Vegetation', 10: 'Terrain', 11: 'Sky', 12: 'Pedestrians',
               13: 'Rider', 14: 'Car', 15: 'Truck', 16: 'Bus', 17: 'Train', 18: 'Motorcycle', 19: 'Bicycle',
               20: 'Static', 21: 'Dynamic', 22: 'Other', 23: 'Water', 24: 'RoadLines', 25: 'Ground', 26: 'Bridge',
               27: 'RailTrack', 28: 'GuardRail'}
VEHICLE_TAGS = (14, 15, 16, 17, 18, 19)
HUMAN_TAGS = (12, 13)


def name(tag: int) -> str:
    return TAGS.get(int(tag), (f'tag {tag}',))[0]


def color(tag: int) -> tuple:
    return TAGS.get(int(tag), ('', (255, 0, 255)))[1]


def palette_lut():
    """A 256 × 3 uint8 lookup (numpy) from tag to RGB; unknown tags magenta."""
    import numpy as np
    lut = np.zeros((256, 3), np.uint8)
    lut[:] = (255, 0, 255)
    for t, (_, rgb, _) in TAGS.items():
        lut[t] = rgb
    return lut


def colorize(tags):
    """An H × W tag image as H × W × 3 RGB (numpy)."""
    return palette_lut()[tags]
