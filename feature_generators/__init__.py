from .base_features_generator import BaseFeaturesGenerator
from .geo_utils import *  # optional, if you want utils exposed at package level

from .ground_features_generator import GroundFeaturesGenerator
from .roof_features_generator import RoofFeaturesGenerator
from .wall_features_generator import WallFeaturesGenerator
from .window_features_generator import WindowFeaturesGenerator
from .door_features_generator import DoorFeaturesGenerator
from .main_object_features_generator import MainObjectFeaturesGenerator

# Registry mapping semantic type names to generator classes
TYPE_GENERATORS = {
    "GroundSurface": GroundFeaturesGenerator,
    "RoofSurface": RoofFeaturesGenerator,
    "WallSurface": WallFeaturesGenerator,
    "Window": WindowFeaturesGenerator,
    "Door": DoorFeaturesGenerator,
    "MainObject": MainObjectFeaturesGenerator,
}
