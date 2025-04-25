from __future__ import annotations

import warnings

from rasterio.errors import NotGeoreferencedWarning

from .color_models import *
from .orthomosaic_tiler import *
from .tiled_color_based_distance import *
from .transforms import *

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Current version
__version__ = "1.1.2"
