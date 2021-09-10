__version__ = "0.1"
__description__ = "a example python package"
__doc__ = """
Document:
    pass

"""

from .video_utils import VideoLoader, VideoDumper
from .draw_text import draw_text
from .resize import resize


__all__ = [
    "VideoLoader",
    "VideoDumper",
    "draw_text",
    "resize",
]
