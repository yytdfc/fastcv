import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .environment import ENV


def draw_text(
    im, text, font_scale=0.1, align="top_center", margin_scale=0.01, color=(255, 0, 0), tx=-1, ty=-1, text_width=80,
):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) > 0:    
        text_split = text.split()
        text_lines = []
        text_line = text_split[0]
        for t in text_split[1:]:
            if len(text_line) + len(t) > text_width:
                text_lines.append(text_line)
                text_line = t
            else:
                text_line += " " + t
        text_lines.append(text_line)
        text = "\n".join(text_lines)
        impil = Image.fromarray(im)
        draw = ImageDraw.Draw(impil)
        fontsize = round(min(impil.size) * font_scale)
        if ENV.is_mac():
            fontStyle = ImageFont.truetype("PingFang", fontsize, encoding="utf-8")
        else:
            fontStyle = ImageFont.truetype(
                f"{os.path.dirname(__file__)}/PingFang.ttc", fontsize, encoding="utf-8"
            )
        w_word, h_word = fontStyle.getsize(text)
        if tx>= 0:
            pass
        elif align.endswith("left"):
            tx = round(im.shape[1] * margin_scale)
        elif align.endswith("right"):
            tx = round(im.shape[1] * (1 - margin_scale) - w_word)
        else:
            # defalut center
            tx = round((im.shape[1] - w_word) / 2)
        if ty>= 0:
            pass
        elif align.startswith("bottom"):
            ty = round(im.shape[0] * (1 - margin_scale) - h_word)
        else:
            # defalut top
            ty = round(im.shape[0] * margin_scale)

        draw.text((tx, ty), text, font=fontStyle, fill=tuple(color))
        return np.array(impil)
    else:
        return im
