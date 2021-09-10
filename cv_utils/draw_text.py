from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .environment import ENV


def draw_text(
    im, text, font_scale=0.1, align="top_center", margin_scale=0.01, color=[255, 0, 0]
):
    if not isinstance(text, str):
        text = str(text)
    impil = Image.fromarray(im)
    draw = ImageDraw.Draw(impil)
    fontsize = min(impil.size) // 15
    if ENV.is_mac():
        fontStyle = ImageFont.truetype("PingFang", fontsize, encoding="utf-8")
    else:
        fontStyle = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSans.ttf", fontsize, encoding="utf-8"
        )
    w_word, h_word = fontStyle.getsize(text)
    if align.endswith("left"):
        tx = round(im.shape[1] * margin_scale)
    elif align.endswith("right"):
        tx = round(im.shape[1] * (1 - margin_scale) - w_word)
    else:
        # defalut center
        tx = round((im.shape[1] - w_word) / 2)
    if align.startswith("bottom"):
        ty = round(im.shape[0] * (1 - margin_scale) - h_word)
    else:
        # defalut top
        ty = round(im.shape[0] * margin_scale)

    draw.text((tx, ty), text, font=fontStyle, fill=tuple(color))
    return np.array(impil)
