import numpy as np


from fastcv import draw_text, cast

if __name__ == "__main__":
    black_im = cast(np.zeros((200, 400, 4), dtype=np.uint8), type="pil")
    # text_im = draw_text(black_im, "fastcv", color=(17, 49, 154, 255), font_scale=0.8, align="center",
    text_im = draw_text(
        black_im, 
        "fastcv", 
        color=(17, 49, 154, 255), font_scale=0.8, align="center",
        font_type="SmileySans-Oblique",
        # font_type="Futura-1",
        stroke_width=2,
        # spacing=-50,

    )
    text_im.show()
    text_im.save("fastcv.webp")

