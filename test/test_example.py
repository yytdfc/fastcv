#!/usr/bin/env python
# encoding: utf-8

import cv2

from tqdm import tqdm
from fastcv import VideoLoader, VideoDumper, draw_text, resize


def test_videos():
    loader = VideoLoader("video_input.mp4")
    dumper1 = VideoDumper("video_output_avc1.mp4", codec="avc1")
    dumper2 = VideoDumper("video_output_mp4v.mp4", codec="mp4v")
    dumper3 = VideoDumper("video_output_png.mp4", dump_mode="png2video", crf=8)
    dumper4 = VideoDumper("video_output", dump_mode="png")
    for i, im in enumerate(tqdm(loader)):
        im_left = draw_text(im, i, align="bottom_left")
        im_center = draw_text(im, i, align="center")
        im_right = draw_text(im, i, align="right")
        im_left = resize(im_left, long=512, factor=8)
        print(im_left.shape)
        im_right = resize(im_right, short=512, factor=8)
        print(im_right.shape)
        im_center = resize(im_center, scale=0.5, factor=8)
        print(im_center.shape)
        dumper1.write(im_left)
        dumper2.write(im_center)
        dumper3.write(im_right)
        dumper4.write(im_right)

if __name__=='__main__':
    test_videos()
