import importlib.util
import struct
from pathlib import Path


def _load_capture_module():
    script = Path("scripts/capture_gazebo_window.py").resolve()
    spec = importlib.util.spec_from_file_location("capture_gazebo_window", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_xwd_to_png_converts_truecolor_pixels(tmp_path: Path) -> None:
    capture = _load_capture_module()
    xwd_path = tmp_path / "sample.xwd"
    png_path = tmp_path / "sample.png"
    window_name = b"sample\0"
    header_size = 25 * 4 + len(window_name)
    header = [
        header_size,  # header_size
        7,  # file_version
        2,  # pixmap_format = ZPixmap
        24,  # pixmap_depth
        2,  # pixmap_width
        1,  # pixmap_height
        0,  # xoffset
        0,  # byte_order = LSBFirst
        32,  # bitmap_unit
        0,  # bitmap_bit_order
        32,  # bitmap_pad
        32,  # bits_per_pixel
        8,  # bytes_per_line
        4,  # visual_class = TrueColor
        0x00FF0000,  # red_mask
        0x0000FF00,  # green_mask
        0x000000FF,  # blue_mask
        8,  # bits_per_rgb
        256,  # colormap_entries
        0,  # ncolors
        2,  # window_width
        1,  # window_height
        0,  # window_x
        0,  # window_y
        0,  # window_bdrwidth
    ]
    pixels = struct.pack("<II", 0x00112233, 0x00445566)
    xwd_path.write_bytes(struct.pack("<25I", *header) + window_name + pixels)

    capture.xwd_to_png(xwd_path, png_path)

    image = capture.Image.open(png_path)
    assert image.size == (2, 1)
    assert image.getpixel((0, 0)) == (0x11, 0x22, 0x33)
    assert image.getpixel((1, 0)) == (0x44, 0x55, 0x66)
