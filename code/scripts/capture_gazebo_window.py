#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import struct
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageGrab


XWD_FILE_VERSION = 7
HEADER_FIELDS = 25
HEADER_BYTES = HEADER_FIELDS * 4


def find_window_id(title_pattern: str) -> str | None:
    output = subprocess.check_output(
        ["xwininfo", "-root", "-tree"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    pattern = re.compile(title_pattern, re.IGNORECASE)
    for line in output.splitlines():
        match = re.match(r"\s*(0x[0-9a-fA-F]+)\s+\"([^\"]*)\"", line)
        if not match or not pattern.search(match.group(2)):
            continue
        geom_match = re.search(r"\s(\d+)x(\d+)[+-]\d+[+-]\d+", line)
        if geom_match:
            width = int(geom_match.group(1))
            height = int(geom_match.group(2))
            if width < 100 or height < 100:
                continue
            return match.group(1)
    return None


def window_geometry(window_id: str) -> tuple[int, int, int, int]:
    output = subprocess.check_output(
        ["xwininfo", "-id", window_id],
        text=True,
        stderr=subprocess.STDOUT,
    )
    values: dict[str, int] = {}
    patterns = {
        "x": r"Absolute upper-left X:\s+(-?\d+)",
        "y": r"Absolute upper-left Y:\s+(-?\d+)",
        "width": r"Width:\s+(\d+)",
        "height": r"Height:\s+(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            raise RuntimeError(f"Could not parse {key} from xwininfo for {window_id}")
        values[key] = int(match.group(1))
    return values["x"], values["y"], values["width"], values["height"]


def capture_xwd(window_id: str, output_path: Path, frame: bool = False) -> None:
    cmd = ["xwd", "-silent", "-id", window_id, "-out", str(output_path)]
    if frame:
        cmd.insert(2, "-frame")
    subprocess.check_call(cmd)


def capture_root_xwd(output_path: Path) -> None:
    subprocess.check_call(["xwd", "-silent", "-root", "-out", str(output_path)])


def capture_pil_window(window_id: str, output_path: Path) -> None:
    x, y, width, height = window_geometry(window_id)
    image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def capture_import_window(window_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["import", "-window", window_id, str(output_path)])


def xwd_to_png(xwd_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            ["convert", str(xwd_path), str(png_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    image = read_xwd_image(xwd_path)
    image.save(png_path)


def read_xwd_image(path: Path) -> Image.Image:
    data = path.read_bytes()
    header, endian = _parse_header(data)
    (
        header_size,
        _file_version,
        _pixmap_format,
        _pixmap_depth,
        pixmap_width,
        pixmap_height,
        _xoffset,
        byte_order,
        _bitmap_unit,
        _bitmap_bit_order,
        _bitmap_pad,
        bits_per_pixel,
        bytes_per_line,
        _visual_class,
        red_mask,
        green_mask,
        blue_mask,
        _bits_per_rgb,
        _colormap_entries,
        ncolors,
        _window_width,
        _window_height,
        _window_x,
        _window_y,
        _window_bdrwidth,
    ) = header
    color_table_bytes = ncolors * 12
    pixel_offset = header_size + color_table_bytes
    if pixel_offset >= len(data):
        raise ValueError("XWD file does not contain pixel data")
    if bits_per_pixel not in (24, 32):
        raise ValueError(f"Unsupported XWD bits_per_pixel={bits_per_pixel}")
    bytes_per_pixel = bits_per_pixel // 8
    pixel_endian = "little" if byte_order == 0 else "big"

    red_shift, red_bits = _mask_shift_and_bits(red_mask)
    green_shift, green_bits = _mask_shift_and_bits(green_mask)
    blue_shift, blue_bits = _mask_shift_and_bits(blue_mask)
    image = Image.new("RGB", (pixmap_width, pixmap_height))
    pixels = image.load()
    for y in range(pixmap_height):
        row_offset = pixel_offset + y * bytes_per_line
        for x in range(pixmap_width):
            start = row_offset + x * bytes_per_pixel
            raw = data[start:start + bytes_per_pixel]
            value = int.from_bytes(raw, pixel_endian, signed=False)
            pixels[x, y] = (
                _extract_channel(value, red_mask, red_shift, red_bits),
                _extract_channel(value, green_mask, green_shift, green_bits),
                _extract_channel(value, blue_mask, blue_shift, blue_bits),
            )
    return image


def _parse_header(data: bytes) -> tuple[tuple[int, ...], str]:
    for endian in (">", "<"):
        if len(data) < HEADER_BYTES:
            raise ValueError("XWD file is too small")
        header = struct.unpack(f"{endian}{HEADER_FIELDS}I", data[:HEADER_BYTES])
        header_size = header[0]
        version = header[1]
        width = header[4]
        height = header[5]
        bits_per_pixel = header[11]
        if (
            version == XWD_FILE_VERSION
            and HEADER_BYTES <= header_size <= len(data)
            and 0 < width < 20000
            and 0 < height < 20000
            and bits_per_pixel in (24, 32)
        ):
            return header, endian
    raise ValueError("Unsupported or invalid XWD header")


def _mask_shift_and_bits(mask: int) -> tuple[int, int]:
    if mask == 0:
        return 0, 0
    shift = 0
    while ((mask >> shift) & 1) == 0:
        shift += 1
    bits = 0
    while ((mask >> (shift + bits)) & 1) == 1:
        bits += 1
    return shift, bits


def _extract_channel(value: int, mask: int, shift: int, bits: int) -> int:
    if mask == 0 or bits == 0:
        return 0
    channel = (value & mask) >> shift
    max_value = (1 << bits) - 1
    if max_value <= 0:
        return 0
    return int(round(channel * 255 / max_value))


def capture_window_png(
    output_path: Path,
    title_pattern: str,
    timeout_sec: float,
    frame: bool,
) -> str:
    deadline = time.time() + timeout_sec
    window_id = None
    while time.time() <= deadline:
        window_id = find_window_id(title_pattern)
        if window_id:
            break
        time.sleep(0.5)
    if not window_id:
        raise RuntimeError(f"No X11 window matched pattern: {title_pattern}")
    xwd_path = output_path.with_suffix(".xwd")
    try:
        try:
            capture_xwd(window_id, xwd_path, frame=frame)
        except subprocess.CalledProcessError:
            if not frame:
                raise
            xwd_path.unlink(missing_ok=True)
            capture_xwd(window_id, xwd_path, frame=False)
    except subprocess.CalledProcessError:
        xwd_path.unlink(missing_ok=True)
        try:
            capture_root_xwd(xwd_path)
        except subprocess.CalledProcessError:
            xwd_path.unlink(missing_ok=True)
            try:
                capture_import_window(window_id, output_path)
                return window_id
            except (FileNotFoundError, subprocess.CalledProcessError):
                output_path.unlink(missing_ok=True)
            capture_pil_window(window_id, output_path)
            return window_id
    try:
        xwd_to_png(xwd_path, output_path)
    finally:
        xwd_path.unlink(missing_ok=True)
    return window_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a Gazebo X11 window as PNG.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title-pattern", default="Gazebo|Ignition")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--frame", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        window_id = capture_window_png(
            Path(args.output),
            title_pattern=args.title_pattern,
            timeout_sec=args.timeout,
            frame=args.frame,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Captured {window_id} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
