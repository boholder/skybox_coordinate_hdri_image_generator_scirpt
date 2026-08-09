#!/usr/bin/env python3
"""Generate transparent HDR angular-grid EXR textures with lat/long labels.

Styles:
  panorama  Seamless 2:1 equirectangular panorama.
  cubemap   One-row cubemap strip: -X, +Z, +Y, -Y, +X, -Z.

Examples:
  python main.py --style panorama --output grid_360.exr --width 4096
  python main.py --style cubemap --output grid_cube.exr --face-size 1024
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import Imath
import OpenEXR
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

FACE_ORDER: Final = ("nx", "pz", "py", "ny", "px", "nz")
MIN_LABEL_FONT_SIZE: Final = 10


@dataclass(frozen=True)
class GridSettings:
    longitude_step: float
    latitude_step: float
    line_pixels: float
    intensity: float
    color: np.ndarray
    labels_enabled: bool
    label_font_size: int
    label_offset: int


def periodic_distance(value: np.ndarray, period: float) -> np.ndarray:
    return np.abs((value + period * 0.5) % period - period * 0.5)


def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def load_font(size: int):
    for path in (
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def grid_values(
    minimum: float, maximum: float, step: float, include_max: bool = True
) -> list[float]:
    first = math.ceil(minimum / step - 1e-7) * step
    limit = maximum + (1e-7 if include_max else -1e-7)
    count = max(0, int(math.floor((limit - first) / step)) + 1)
    return [first + index * step for index in range(count)]


def format_degrees(value: float) -> str:
    rounded = round(value)
    value_text = str(rounded) if abs(value - rounded) < 1e-6 else f"{value:g}"
    return f"{value_text}"


def label_text(latitude: float, longitude: float) -> str:
    return f"{format_degrees(latitude)}, {format_degrees(longitude)}"


def label_font_size(latitude: float, base_size: int) -> int:
    """Scale labels with longitude-circle spacing while keeping them readable."""
    scale = math.cos(math.radians(abs(latitude)))
    return max(MIN_LABEL_FONT_SIZE, round(base_size * scale))


def render_grid(
    latitude: np.ndarray,
    longitude: np.ndarray,
    angular_pixel: float,
    settings: GridSettings,
) -> np.ndarray:
    latitude_distance = periodic_distance(latitude, settings.latitude_step)
    longitude_distance = periodic_distance(
        longitude, settings.longitude_step
    ) * np.maximum(np.cos(latitude), 1e-5)
    distance = np.minimum(latitude_distance, longitude_distance)

    half_width = settings.line_pixels * angular_pixel * 0.5
    feather = angular_pixel * 0.8
    alpha = smoothstep01((half_width + feather - distance) / feather)

    rgba = np.zeros((*alpha.shape, 4), dtype=np.float32)
    rgba[..., :3] = alpha[..., None] * settings.color * settings.intensity
    rgba[..., 3] = alpha
    return rgba


def composite_text(
    image: np.ndarray, mask: Image.Image, settings: GridSettings
) -> np.ndarray:
    """Composite opaque text over premultiplied RGBA image data."""
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    text_rgb = alpha[..., None] * settings.color * settings.intensity
    image[..., :3] = text_rgb + image[..., :3] * (1.0 - alpha[..., None])
    image[..., 3] = alpha + image[..., 3] * (1.0 - alpha)
    return image


def draw_panorama_labels(image: np.ndarray, settings: GridSettings) -> np.ndarray:
    if not settings.labels_enabled:
        return image

    height, width, _ = image.shape
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    latitude_step = math.degrees(settings.latitude_step)
    longitude_step = math.degrees(settings.longitude_step)

    for latitude in grid_values(-90.0, 90.0, latitude_step):
        # The poles are single points, not one independent vertex per longitude.
        longitudes = (
            [0.0]
            if abs(abs(latitude) - 90.0) < 1e-6
            else grid_values(-180.0, 180.0, longitude_step, include_max=False)
        )
        for longitude in longitudes:
            x = int((longitude + 180.0) / 360.0 * width) % width
            y = int((90.0 - latitude) / 180.0 * height)
            text = label_text(latitude, longitude)
            font = load_font(label_font_size(latitude, settings.label_font_size))
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            x = min(x + settings.label_offset, width - (right - left) - 1)
            y = min(y + settings.label_offset, height - (bottom - top) - 1)
            draw.text((max(0, x), max(0, y)), text, fill=255, font=font)

    return composite_text(image, mask, settings)


def render_panorama(width: int, settings: GridSettings) -> np.ndarray:
    if width < 4 or width % 2:
        raise ValueError("--width must be an even integer of at least 4")

    height = width // 2
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    longitude = 2.0 * math.pi * u[None, :] - math.pi
    latitude = math.pi * (0.5 - v[:, None])
    image = render_grid(latitude, longitude, 2.0 * math.pi / width, settings)
    return draw_panorama_labels(image, settings)


def cubemap_directions(face: str, size: int) -> np.ndarray:
    u = 2.0 * (np.arange(size, dtype=np.float32) + 0.5) / size - 1.0
    v = 1.0 - 2.0 * (np.arange(size, dtype=np.float32) + 0.5) / size
    u, v = np.meshgrid(u, v)
    vectors = {
        "px": (np.ones_like(u), v, -u),
        "nx": (-np.ones_like(u), v, u),
        "py": (u, np.ones_like(u), -v),
        "ny": (u, -np.ones_like(u), v),
        "pz": (u, v, np.ones_like(u)),
        "nz": (-u, v, -np.ones_like(u)),
    }
    direction = np.stack(vectors[face], axis=-1)
    return direction / np.linalg.norm(direction, axis=-1, keepdims=True)


def cube_face_and_uv(direction: np.ndarray) -> tuple[str, float, float]:
    x, y, z = direction
    axis = int(np.argmax(np.abs(direction)))
    if axis == 0 and x >= 0:
        return "px", -z / x, y / x
    if axis == 0:
        return "nx", z / -x, y / -x
    if axis == 1 and y >= 0:
        return "py", x / y, -z / y
    if axis == 1:
        return "ny", x / -y, z / -y
    if z >= 0:
        return "pz", x / z, y / z
    return "nz", -x / -z, y / -z


def spherical_direction(latitude: float, longitude: float) -> np.ndarray:
    lat = math.radians(latitude)
    lon = math.radians(longitude)

    return np.array(
        (
            math.cos(lat) * math.cos(lon),
            math.sin(lat),
            math.cos(lat) * math.sin(lon),
        ),
        dtype=np.float32,
    )


def cube_face_pixel(
    latitude: float,
    longitude: float,
    face_size: int,
) -> tuple[str, float, float]:
    face, u, v = cube_face_and_uv(spherical_direction(latitude, longitude))

    x = (u + 1.0) * face_size * 0.5
    y = (1.0 - v) * face_size * 0.5

    return face, x, y


def latitude_tangent_angle(
    face: str,
    latitude: float,
    longitude: float,
    face_size: int,
) -> float:
    """Return the label rotation for the current cubemap face."""

    sample_offset = 0.05

    face_a, x_a, y_a = cube_face_pixel(
        latitude,
        longitude - sample_offset,
        face_size,
    )

    face_b, x_b, y_b = cube_face_pixel(
        latitude,
        longitude + sample_offset,
        face_size,
    )

    # Avoid unstable rotations when the tangent crosses a cube-face boundary.
    if face_a != face or face_b != face:
        return 0.0

    angle = math.degrees(math.atan2(y_b - y_a, x_b - x_a))

    # On the +Y/-Y cap faces, preserve the continuous latitude-circle tangent.
    if face in ("py", "ny"):
        return angle - 180.0

    # On the four side faces (+X, -X, +Z, -Z), keep text upright.
    if angle > 90.0:
        return angle - 180.0
    elif angle <= -90.0:
        return angle + 180.0
    else:
        return angle


def draw_rotated_text(
    target: Image.Image,
    x: int,
    y: int,
    text: str,
    font_size: int,
    angle: float,
    supersample: int = 4,
) -> Image.Image:
    """Draw a sharp, rotated label centered at x/y."""

    font = load_font(font_size * supersample)
    left, top, right, bottom = font.getbbox(text)

    text_width = right - left
    text_height = bottom - top
    padding = 6 * supersample

    high_res = Image.new(
        "L",
        (
            text_width + padding * 2,
            text_height + padding * 2,
        ),
        0,
    )

    high_res_draw = ImageDraw.Draw(high_res)
    high_res_draw.text(
        (padding - left, padding - top),
        text,
        fill=255,
        font=font,
    )

    # Rotate at high resolution, then downsample once.
    rotated_high_res = high_res.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )

    output_size = (
        max(1, round(rotated_high_res.width / supersample)),
        max(1, round(rotated_high_res.height / supersample)),
    )

    rotated = rotated_high_res.resize(
        output_size,
        resample=Image.Resampling.LANCZOS,
    )

    paste_x = int(x - rotated.width * 0.5)
    paste_y = int(y - rotated.height * 0.5)

    layer = Image.new("L", target.size, 0)
    layer.paste(rotated, (paste_x, paste_y))

    return ImageChops.lighter(target, layer)


def draw_cube_face_labels(
    image: np.ndarray,
    face: str,
    settings: GridSettings,
) -> np.ndarray:
    if not settings.labels_enabled:
        return image

    face_size = image.shape[0]
    mask = Image.new("L", (face_size, face_size), 0)

    latitude_step = math.degrees(settings.latitude_step)
    longitude_step = math.degrees(settings.longitude_step)

    for latitude in grid_values(-90.0, 90.0, latitude_step):
        is_pole = abs(abs(latitude) - 90.0) < 1e-6

        longitudes = (
            [0.0]
            if is_pole
            else grid_values(
                -180.0,
                180.0,
                longitude_step,
                include_max=False,
            )
        )

        for longitude in longitudes:
            target_face, x, y = cube_face_pixel(
                latitude,
                longitude,
                face_size,
            )

            if target_face != face:
                continue

            text = label_text(latitude, longitude)

            # Text baseline follows the local latitude-circle tangent.
            angle = (
                0.0
                if is_pole
                else latitude_tangent_angle(
                    face,
                    latitude,
                    longitude,
                    face_size,
                )
            )

            mask = draw_rotated_text(
                mask,
                int(x + settings.label_offset),
                int(y + settings.label_offset),
                text,
                label_font_size(latitude, settings.label_font_size),
                angle,
            )

    return composite_text(image, mask, settings)


def render_cubemap_strip(face_size: int, settings: GridSettings) -> np.ndarray:
    if face_size < 4:
        raise ValueError("--face-size must be at least 4")

    faces = []
    for face in FACE_ORDER:
        directions = cubemap_directions(face, face_size)
        latitude = np.arcsin(np.clip(directions[..., 1], -1.0, 1.0))
        longitude = np.arctan2(directions[..., 2], directions[..., 0])
        image = render_grid(latitude, longitude, (math.pi * 0.5) / face_size, settings)
        faces.append(draw_cube_face_labels(image, face, settings))
    return np.concatenate(faces, axis=1)


def write_rgba_exr(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("Expected an H x W x 4 RGBA image")

    height, width, _ = image.shape
    image = np.ascontiguousarray(image, dtype=np.float32)
    channel = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header = OpenEXR.Header(width, height)
    header["channels"] = {name: channel for name in "RGBA"}

    output = OpenEXR.OutputFile(str(path), header)
    try:
        output.writePixels(
            {name: image[..., index].tobytes() for index, name in enumerate("RGBA")}
        )
    finally:
        output.close()


def parse_rgb_255(value: str) -> np.ndarray:
    srgb = np.asarray(
        [float(part.strip()) for part in value.split(",")], dtype=np.float32
    )
    srgb /= 255.0
    return np.where(
        srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("panorama", "cubemap"), default="cubemap")
    parser.add_argument(
        "--output", type=Path, default=Path("grid.exr")
    )
    parser.add_argument(
        "--width", type=int, default=4096, help="Panorama width; height is width / 2"
    )
    parser.add_argument(
        "--face-size", type=int, default=2048, help="Width/height of every cubemap face"
    )
    parser.add_argument(
        "--longitude-step",
        type=float,
        default=15.0,
        help="Grid longitude spacing, degrees",
    )
    parser.add_argument(
        "--latitude-step",
        type=float,
        default=15.0,
        help="Grid latitude spacing, degrees",
    )
    parser.add_argument(
        "--line-pixels", type=float, default=1.0, help="Grid line width, pixels"
    )
    parser.add_argument(
        "--intensity", type=float, default=8.0, help="HDR grid brightness multiplier"
    )
    parser.add_argument(
        "--color",
        type=parse_rgb_255,
        default=parse_rgb_255("4,255,20"),
        metavar="R,G,B",
        help="Grid colour in 0..255 RGB notation",
    )
    parser.add_argument("--font-size", type=int, default=30)
    parser.add_argument("--label-offset", type=int, default=25)
    parser.add_argument("--no-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = GridSettings(
        longitude_step=math.radians(args.longitude_step),
        latitude_step=math.radians(args.latitude_step),
        line_pixels=args.line_pixels,
        intensity=args.intensity,
        color=args.color,
        labels_enabled=not args.no_labels,
        label_font_size=args.font_size,
        label_offset=args.label_offset,
    )

    if args.style == "panorama":
        # 2:1 equirectangular panorama; premultiplied alpha is the angular-grid mask
        image = render_panorama(args.width, settings)
    else:
        # Cubemap strip order: -X +Z +Y -Y +X -Z; premultiplied alpha is the angular-grid mask
        image = render_cubemap_strip(args.face_size, settings)

    write_rgba_exr(args.output, image)
    print(
        f"Wrote {args.output}: {image.shape[1]}x{image.shape[0]} float32 RGBA ({args.style})"
    )


if __name__ == "__main__":
    main()
