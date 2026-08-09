#!/usr/bin/env python3
"""Generate transparent HDR angular-grid EXR textures.

Styles:
  panorama       Seamless 2:1 equirectangular 360-degree panorama.
  cubemap-strip  One-row cubemap strip ordered -X, +Z, +Y, -Y, +X, -Z.

The output is float32, premultiplied RGBA OpenEXR:
  RGB = HDR grid colour
  A   = anti-aliased grid mask

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

FACE_ORDER: Final = ("nx", "pz", "py", "ny", "px", "nz")
FLOAT_CHANNEL: Final = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))


@dataclass(frozen=True)
class GridSettings:
    longitude_step: float
    latitude_step: float
    line_pixels: float
    intensity: float
    color: np.ndarray


def periodic_distance(value: np.ndarray, period: float) -> np.ndarray:
    """Return each value's distance to the nearest multiple of period."""
    return np.abs((value + period * 0.5) % period - period * 0.5)


def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def render_grid(
    latitude: np.ndarray,
    longitude: np.ndarray,
    angular_pixel_size: float,
    settings: GridSettings,
) -> np.ndarray:
    """Render premultiplied RGBA grid data from spherical coordinates."""
    latitude_distance = periodic_distance(latitude, settings.latitude_step)
    longitude_distance = (
        periodic_distance(longitude, settings.longitude_step)
        * np.maximum(np.cos(latitude), 1e-5)
    )

    distance = np.minimum(latitude_distance, longitude_distance)
    half_line_width = settings.line_pixels * angular_pixel_size * 0.5
    feather_width = angular_pixel_size * 0.8
    alpha = smoothstep01((half_line_width + feather_width - distance) / feather_width)

    rgba = np.zeros((*alpha.shape, 4), dtype=np.float32)
    rgba[..., :3] = alpha[..., None] * settings.color * settings.intensity
    rgba[..., 3] = alpha
    return rgba


def render_panorama(width: int, settings: GridSettings) -> np.ndarray:
    """Render a 2:1 equirectangular panorama."""
    if width < 4 or width % 2:
        raise ValueError("--width must be an even integer of at least 4")

    height = width // 2
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    longitude = 2.0 * math.pi * u[None, :] - math.pi
    latitude = math.pi * (0.5 - v[:, None])

    return render_grid(latitude, longitude, 2.0 * math.pi / width, settings)


def cubemap_directions(face: str, size: int) -> np.ndarray:
    """Return normalized right-handed, +Y-up directions for one cubemap face."""
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
    try:
        directions = np.stack(vectors[face], axis=-1)
    except KeyError as error:
        raise ValueError(f"Unknown cubemap face: {face}") from error

    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def render_cubemap_strip(face_size: int, settings: GridSettings) -> np.ndarray:
    """Render one strip ordered left-to-right as -X, +Z, +Y, -Y, +X, -Z."""
    if face_size < 4:
        raise ValueError("--face-size must be at least 4")

    faces = []
    for face in FACE_ORDER:
        directions = cubemap_directions(face, face_size)
        latitude = np.arcsin(np.clip(directions[..., 1], -1.0, 1.0))
        longitude = np.arctan2(directions[..., 2], directions[..., 0])
        faces.append(
            render_grid(latitude, longitude, (math.pi * 0.5) / face_size, settings)
        )

    return np.concatenate(faces, axis=1)


def write_rgba_exr(path: Path, image: np.ndarray) -> None:
    """Write a float32, premultiplied-RGBA OpenEXR image."""
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("Expected an H x W x 4 RGBA image")

    height, width, _ = image.shape
    image = np.ascontiguousarray(image, dtype=np.float32)
    header = OpenEXR.Header(width, height)
    header["channels"] = {name: FLOAT_CHANNEL for name in "RGBA"}

    output = OpenEXR.OutputFile(str(path), header)
    try:
        output.writePixels({
            name: image[..., index].tobytes()
            for index, name in enumerate("RGBA")
        })
    finally:
        output.close()


def parse_rgb_255(value: str) -> np.ndarray:
    """Parse R,G,B sRGB values in 0..255 and convert them to linear RGB."""
    try:
        srgb = np.asarray(
            [float(part.strip()) for part in value.split(",")],
            dtype=np.float32,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--color must be three comma-separated numbers"
        ) from error

    if srgb.shape != (3,) or np.any(srgb < 0.0) or np.any(srgb > 255.0):
        raise argparse.ArgumentTypeError(
            "--color must contain exactly three values in the range 0..255"
        )

    srgb /= 255.0

    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("panorama", "cubemap"), default="panorama")
    parser.add_argument("--output", type=Path, default=Path("angle_coordinate_grid.exr"))
    parser.add_argument("--width", type=int, default=4096, help="Panorama width; height is width / 2")
    parser.add_argument("--face-size", type=int, default=2048, help="Width/height of every cubemap face")
    parser.add_argument("--longitude-step", type=float, default=15.0, help="Grid longitude spacing, degrees")
    parser.add_argument("--latitude-step", type=float, default=15.0, help="Grid latitude spacing, degrees")
    parser.add_argument("--line-pixels", type=float, default=1.0, help="Grid line width, pixels")
    parser.add_argument("--intensity", type=float, default=8.0, help="HDR grid brightness multiplier")
    parser.add_argument(
        "--color",
        type=parse_rgb_255,
        default=parse_rgb_255("4,255,20"),
        metavar="R,G,B",
        help="Grid colour in 0..255 RGB notation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.longitude_step <= 0.0 or args.latitude_step <= 0.0:
        raise SystemExit("error: grid steps must be positive")
    if args.line_pixels <= 0.0 or args.intensity < 0.0:
        raise SystemExit("error: --line-pixels must be positive and --intensity cannot be negative")

    settings = GridSettings(
        longitude_step=math.radians(args.longitude_step),
        latitude_step=math.radians(args.latitude_step),
        line_pixels=args.line_pixels,
        intensity=args.intensity,
        color=args.color,
    )

    if args.style == "panorama":
        # 2:1 equirectangular panorama; premultiplied alpha is the angular-grid mask
        image = render_panorama(args.width, settings)
    else:
        # Cubemap strip order: -X +Z +Y -Y +X -Z; premultiplied alpha is the angular-grid mask
        image = render_cubemap_strip(args.face_size, settings)

    write_rgba_exr(args.output, image)
    print(f"Wrote {args.output}: {image.shape[1]}x{image.shape[0]} float32 RGBA ({args.style})")


if __name__ == "__main__":
    main()
