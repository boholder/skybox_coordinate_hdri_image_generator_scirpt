#!/usr/bin/env python3

"""Generate one float32 EXR cubemap strip, ordered left-to-right:
    -X, +Z, +Y, -Y, +X, -Z

Output dimensions are (6 * face_size) x face_size. Each square is a complete
cubemap face. This is a custom scroll/strip layout, not a standard horizontal
cross; select a custom six-face import layout or split it before importing if
your engine does not support cube strips.
"""

import argparse
import math
from pathlib import Path

import Imath
import OpenEXR
import numpy as np


def periodic_distance(x: np.ndarray, period: float) -> np.ndarray:
    return np.abs((x + 0.5 * period) % period - 0.5 * period)


def directions_for_face(face: str, size: int) -> np.ndarray:
    # u increases left -> right; v maps image top -> bottom to +1 -> -1.
    u = 2.0 * (np.arange(size, dtype=np.float32) + 0.5) / size - 1.0
    v = 1.0 - 2.0 * (np.arange(size, dtype=np.float32) + 0.5) / size
    u, v = np.meshgrid(u, v)
    source = {
        "px": np.stack((np.ones_like(u), v, -u), axis=-1),
        "nx": np.stack((-np.ones_like(u), v, u), axis=-1),
        "py": np.stack((u, np.ones_like(u), -v), axis=-1),
        "ny": np.stack((u, -np.ones_like(u), v), axis=-1),
        "pz": np.stack((u, v, np.ones_like(u)), axis=-1),
        "nz": np.stack((-u, v, -np.ones_like(u)), axis=-1),
    }[face]
    return source / np.linalg.norm(source, axis=-1, keepdims=True)


def render_face(
    face: str,
    size: int,
    longitude_step: float,
    latitude_step: float,
    line_pixels: float,
    intensity: float,
    color: list[float]
) -> np.ndarray:
    direction = directions_for_face(face, size)
    x = direction[..., 0]
    y = direction[..., 1]
    z = direction[..., 2]

    latitude = np.arcsin(np.clip(y, -1.0, 1.0))
    longitude = np.arctan2(z, x)

    latitude_distance = periodic_distance(latitude, latitude_step)

    # Scale longitude-line thickness by cos(latitude) so it remains angular.
    longitude_distance = (
        periodic_distance(longitude, longitude_step)
        * np.maximum(np.cos(latitude), 1e-5)
    )

    distance = np.minimum(latitude_distance, longitude_distance)

    angular_pixel = (math.pi / 2.0) / size
    half_width = line_pixels * angular_pixel * 0.5
    feather = angular_pixel * 0.8

    # 0.0 outside the grid, 1.0 in the grid center.
    alpha = np.clip(
        (half_width + feather - distance) / feather,
        0.0,
        1.0,
    )

    # Smooth anti-aliased grid edge.
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    # Premultiplied-alpha RGBA:
    # RGB is exactly 0 where alpha is 0, so the background is fully transparent.]
    green = np.array(color, dtype=np.float32) * intensity

    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[..., 0:3] = alpha[..., None] * green
    rgba[..., 3] = alpha

    return rgba


def write_exr(path: Path, image: np.ndarray) -> None:
    height, width, channel_count = image.shape

    if channel_count != 4:
        raise ValueError(
            f"Expected RGBA image with four channels, got {channel_count}"
        )

    header = OpenEXR.Header(width, height)
    float_channel = Imath.Channel(
        Imath.PixelType(Imath.PixelType.FLOAT)
    )

    header["channels"] = {
        "R": float_channel,
        "G": float_channel,
        "B": float_channel,
        "A": float_channel,
    }

    output = OpenEXR.OutputFile(str(path), header)

    output.writePixels({
        "R": image[..., 0].astype(np.float32).tobytes(),
        "G": image[..., 1].astype(np.float32).tobytes(),
        "B": image[..., 2].astype(np.float32).tobytes(),
        "A": image[..., 3].astype(np.float32).tobytes(),
    })

    output.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="angle_coordinate_grid.exr")
    parser.add_argument("--face-size", type=int, default=2048)
    parser.add_argument("--longitude-step", type=float, default=15.0)
    parser.add_argument("--latitude-step", type=float, default=15.0)
    parser.add_argument("--line-pixels", type=float, default=1.25)
    parser.add_argument("--intensity", type=float, default=8.0)
    parser.add_argument("--color", type=str, default="4,255,20")
    args = parser.parse_args()
    if args.face_size < 4:
        parser.error("--face-size must be at least 4")
    if args.longitude_step <= 0 or args.latitude_step <= 0:
        parser.error("Grid steps must be positive")

    longitude_step = math.radians(args.longitude_step)
    latitude_step = math.radians(args.latitude_step)
    order = ("nx", "pz", "py", "ny", "px", "nz")
    grid_color = [float(x)/255 for x in args.color.split(",")]
    faces = [render_face(face, args.face_size, longitude_step, latitude_step,
                         args.line_pixels, args.intensity, grid_color) for face in order]
    strip = np.concatenate(faces, axis=1)
    write_exr(Path(args.output), strip)
    print(f"Wrote {args.output}: {strip.shape[1]}x{strip.shape[0]} float32 RGB cubemap strip (-X +Z +Y -Y +X -Z)")


if __name__ == "__main__":
    main()
