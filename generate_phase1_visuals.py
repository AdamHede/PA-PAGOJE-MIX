#!/usr/bin/env python3
"""
Phase 1 Visual Generator
Generates stage visuals for the first three tracks of the PA-PAGØJE festival mix.

Design references (VISUAL_PLAN_COMPLETE.md):
  1. Se den lille kattekilling — fluid kitten silhouette, exploratory motion, particle dissolve.
  2. I en kælder sort som kul — descending geometric cellar, wireframe glow, architectural depth.
  3. Hundred' mus med haler på — minimalist particle trails forming emergent mandala.

Outputs one rendered video per track, each with entry/exit fades for live mixing flexibility.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import wave
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

# === Global configuration ===
WIDTH = 1280
HEIGHT = 720
FPS = 24
FADE_SECONDS = 1.5

OUTPUT_ROOT = "visuals_phase1"
AUDIO_DIR = os.path.join(OUTPUT_ROOT, "audio_segments")
FRAMES_DIR = os.path.join(OUTPUT_ROOT, "frames")
RENDERS_DIR = os.path.join(OUTPUT_ROOT, "renders")
METADATA_DIR = os.path.join(OUTPUT_ROOT, "metadata")


# === Utility helpers ===
def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def extract_audio_segment(
    mix_path: str, start: float, duration: float, output_path: str, *, overwrite: bool = False
) -> None:
    if os.path.exists(output_path) and not overwrite:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        mix_path,
        "-t",
        f"{duration:.6f}",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "1",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def load_audio_envelope(audio_path: str, duration: float, fps: int) -> np.ndarray:
    """Return normalized RMS amplitude per video frame."""
    with wave.open(audio_path, "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale

    samples_per_frame = max(1, int(sample_rate / fps))
    total_frames = int(math.ceil(duration * fps))

    envelope = np.zeros(total_frames, dtype=np.float32)
    for i in range(total_frames):
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame
        segment = samples[start_idx:end_idx]
        if segment.size == 0:
            break
        rms = math.sqrt(float(np.mean(segment**2)))
        envelope[i] = rms

    max_val = float(np.max(envelope)) or 1.0
    envelope /= max_val

    kernel = np.array([0.2, 0.3, 0.0, 0.3, 0.2], dtype=np.float32)
    padded = np.pad(envelope, (2, 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def apply_global_fade(img: Image.Image, frame_idx: int, total_frames: int) -> Image.Image:
    fade_frames = int(FADE_SECONDS * FPS)
    fade_in_factor = 1.0
    fade_out_factor = 1.0

    if frame_idx < fade_frames:
        fade_in_factor = frame_idx / max(fade_frames, 1)
    if frame_idx > total_frames - fade_frames:
        remaining = total_frames - frame_idx
        fade_out_factor = remaining / max(fade_frames, 1)

    factor = min(fade_in_factor, fade_out_factor)
    if factor >= 0.999:
        return img

    arr = np.array(img).astype(np.float32)
    arr *= np.clip(factor, 0.0, 1.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_glow_layer(img: Image.Image, radius: float, intensity: float) -> Image.Image:
    blur = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.blend(img, blur, intensity)


def add_grain(img: Image.Image, strength: float = 0.04) -> Image.Image:
    noise = np.random.normal(0.0, strength, (HEIGHT, WIDTH, 1)).astype(np.float32)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def offset_image(image: Image.Image, dx: int, dy: int, fill_color: Tuple[int, ...] | int | None = None) -> Image.Image:
    """Shift an image without wraparound, filling newly exposed regions."""
    if fill_color is None:
        if image.mode in ("L", "1"):
            fill_color = 0
        elif image.mode == "RGBA":
            fill_color = (0, 0, 0, 0)
        else:
            fill_color = (0, 0, 0)

    w, h = image.size
    background = Image.new(image.mode, (w, h), fill_color)
    background.paste(image, (dx, dy))
    return background


# === Track-specific visual classes ===
class CatLiquidVisual:
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(-1.0, 1.0, WIDTH), np.linspace(-1.0, 1.0, HEIGHT)
        )
        self.cat_mask, self.cat_edge, self.mask_coords = self._build_cat_masks()
        self.orbs = self._init_orbs()
        self.particle_state = None

    def _build_cat_masks(self) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(mask)

        cx = WIDTH * 0.45
        cy = HEIGHT * 0.58
        body_w = WIDTH * 0.24
        body_h = HEIGHT * 0.35
        head_r = WIDTH * 0.09

        draw.ellipse(
            [(cx - body_w / 2, cy - body_h / 2), (cx + body_w / 2, cy + body_h / 2)],
            fill=255,
        )

        head_center_y = cy - body_h * 0.55
        draw.ellipse(
            [(cx - head_r, head_center_y - head_r), (cx + head_r, head_center_y + head_r)],
            fill=255,
        )

        ear_offset = head_r * 0.8
        ear_height = head_r * 1.1
        draw.polygon(
            [
                (cx - ear_offset, head_center_y - head_r * 0.2),
                (cx - ear_offset - head_r * 0.4, head_center_y - head_r - ear_height),
                (cx - ear_offset + head_r * 0.2, head_center_y - head_r - ear_height * 0.2),
            ],
            fill=255,
        )
        draw.polygon(
            [
                (cx + ear_offset, head_center_y - head_r * 0.2),
                (cx + ear_offset + head_r * 0.4, head_center_y - head_r - ear_height),
                (cx + ear_offset - head_r * 0.2, head_center_y - head_r - ear_height * 0.2),
            ],
            fill=255,
        )

        tail_points = [
            (cx - body_w * 0.5, cy + body_h * 0.2),
            (cx - body_w * 0.7, cy + body_h * 0.05),
            (cx - body_w * 0.85, cy - body_h * 0.15),
            (cx - body_w * 0.7, cy - body_h * 0.35),
            (cx - body_w * 0.55, cy - body_h * 0.25),
        ]
        draw.line(tail_points, fill=255, width=int(body_w * 0.08))

        paw_radius = body_w * 0.12
        paw_y = cy + body_h * 0.35
        draw.ellipse(
            [(cx - paw_radius * 1.2, paw_y - paw_radius), (cx - paw_radius * 0.2, paw_y + paw_radius)],
            fill=255,
        )
        draw.ellipse(
            [(cx + paw_radius * 0.2, paw_y - paw_radius), (cx + paw_radius * 1.2, paw_y + paw_radius)],
            fill=255,
        )

        mask_blur = mask.filter(ImageFilter.GaussianBlur(radius=4))
        edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=2))
        coords = np.column_stack(np.nonzero(np.array(mask_blur) > 10)).astype(np.float32)
        return mask_blur, edge, coords

    def _init_orbs(self) -> List[Dict[str, float]]:
        random.seed(42)
        orbs: List[Dict[str, float]] = []
        cx = WIDTH * 0.48
        cy = HEIGHT * 0.6
        orbit_radius = WIDTH * 0.2
        for _ in range(8):
            angle = random.uniform(0, math.tau)
            orbs.append(
                {
                    "angle": angle,
                    "radius": orbit_radius * random.uniform(0.6, 1.2),
                    "speed": random.uniform(0.2, 0.45),
                    "phase": random.uniform(0, math.tau),
                    "color": random.choice(
                        [
                            (30, 140, 255, 160),
                            (120, 80, 255, 150),
                            (20, 200, 230, 170),
                        ]
                    ),
                    "center": (cx, cy),
                }
            )
        return orbs

    def _build_background(self, t: float, amplitude: float) -> Image.Image:
        phase = t * 0.35
        wave1 = np.sin(self.grid_x * 5.0 + phase * 2.3)
        wave2 = np.cos(self.grid_y * 6.5 - phase * 1.1)
        wave3 = np.sin((self.grid_x + self.grid_y) * 4.2 + phase * 3.1)
        field = wave1 + 0.5 * wave2 + 0.35 * wave3

        norm = (field - field.min()) / (np.ptp(field) + 1e-5)

        base_deep = np.array([10, 14, 30], dtype=np.float32)
        base_mid = np.array([22, 30, 70], dtype=np.float32)
        accent = np.array([35, 90, 150], dtype=np.float32)

        mix = base_deep + (base_mid - base_deep) * norm[..., None]
        mix += accent * (norm[..., None] ** 2) * (0.4 + amplitude * 0.5)
        mix = np.clip(mix, 0, 255).astype(np.uint8)
        img = Image.fromarray(mix, mode="RGB")

        vignette = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(vignette)
        draw.ellipse([(WIDTH * -0.1, HEIGHT * 0.1), (WIDTH * 1.1, HEIGHT * 1.3)], fill=255)
        vignette_blur = vignette.filter(ImageFilter.GaussianBlur(radius=200))
        img = Image.composite(img, Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 20)), vignette_blur)
        return img

    def _cat_fill_texture(self, t: float, amplitude: float) -> Image.Image:
        phase = t * 0.8
        swirl = np.sin(self.grid_x * 9.0 - phase * 2.6) + np.cos(self.grid_y * 7.5 + phase * 3.3)
        swirl += np.sin((self.grid_x - self.grid_y) * 5.2 + phase * 1.7)
        swirl_norm = (swirl - swirl.min()) / (np.ptp(swirl) + 1e-5)

        color_a = np.array([30, 120, 255], dtype=np.float32)
        color_b = np.array([160, 60, 255], dtype=np.float32)
        mix = color_a + (color_b - color_a) * swirl_norm[..., None]
        mix *= (0.45 + 0.35 * amplitude)
        mix = np.clip(mix, 0, 255).astype(np.uint8)
        return Image.fromarray(mix, mode="RGB")

    def _update_orbs(self, t: float, amplitude: float) -> List[Tuple[float, float, float, Tuple[int, int, int, int]]]:
        rendered = []
        for orb in self.orbs:
            speed = orb["speed"] * (0.7 + amplitude * 1.1)
            orb["angle"] += speed / FPS
            jitter = math.sin(t * 2.0 + orb["phase"]) * 0.04
            radius = orb["radius"] * (0.9 + 0.2 * math.sin(t * 1.3 + orb["phase"]))
            x = orb["center"][0] + math.cos(orb["angle"]) * radius
            y = orb["center"][1] + math.sin(orb["angle"] + jitter) * radius * 0.6
            size = 30 + amplitude * 40
            rendered.append((x, y, size, orb["color"]))
        return rendered

    def _init_particle_state(self) -> None:
        if self.particle_state is not None:
            return
        coords = self.mask_coords
        chosen_idx = np.random.choice(len(coords), size=min(len(coords), 4500), replace=False)
        selected = coords[chosen_idx]
        particles = []
        for y, x in selected:
            direction = np.array(
                [
                    np.clip((x - WIDTH * 0.48) / (WIDTH * 0.3), -1.0, 1.0),
                    np.clip((y - HEIGHT * 0.55) / (HEIGHT * 0.3), -1.0, 1.0),
                ]
            )
            direction += np.random.normal(scale=0.25, size=2)
            norm = np.linalg.norm(direction) + 1e-6
            direction /= norm
            speed = np.random.uniform(40.0, 110.0)
            particles.append({"pos": np.array([x, y], dtype=np.float32), "vel": direction * speed})
        self.particle_state = particles

    def _render_particles(self, base: Image.Image, t: float, progress: float, amplitude: float) -> Image.Image:
        self._init_particle_state()
        particles = self.particle_state
        canvas = np.array(base).astype(np.float32)
        dissolve = (progress - 0.7) / max(1.0 - 0.7, 1e-5)
        dissolve = np.clip(dissolve, 0.0, 1.0)

        for particle in particles:
            particle["pos"] += particle["vel"] * (0.015 + amplitude * 0.02)
            x, y = particle["pos"]
            if not (-100 <= x <= WIDTH + 100 and -100 <= y <= HEIGHT + 100):
                continue
            radius = 4 + amplitude * 10 * (1.0 - dissolve * 0.6)
            color = np.array([40 + dissolve * 210, 80 + dissolve * 130, 220 + dissolve * 30])
            self._draw_glow_circle(canvas, x, y, radius, color, alpha=0.7 * (1 - dissolve * 0.5))

        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        return Image.fromarray(canvas, mode="RGB")

    @staticmethod
    def _draw_glow_circle(
        canvas: np.ndarray, cx: float, cy: float, radius: float, color: np.ndarray, alpha: float
    ) -> None:
        min_x = int(max(cx - radius, 0))
        max_x = int(min(cx + radius + 1, WIDTH))
        min_y = int(max(cy - radius, 0))
        max_y = int(min(cy + radius + 1, HEIGHT))
        if min_x >= max_x or min_y >= max_y:
            return
        yy, xx = np.ogrid[min_y:max_y, min_x:max_x]
        distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.clip(1.0 - (distances / max(radius, 1e-5)) ** 2, 0.0, 1.0)
        for c in range(3):
            canvas[min_y:max_y, min_x:max_x, c] += color[c] * mask * alpha

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = self._build_background(t, amplitude)
        cat_texture = self._cat_fill_texture(t, amplitude)

        stage1_end = 0.32
        stage2_end = 0.7

        sway_x = math.sin(t * 0.6) * (8 + amplitude * 6)
        sway_y = math.sin(t * 0.85 + 1.1) * (5 + amplitude * 4)
        dx = int(round(sway_x))
        dy = int(round(sway_y))

        if progress < stage1_end:
            emerge = progress / max(stage1_end, 1e-5)
            cat = ImageEnhance.Brightness(cat_texture).enhance(0.4 + 0.8 * emerge)
            shifted_cat = offset_image(cat, dx, dy)
            shifted_mask = offset_image(self.cat_mask.point(lambda v: int(v * emerge)), dx, dy, 0)
            base = Image.composite(shifted_cat, base, shifted_mask)
        elif progress < stage2_end:
            orbiters = self._update_orbs(t, amplitude)
            cat = ImageEnhance.Color(cat_texture).enhance(1.4 + amplitude * 0.6)
            cat = ImageEnhance.Brightness(cat).enhance(0.7 + amplitude * 0.5)
            shifted_cat = offset_image(cat, dx, dy)
            shifted_mask = offset_image(self.cat_mask, dx, dy, 0)
            base = Image.composite(shifted_cat, base, shifted_mask)
            draw = ImageDraw.Draw(base, "RGBA")
            for x, y, size, color in orbiters:
                bbox = [(x - size, y - size), (x + size, y + size)]
                draw.ellipse(bbox, fill=color)
            shifted_edge = offset_image(self.cat_edge, dx, dy, 0)
            edge_glow = shifted_edge.filter(ImageFilter.GaussianBlur(radius=5))
            base = Image.composite(
                Image.new("RGB", (WIDTH, HEIGHT), (30, 5, 50)),
                base,
                edge_glow.point(lambda v: int(v * (0.35 + amplitude * 0.4))),
            )
        else:
            dissolve_img = self._render_particles(base, t, progress, amplitude)
            base_cat = offset_image(cat_texture, dx, dy)
            shifted_mask = offset_image(self.cat_mask, dx, dy, 0)
            fade_mask = shifted_mask.point(
                lambda v: int(
                    v
                    * max(
                        0.0,
                        (1.0 - (progress - stage2_end) / max(1.0 - stage2_end, 1e-5)),
                    )
                )
            )
            base = Image.composite(base_cat, dissolve_img, fade_mask)

        base = add_glow_layer(base, radius=12, intensity=0.25)
        base = add_grain(base, strength=0.02)
        base = apply_global_fade(base, frame_idx, total)
        return base


class BasementDescentVisual:
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        self.seed_shapes = self._init_shapes()

    def _init_shapes(self) -> List[Dict[str, float]]:
        shapes = []
        random.seed(84)
        for _ in range(20):
            shapes.append(
                {
                    "angle": random.uniform(0, math.tau),
                    "radius": random.uniform(0.15, 0.6),
                    "depth": random.uniform(0.0, 1.0),
                }
            )
        return shapes

    def _draw_layer(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        size: float,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        diamond = [
            (cx, cy - size),
            (cx + size, cy),
            (cx, cy + size),
            (cx - size, cy),
        ]
        draw.polygon(diamond, outline=color, width=thickness)

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (2, 4, 8))
        draw = ImageDraw.Draw(base, "RGB")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.55
        base_size = HEIGHT * 0.28
        depth_shift = progress * 8.0

        for idx in range(14):
            depth = idx + depth_shift
            scale = 1.0 / (0.8 + depth * 0.12)
            size = base_size * scale
            brightness = int(40 + 180 * math.exp(-depth * 0.18))
            color = (0, brightness, 255 if idx < 6 else 220)
            thickness = max(1, int(3 * scale + amplitude * 3))
            self._draw_layer(draw, cx, cy + depth * 12, size, color, thickness)

        pillar_color = (0, 120, 220)
        pillar_spacing = WIDTH * 0.12
        for side in (-1, 1):
            x = cx + side * pillar_spacing
            draw.line([(x, cy - base_size * 1.8), (x, HEIGHT)], fill=pillar_color, width=3)

        if progress > 0.25:
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            for shape in self.seed_shapes:
                motion = progress * (0.5 + amplitude)
                depth_factor = (shape["depth"] + motion) % 1.0
                radius = base_size * shape["radius"] * (0.6 + depth_factor * 0.6)
                angle = shape["angle"] + t * (0.2 + amplitude * 0.8)
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius * 0.6 - depth_factor * HEIGHT * 0.3
                glow = int(60 + 160 * (1.0 - depth_factor))
                size = 18 + 40 * (1.0 - depth_factor)
                bbox = [(x - size, y - size), (x + size, y + size)]
                overlay_draw.arc(bbox, start=0, end=360, fill=(glow, glow, 255, 150), width=3)
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=4))
            base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

        grid_img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        grid_draw = ImageDraw.Draw(grid_img)
        grid_spacing = 90
        sway = math.sin(t * 0.6) * 40
        for y in range(int(cy), HEIGHT + grid_spacing, grid_spacing):
            offset = (y - cy) * 0.12 + sway
            brightness = int(20 + 80 * (1 - (y - cy) / HEIGHT))
            grid_draw.line(
                [(cx - base_size * 3 + offset, y), (cx + base_size * 3 - offset, y)],
                fill=(0, brightness, 200),
                width=1,
            )
        grid_img = grid_img.filter(ImageFilter.GaussianBlur(radius=3))
        base = Image.blend(base, grid_img, 0.4)

        base = add_glow_layer(base, radius=18, intensity=0.35 + amplitude * 0.1)
        base = add_grain(base, strength=0.025)
        base = apply_global_fade(base, frame_idx, total)
        return base


class MiceTrailsVisual:
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        self.mice = self._init_mice()
        self.trail_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        self.symmetry_strength = 0.0

    def _init_mice(self):
        random.seed(1729)
        mice = []
        for _ in range(36):
            mice.append(
                {
                    "radius": random.uniform(0.12, 0.48),
                    "speed": random.uniform(0.7, 1.4),
                    "phase": random.uniform(0, math.tau),
                    "curviness": random.uniform(0.6, 1.4),
                    "angle": random.uniform(0, math.tau),
                    "color": np.array(
                        random.choice(
                            [
                                (0.1, 0.9, 0.6),
                                (0.2, 0.8, 0.7),
                                (0.0, 1.0, 0.5),
                            ]
                        ),
                        dtype=np.float32,
                    ),
                }
            )
        return mice

    def _update_mouse(self, mouse, t: float, amplitude: float) -> Tuple[float, float]:
        mouse["angle"] += mouse["speed"] * (0.5 + amplitude * 1.3) / FPS
        mouse["phase"] += 0.01 * mouse["curviness"]
        base_radius = mouse["radius"]
        radius = base_radius * (HEIGHT * 0.45)
        cx = WIDTH * 0.5
        cy = HEIGHT * 0.5
        x = cx + math.cos(mouse["angle"]) * radius * (0.9 + 0.1 * math.sin(mouse["phase"]))
        y = cy + math.sin(mouse["angle"] * mouse["curviness"]) * radius * 0.6
        x += math.sin(t * 0.9 + mouse["phase"]) * 40
        y += math.cos(t * 1.3 + mouse["phase"]) * 30
        return x, y

    def _deposit(self, x: float, y: float, color: np.ndarray, intensity: float) -> None:
        radius = 8 + intensity * 12
        min_x = int(max(x - radius, 0))
        max_x = int(min(x + radius + 1, WIDTH))
        min_y = int(max(y - radius, 0))
        max_y = int(min(y + radius + 1, HEIGHT))
        if min_x >= max_x or min_y >= max_y:
            return
        yy, xx = np.ogrid[min_y:max_y, min_x:max_x]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = np.clip(1.0 - (dist / max(radius, 1e-5)) ** 1.8, 0.0, 1.0)
        self.trail_canvas[min_y:max_y, min_x:max_x, :] += color * intensity * mask[..., None]

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        fade_rate = 0.96 - amplitude * 0.08
        self.trail_canvas *= fade_rate

        if progress > 0.55:
            self.symmetry_strength = min(1.0, self.symmetry_strength + 0.01)

        points = []
        for mouse in self.mice:
            x, y = self._update_mouse(mouse, t, amplitude)
            intensity = 0.6 + amplitude * 0.8
            color = mouse["color"] * (0.6 + 0.4 * math.sin(t * 1.5 + mouse["phase"] * 2))
            color = np.clip(color, 0.0, 1.0)
            self._deposit(x, y, color, intensity)
            points.append((x, y, color, intensity))

        if self.symmetry_strength > 0:
            for x, y, color, intensity in points:
                mirrored = [(WIDTH - x, y), (x, HEIGHT - y), (WIDTH - x, HEIGHT - y)]
                for mx, my in mirrored:
                    self._deposit(mx, my, color, intensity * (0.6 + 0.4 * self.symmetry_strength))

        canvas = np.clip(self.trail_canvas, 0.0, 1.0)
        canvas = canvas ** 0.9
        arr = (canvas * 255).astype(np.uint8)

        bg = Image.new("RGB", (WIDTH, HEIGHT), (4, 20, 25))
        gradient = Image.linear_gradient("L").resize((1, HEIGHT))
        gradient = gradient.filter(ImageFilter.GaussianBlur(radius=8))
        gradient_color = Image.new("RGB", (1, HEIGHT), (10, 90, 70))
        gradient_color.putalpha(gradient.point(lambda v: int(v * 0.9)))
        bg = Image.alpha_composite(bg.convert("RGBA"), gradient_color.resize((WIDTH, HEIGHT))).convert("RGB")

        trails = Image.fromarray(arr, mode="RGB")
        combined = ImageChops.add_modulo(bg, trails)

        combined = add_glow_layer(combined, radius=14, intensity=0.3 + amplitude * 0.1)
        combined = add_grain(combined, strength=0.03)
        combined = apply_global_fade(combined, frame_idx, total)
        return combined


# === Track configuration ===
@dataclass
class TrackConfig:
    key: str
    title: str
    start: float
    duration: float
    visual_class: type
    description: str


TRACKS: Dict[str, TrackConfig] = {
    "01": TrackConfig(
        key="01",
        title="Se den lille kattekilling (Deep House)",
        start=0.0,
        duration=143.319979,
        visual_class=CatLiquidVisual,
        description="Fluid kitten silhouette with underwater aesthetic and particle dissolve.",
    ),
    "02": TrackConfig(
        key="02",
        title="I en kælder sort som kul (Dub techno)",
        start=142.7441723356009,
        duration=181.959979,
        visual_class=BasementDescentVisual,
        description="Descending wireframe architecture with glowing cyan outlines.",
    ),
    "03": TrackConfig(
        key="03",
        title="Hundred' mus med haler på (Microhouse)",
        start=314.29213151927434,
        duration=129.959979,
        visual_class=MiceTrailsVisual,
        description="Generative particle trails accumulating into synchronous mandala.",
    ),
}


def render_track(config: TrackConfig, mix_path: str, overwrite: bool = False) -> Dict[str, str]:
    ensure_dirs(OUTPUT_ROOT, AUDIO_DIR, FRAMES_DIR, RENDERS_DIR, METADATA_DIR)

    slug = slugify(config.title)
    audio_path = os.path.join(AUDIO_DIR, f"{config.key}_{slug}.wav")
    frames_path = os.path.join(FRAMES_DIR, f"{config.key}_{slug}")
    output_video = os.path.join(RENDERS_DIR, f"{config.key}_{slug}.mp4")
    metadata_path = os.path.join(METADATA_DIR, f"{config.key}_{slug}.json")

    ensure_dirs(frames_path)
    for filename in os.listdir(frames_path):
        if filename.startswith("frame_") and filename.endswith(".png"):
            os.remove(os.path.join(frames_path, filename))

    print("=" * 100)
    print(f"[TRACK {config.key}] {config.title}")
    print(f"Duration: {config.duration:.2f}s | Frames @ {FPS}fps: {int(math.ceil(config.duration * FPS))}")
    print(f"Output: {output_video}")
    print("=" * 100)

    extract_audio_segment(mix_path, config.start, config.duration, audio_path, overwrite=overwrite)
    envelope = load_audio_envelope(audio_path, config.duration, FPS)

    visual = config.visual_class(config.duration, envelope)
    total_frames = visual.total_frames

    for frame_idx in range(total_frames):
        if frame_idx % max(1, total_frames // 40) == 0:
            pct = frame_idx / max(total_frames - 1, 1) * 100
            print(f"  Rendering frame {frame_idx:05d}/{total_frames:05d} ({pct:5.1f}%)")
        frame = visual.render_frame(frame_idx)
        frame.save(os.path.join(frames_path, f"frame_{frame_idx:05d}.png"))

    print("  ✓ Frames complete. Encoding video...")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        os.path.join(frames_path, "frame_%05d.png"),
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        output_video,
    ]
    subprocess.run(cmd, check=True)
    print("  ✓ Video render complete.")

    metadata = {
        "track_key": config.key,
        "title": config.title,
        "description": config.description,
        "duration_seconds": config.duration,
        "fps": FPS,
        "frames": total_frames,
        "audio_path": audio_path,
        "frames_path": frames_path,
        "video_path": output_video,
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PA-PAGØJE phase 1 visuals.")
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["01", "02", "03"],
        help="Track IDs to render (01, 02, 03).",
    )
    parser.add_argument("--mix", default="PA-PAGOJE_Festival_Mix2.wav", help="Path to the master audio mix.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached audio segments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mix_path = args.mix
    if not os.path.exists(mix_path):
        raise FileNotFoundError(f"Mix audio not found: {mix_path}")

    selected = []
    for key in args.tracks:
        if key not in TRACKS:
            raise ValueError(f"Unknown track key: {key} (expected among {sorted(TRACKS.keys())})")
        selected.append(TRACKS[key])

    manifest: List[Dict[str, str]] = []
    for config in selected:
        metadata = render_track(config, mix_path, overwrite=args.overwrite)
        manifest.append(metadata)

    manifest_path = os.path.join(METADATA_DIR, "phase1_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("\nAll requested tracks rendered successfully.")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
