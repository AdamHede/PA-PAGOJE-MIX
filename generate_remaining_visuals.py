#!/usr/bin/env python3
"""
PA-PAGØJE Festival Visuals - Tracks 4-20
Implements visual generators for all remaining tracks based on VISUAL_PLAN_COMPLETE.md

Phase 2: First Build (tracks 4-8)
Phase 3: First Peak (tracks 9-13) - track 9 already done separately
Phase 4: Breather (tracks 14-15)
Phase 5: Second Build (tracks 16-17)
Phase 6: Final Peak (tracks 18-20)
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

OUTPUT_ROOT = "visuals_all_tracks"
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
        "ffmpeg", "-y", "-ss", f"{start:.6f}", "-i", mix_path,
        "-t", f"{duration:.6f}", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "1", output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


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


# === TRACK 4: Spørge Jørgen (Minimal Techno) - Single line art morphing animals ===
class LineArtAnimalsVisual:
    """Single continuous line drawing animals morphing"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        self.animal_shapes = self._define_animals()

    def _define_animals(self) -> List[List[Tuple[float, float]]]:
        """Define animals as normalized point sequences (0-1 coords)"""
        # Each animal is a sequence of (x, y) points in 0-1 space
        question = [(0.5, 0.3), (0.52, 0.25), (0.5, 0.2), (0.48, 0.25), (0.5, 0.3), (0.5, 0.5)]

        cow = [(0.3, 0.5), (0.4, 0.4), (0.5, 0.35), (0.6, 0.4), (0.7, 0.5), (0.65, 0.6), (0.55, 0.65), (0.45, 0.65), (0.35, 0.6)]

        pig = [(0.35, 0.45), (0.45, 0.4), (0.55, 0.4), (0.65, 0.45), (0.6, 0.55), (0.5, 0.6), (0.4, 0.55)]

        chicken = [(0.5, 0.35), (0.55, 0.4), (0.55, 0.5), (0.5, 0.6), (0.45, 0.65), (0.4, 0.6), (0.4, 0.5), (0.45, 0.4)]

        horse = [(0.3, 0.6), (0.35, 0.5), (0.4, 0.4), (0.5, 0.35), (0.6, 0.4), (0.65, 0.5), (0.7, 0.6), (0.65, 0.65)]

        cat = [(0.45, 0.4), (0.5, 0.35), (0.55, 0.4), (0.6, 0.5), (0.55, 0.6), (0.45, 0.6), (0.4, 0.5)]

        return [question, cow, pig, chicken, horse, cat, question]

    def _morph_shapes(self, shape1: List, shape2: List, blend: float) -> List[Tuple[float, float]]:
        """Morph between two shapes with equal point counts"""
        # Resample both to same length
        target_len = max(len(shape1), len(shape2))
        s1 = self._resample_shape(shape1, target_len)
        s2 = self._resample_shape(shape2, target_len)

        result = []
        for (x1, y1), (x2, y2) in zip(s1, s2):
            x = x1 + (x2 - x1) * blend
            y = y1 + (y2 - y1) * blend
            result.append((x, y))
        return result

    def _resample_shape(self, shape: List, target_len: int) -> List[Tuple[float, float]]:
        """Resample shape to target number of points"""
        if len(shape) >= target_len:
            step = len(shape) / target_len
            return [shape[int(i * step)] for i in range(target_len)]
        else:
            # Interpolate
            result = []
            for i in range(target_len):
                t = i / max(target_len - 1, 1) * (len(shape) - 1)
                idx = int(t)
                if idx >= len(shape) - 1:
                    result.append(shape[-1])
                else:
                    blend = t - idx
                    x = shape[idx][0] + (shape[idx+1][0] - shape[idx][0]) * blend
                    y = shape[idx][1] + (shape[idx+1][1] - shape[idx][1]) * blend
                    result.append((x, y))
            return result

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(base, "RGB")

        # Progress through animals
        num_animals = len(self.animal_shapes) - 1
        animal_progress = progress * num_animals
        current_idx = int(animal_progress)
        blend = animal_progress - current_idx

        if current_idx >= num_animals:
            current_idx = num_animals - 1
            blend = 1.0

        current_shape = self._morph_shapes(
            self.animal_shapes[current_idx],
            self.animal_shapes[current_idx + 1],
            blend
        )

        # Convert to screen coordinates with audio reactivity
        points = []
        for x_norm, y_norm in current_shape:
            x = WIDTH * x_norm + math.sin(frame_idx / FPS * 2 + x_norm) * amplitude * 15
            y = HEIGHT * y_norm + math.cos(frame_idx / FPS * 2 + y_norm) * amplitude * 12
            points.append((x, y))

        # Draw continuous line
        color = (255, 215, 0)  # Gold
        thickness = int(3 + amplitude * 4)
        draw.line(points, fill=color, width=thickness, joint="curve")

        # Add glow
        base = add_glow_layer(base, radius=20, intensity=0.4 + amplitude * 0.2)
        base = add_grain(base, strength=0.03)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 5: Der sad to katte på et bord (Tech House) - Geometric cats ===
class GeometricCatsVisual:
    """Two geometric cat silhouettes dancing symmetrically"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def _draw_geometric_cat(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, morph: float, color: Tuple[int, int, int]):
        """Draw geometric cat made of triangles and polygons"""
        # Body (hexagon morphing)
        body_pts = []
        for i in range(6):
            angle = i * math.pi / 3 + morph * 0.5
            r = size * (0.8 + morph * 0.2)
            body_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(body_pts, fill=color, outline=(255, 255, 100), width=3)

        # Head (triangle)
        head_y = cy - size * 1.2
        head_pts = [
            (cx, head_y - size * 0.5),
            (cx - size * 0.4, head_y + size * 0.3),
            (cx + size * 0.4, head_y + size * 0.3),
        ]
        draw.polygon(head_pts, fill=color, outline=(255, 255, 100), width=3)

        # Ears (small triangles)
        for side in [-1, 1]:
            ear_x = cx + side * size * 0.35
            ear_pts = [
                (ear_x, head_y - size * 0.5),
                (ear_x - side * size * 0.2, head_y - size * 0.8),
                (ear_x + side * size * 0.15, head_y - size * 0.6),
            ]
            draw.polygon(ear_pts, fill=color)

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 5))
        draw = ImageDraw.Draw(base, "RGBA")

        # Two cats facing each other
        size = 80 + amplitude * 30
        morph = math.sin(t * 3) * (0.3 + amplitude * 0.5)

        color1 = (0, 255, 150)
        color2 = (255, 255, 0)

        # Left cat
        if progress < 0.75:
            self._draw_geometric_cat(draw, WIDTH * 0.3, HEIGHT * 0.5, size, morph, color1)

        # Right cat
        if progress < 0.75:
            self._draw_geometric_cat(draw, WIDTH * 0.7, HEIGHT * 0.5, size, -morph, color2)

        # Merge phase
        if progress > 0.5:
            merge_progress = (progress - 0.5) / 0.5
            merged_x = WIDTH * 0.5
            merged_y = HEIGHT * 0.5
            merged_size = size * (1.5 + merge_progress * 0.5)
            blend_color = (
                int(color1[0] * (1 - merge_progress) + color2[0] * merge_progress),
                int(color1[1] * (1 - merge_progress) + color2[1] * merge_progress),
                int(color1[2] * (1 - merge_progress) + color2[2] * merge_progress),
            )
            self._draw_geometric_cat(draw, merged_x, merged_y, merged_size, morph * 2, blend_color)

        base = add_glow_layer(base, radius=15, intensity=0.35)
        base = add_grain(base, strength=0.025)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 6: Mariehønen Evigglad (French House) - Kaleidoscopic ladybug ===
class KaleidoscopeLadybugVisual:
    """Radial symmetry ladybug pattern blooming"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        # Create base segment
        segment = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(segment, "RGBA")

        # Ladybug colors
        red = (255, 20, 20)
        black = (0, 0, 0)
        gold = (255, 215, 0)

        cx, cy = WIDTH * 0.5, HEIGHT * 0.5
        scale = 0.5 + progress * 0.8 + amplitude * 0.3

        # Draw ladybug segment
        radius = HEIGHT * 0.15 * scale

        # Red shell
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=red + (200,))

        # Black spots
        for i in range(3):
            spot_angle = i * 0.6 + t * 0.5
            spot_x = cx + math.cos(spot_angle) * radius * 0.5
            spot_y = cy + math.sin(spot_angle) * radius * 0.5
            spot_r = radius * 0.2
            draw.ellipse([spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r], fill=black + (255,))

        # Apply kaleidoscope effect (6-fold symmetry)
        base = Image.new("RGB", (WIDTH, HEIGHT), (5, 0, 10))
        symmetry = 6 if progress > 0.3 else int(1 + progress / 0.3 * 5)

        for i in range(symmetry):
            angle = (2 * math.pi / symmetry) * i
            rotated = segment.rotate(math.degrees(angle), center=(cx, cy), resample=Image.BICUBIC)
            base = Image.alpha_composite(base.convert("RGBA"), rotated).convert("RGB")

        base = add_glow_layer(base, radius=18, intensity=0.4)
        base = add_grain(base, strength=0.02)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 7: Lille sky gik morgentur (Future Garage) - Cloud over landscape ===
class CloudLandscapeVisual:
    """Soft cloud floating over geometric terrain"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (135, 180, 200))  # Morning sky
        draw = ImageDraw.Draw(base, "RGBA")

        # Geometric terrain (parallax scrolling)
        terrain_y = HEIGHT * 0.7
        num_mountains = 8
        for layer_idx in range(3):
            depth = layer_idx + 1
            offset = -progress * WIDTH * (0.3 + layer_idx * 0.2)
            brightness = 100 - layer_idx * 30

            for i in range(num_mountains + 2):
                x_base = (i - 1) * (WIDTH / num_mountains) + offset
                while x_base < -WIDTH * 0.3:
                    x_base += WIDTH * 1.5

                peak_height = HEIGHT * (0.15 + (i + layer_idx) % 3 * 0.08)
                mountain = [
                    (x_base, terrain_y),
                    (x_base + WIDTH / num_mountains / 2, terrain_y - peak_height),
                    (x_base + WIDTH / num_mountains, terrain_y),
                ]
                color = (brightness, brightness, brightness)
                draw.polygon(mountain, fill=color)

        # Soft cloud
        cloud_x = WIDTH * (0.2 + progress * 0.6)
        cloud_y = HEIGHT * (0.3 + math.sin(t) * 0.05)

        cloud_base = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        cloud_draw = ImageDraw.Draw(cloud_base, "RGBA")

        # Multiple overlapping circles for cloud
        cloud_puffs = 7
        for i in range(cloud_puffs):
            puff_x = cloud_x + (i - cloud_puffs / 2) * 35
            puff_y = cloud_y + math.sin(i + t * 0.8) * 20
            puff_r = 40 + amplitude * 20 + (cloud_puffs / 2 - abs(i - cloud_puffs / 2)) * 15
            cloud_draw.ellipse(
                [puff_x - puff_r, puff_y - puff_r, puff_x + puff_r, puff_y + puff_r],
                fill=(255, 255, 255, 180)
            )

        cloud_base = cloud_base.filter(ImageFilter.GaussianBlur(radius=15))
        base = Image.alpha_composite(base.convert("RGBA"), cloud_base).convert("RGB")

        # Light trails if late in song
        if progress > 0.65:
            trail_alpha = int((progress - 0.65) / 0.35 * 150)
            for i in range(5):
                trail_x = cloud_x - i * 60
                trail_y = cloud_y + i * 5
                trail_r = 30 - i * 4
                if trail_r > 0:
                    draw.ellipse(
                        [trail_x - trail_r, trail_y - trail_r, trail_x + trail_r, trail_y + trail_r],
                        fill=(255, 255, 200, trail_alpha // (i + 1))
                    )

        base = add_grain(base, strength=0.02)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 8: Hønsefødder og gulerødder (Glitch Hop) - Glitching shapes ===
class GlitchShapesVisual:
    """Chicken footprints glitching into carrots"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def _draw_footprint(self, draw: ImageDraw.ImageDraw, x: float, y: float, size: float, color: Tuple[int, int, int]):
        # Three toes
        for toe in [-1, 0, 1]:
            toe_x = x + toe * size * 0.3
            toe_y = y
            draw.ellipse([toe_x - size * 0.15, toe_y - size * 0.4, toe_x + size * 0.15, toe_y], fill=color)
        # Back toe
        draw.ellipse([x - size * 0.1, y + size * 0.1, x + size * 0.1, y + size * 0.3], fill=color)

    def _draw_carrot(self, draw: ImageDraw.ImageDraw, x: float, y: float, size: float, color: Tuple[int, int, int]):
        # Triangle carrot
        pts = [(x, y + size), (x - size * 0.4, y - size * 0.3), (x + size * 0.4, y - size * 0.3)]
        draw.polygon(pts, fill=color)
        # Green top
        for i in range(3):
            leaf_x = x + (i - 1) * size * 0.2
            draw.line([(x, y - size * 0.3), (leaf_x, y - size * 0.7)], fill=(0, 200, 0), width=3)

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (20, 15, 0))
        draw = ImageDraw.Draw(base, "RGBA")

        # Glitch threshold
        glitch_mode = int(t * 4) % 2 == 0 or progress > 0.6

        grid_x = 6
        grid_y = 4
        for gy in range(grid_y):
            for gx in range(grid_x):
                x = (gx + 0.5) / grid_x * WIDTH
                y = (gy + 0.5) / grid_y * HEIGHT

                # Add glitch offset
                if random.random() < amplitude * 0.5:
                    x += random.randint(-50, 50)
                    y += random.randint(-30, 30)

                size = 30 + amplitude * 20

                if glitch_mode or (gx + gy) % 2 == 0:
                    self._draw_carrot(draw, x, y, size, (255, 150, 0))
                else:
                    self._draw_footprint(draw, x, y, size, (255, 220, 0))

        # Add digital glitch effects
        if amplitude > 0.5:
            arr = np.array(base)
            shift = int(amplitude * 20)
            arr[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)
            base = Image.fromarray(arr)

        base = add_glow_layer(base, radius=12, intensity=0.3)
        base = add_grain(base, strength=0.04)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 09: Jeg er en papegøje (Electroclash) - Digital parrot ===
class DigitalParrotVisual:
    """Neon geometric parrot with glitch effects and progressive transformation"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        self.colors = [
            (0, 255, 255),      # cyan
            (255, 0, 255),      # magenta
            (255, 255, 0),      # yellow
            (0, 200, 255),      # electric blue
            (255, 20, 147),     # hot pink
            (0, 255, 100),      # lime
            (255, 150, 0),      # orange
        ]

    def _create_parrot_shape(self, center_x: float, center_y: float, size: float,
                            rotation: float, progress: float) -> Dict:
        """Generate geometric parrot shape components"""
        # Body triangle
        body = [
            (center_x - size * 0.3, center_y - size * 0.4),
            (center_x - size * 0.3, center_y + size * 0.4),
            (center_x + size * 0.5, center_y),
        ]

        # Head circle
        head_x = center_x + size * 0.5
        head_y = center_y - size * 0.2
        head_r = size * 0.25

        # Beak triangle
        beak = [
            (head_x + head_r, head_y - size * 0.05),
            (head_x + head_r, head_y + size * 0.05),
            (head_x + head_r + size * 0.2, head_y),
        ]

        # Tail fan
        tail_lines = []
        for i in range(5):
            angle = rotation + (i - 2) * 0.3
            tail_lines.append([
                (center_x - size * 0.3, center_y),
                (center_x - size * 0.3 - size * 0.4 * math.cos(angle),
                 center_y + size * 0.4 * math.sin(angle))
            ])

        return {'body': body, 'head': (head_x, head_y, head_r), 'beak': beak, 'tail': tail_lines}

    def _add_glitch_effect(self, img: Image.Image, intensity: float) -> Image.Image:
        """Add RGB shift and scan lines for glitch aesthetic"""
        if intensity < 0.3:
            return img

        arr = np.array(img)
        height, width = arr.shape[:2]

        # RGB channel shift
        shift = int(10 * intensity)
        shifted = arr.copy()
        shifted[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
        shifted[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)

        # Random horizontal scan lines
        if intensity > 0.5:
            for _ in range(int(10 * intensity)):
                y = random.randint(0, height - 1)
                shifted[y, :] = np.roll(shifted[y, :], random.randint(-20, 20))

        return Image.fromarray(shifted)

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        # Beat sync at 128 BPM
        beat_time = frame_idx / FPS
        beat = abs(math.sin(beat_time * (128 / 60.0) * 2 * math.pi))

        img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Progressive energy build
        energy = 0.5 + 0.5 * progress

        # Multiple parrots with increasing complexity
        num_parrots = 1 + int(progress * 4)  # 1 to 5 parrots over time

        for i in range(num_parrots):
            # Position and size variation
            angle = (i / max(num_parrots, 1)) * 2 * math.pi + progress * math.pi
            radius = 150 + 100 * progress
            cx = WIDTH // 2 + radius * math.cos(angle)
            cy = HEIGHT // 2 + radius * math.sin(angle)

            size = 80 + 60 * amplitude * energy
            rotation = progress * 2 * math.pi + beat * 0.3

            parrot = self._create_parrot_shape(cx, cy, size, rotation, progress)

            # Color cycling
            color_idx = (i + int(progress * 10)) % len(self.colors)
            color = self.colors[color_idx]

            # Draw body
            draw.polygon(parrot['body'], fill=color, outline=color)

            # Draw head
            hx, hy, hr = parrot['head']
            draw.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=color, outline=color)

            # Draw beak
            draw.polygon(parrot['beak'], fill=(255, 255, 0), outline=(255, 255, 0))

            # Draw tail lines
            tail_color = tuple(int(c * 0.7) for c in color)
            for line in parrot['tail']:
                draw.line(line, fill=tail_color, width=int(3 + 3 * amplitude))

        # Central glow pulse
        glow_size = int(50 + 100 * beat * energy)
        glow_color = self.colors[int(progress * len(self.colors)) % len(self.colors)]
        draw.ellipse(
            (WIDTH // 2 - glow_size, HEIGHT // 2 - glow_size,
             WIDTH // 2 + glow_size, HEIGHT // 2 + glow_size),
            fill=None,
            outline=glow_color,
            width=int(2 + 4 * amplitude)
        )

        # Add glow
        img = add_glow_layer(img, 8 + 12 * amplitude, 0.3 + 0.2 * energy)

        # Glitch effect (increases with energy)
        glitch_intensity = energy * amplitude
        if glitch_intensity > 0.3:
            img = self._add_glitch_effect(img, glitch_intensity)

        # Grain texture
        img = add_grain(img, 0.03)

        # Fade in/out
        img = apply_global_fade(img, frame_idx, total)
        return img


# === TRACK 10: Op lille Hans (Big Beat) - Explosive sun/alarm ===
class ExplosiveSunVisual:
    """Aggressive alarm clock with explosive sun rays"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (40, 0, 0))
        draw = ImageDraw.Draw(base, "RGBA")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.5

        # Stage 1: Alarm bell (0-30%)
        if progress < 0.3:
            bell_size = 60 + amplitude * 40 + progress * 50
            draw.arc([cx - bell_size, cy - bell_size, cx + bell_size, cy + bell_size],
                    start=200, end=340, fill=(255, 100, 0), width=8)
            # Clapper
            clapper_swing = math.sin(t * 8) * 20
            draw.line([(cx, cy), (cx + clapper_swing, cy + bell_size * 0.7)],
                     fill=(255, 255, 0), width=6)

        # Stage 2: Sun rises (30-70%)
        elif progress < 0.7:
            sun_progress = (progress - 0.3) / 0.4
            sun_y = HEIGHT * (1.2 - sun_progress * 0.7)
            sun_r = 80 + sun_progress * 100 + amplitude * 50

            # Sun core
            draw.ellipse([cx - sun_r, sun_y - sun_r, cx + sun_r, sun_y + sun_r],
                        fill=(255, 150, 0))

            # Explosive rays
            num_rays = 12
            for i in range(num_rays):
                angle = (i / num_rays) * 2 * math.pi + t * 2
                burst = amplitude * 150 + 100
                x1 = cx + math.cos(angle) * sun_r
                y1 = sun_y + math.sin(angle) * sun_r
                x2 = cx + math.cos(angle) * (sun_r + burst)
                y2 = sun_y + math.sin(angle) * (sun_r + burst)
                width = int(6 + amplitude * 10)
                draw.line([(x1, y1), (x2, y2)], fill=(255, 200, 0), width=width)

        # Stage 3: Maximum strobing (70-100%)
        else:
            strobe = int(t * 10) % 2
            sun_r = 120 + amplitude * 80
            color = (255, 255, 0) if strobe else (255, 100, 0)

            draw.ellipse([cx - sun_r, cy - sun_r, cx + sun_r, cy + sun_r], fill=color)

            # Aggressive strobing rays
            num_rays = 16
            for i in range(num_rays):
                angle = (i / num_rays) * 2 * math.pi + t * 4
                burst = amplitude * 200 + 150
                x2 = cx + math.cos(angle) * burst
                y2 = cy + math.sin(angle) * burst
                draw.line([(cx, cy), (x2, y2)], fill=color, width=int(8 + amplitude * 12))

        base = add_glow_layer(base, radius=25, intensity=0.5)
        base = add_grain(base, strength=0.03)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 11: Oles nye autobil (EBM/Industrial) - Blueprint car ===
class IndustrialBlueprintVisual:
    """Technical blueprint car with mechanical pistons"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 20))
        draw = ImageDraw.Draw(base, "RGB")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.5
        blueprint_color = (100, 200, 255)
        accent_color = (255, 50, 50)

        # Stage 1: Assembly (0-40%)
        assembly_prog = min(progress / 0.4, 1.0)

        # Car body (rectangle)
        if assembly_prog > 0.2:
            car_w = 300 * min((assembly_prog - 0.2) / 0.2, 1.0)
            car_h = 100
            draw.rectangle([cx - car_w/2, cy - car_h/2, cx + car_w/2, cy + car_h/2],
                         outline=blueprint_color, width=3)

        # Wheels
        if assembly_prog > 0.5:
            wheel_prog = min((assembly_prog - 0.5) / 0.3, 1.0)
            wheel_r = 40 * wheel_prog
            for wx in [cx - 100, cx + 100]:
                draw.ellipse([wx - wheel_r, cy + 30, wx + wheel_r, cy + 30 + wheel_r * 2],
                           outline=blueprint_color, width=3)

        # Stage 2: Pistons firing (40-90%)
        if progress > 0.4:
            piston_beat = math.sin(t * 6) * 0.5 + 0.5  # TØF TØF TØF rhythm
            piston_y = cy - 80 - piston_beat * 30 * amplitude

            # Draw 4 pistons
            for i in range(4):
                px = cx - 120 + i * 80
                draw.line([(px, cy - 80), (px, piston_y)], fill=accent_color, width=4)
                draw.ellipse([px - 10, piston_y - 15, px + 10, piston_y + 5], fill=accent_color)

        # Stage 3: Motion trails (90-100%)
        if progress > 0.9:
            trail_prog = (progress - 0.9) / 0.1
            for i in range(5):
                trail_x = cx + i * 40 - trail_prog * 200
                trail_alpha = int(255 * (1 - i / 5))
                # Create motion blur effect with overlapping rectangles
                draw.rectangle([trail_x - 150, cy - 50, trail_x, cy + 50],
                             fill=(50, 100, 150))

        base = add_glow_layer(base, radius=10, intensity=0.3)
        base = add_grain(base, strength=0.035)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 12: Jeg gik mig over sø og land (Progressive House) - Topographic map ===
class TopographicMapVisual:
    """Aerial topographic map with glowing path"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        random.seed(999)
        self.terrain = self._generate_terrain()

    def _generate_terrain(self):
        """Generate simple height map"""
        terrain = np.zeros((40, 60), dtype=np.float32)
        for _ in range(15):
            cx = random.randint(5, 55)
            cy = random.randint(5, 35)
            height = random.uniform(0.3, 1.0)
            radius = random.randint(4, 12)

            for y in range(max(0, cy - radius), min(40, cy + radius)):
                for x in range(max(0, cx - radius), min(60, cx + radius)):
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        terrain[y, x] += height * (1 - dist / radius)

        terrain = np.clip(terrain, 0, 1)
        return terrain

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        # Render terrain
        terrain_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        cell_h = HEIGHT // 40
        cell_w = WIDTH // 60

        for y in range(40):
            for x in range(60):
                height = self.terrain[y, x]
                # Water = blue, Land = green, Mountains = brown
                if height < 0.3:
                    color = (0, 50, 150)  # Water
                elif height < 0.6:
                    color = (0, int(100 + height * 100), 0)  # Land
                else:
                    color = (int(100 + height * 50), int(80 + height * 50), 0)  # Mountains

                terrain_img[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w] = color

        base = Image.fromarray(terrain_img)
        draw = ImageDraw.Draw(base, "RGBA")

        # Draw glowing path
        path_points = []
        num_points = int(20 + progress * 40)
        for i in range(num_points):
            t_param = i / max(num_points - 1, 1)
            x = WIDTH * (0.1 + t_param * 0.8) + math.sin(t_param * 8) * WIDTH * 0.15
            y = HEIGHT * (0.2 + t_param * 0.6) + math.cos(t_param * 6) * HEIGHT * 0.15
            path_points.append((x, y))

        if len(path_points) > 1:
            draw.line(path_points, fill=(255, 255, 100, 200), width=int(4 + amplitude * 6), joint="curve")

        # Progressive layers revealed
        if progress > 0.5:
            # Add contour lines
            for level in [0.2, 0.4, 0.6, 0.8]:
                for y in range(1, 40):
                    for x in range(1, 60):
                        if self.terrain[y-1, x] < level <= self.terrain[y, x]:
                            draw.point((x * cell_w, y * cell_h), fill=(255, 255, 255, 100))

        base = add_glow_layer(base, radius=14, intensity=0.3)
        base = add_grain(base, strength=0.025)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 13: I østen stiger solen op (Acid Techno) - Psychedelic acid sun ===
class AcidSunVisual:
    """Melting psychedelic sun with liquid distortions"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (20, 0, 40))
        draw = ImageDraw.Draw(base, "RGB")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.4
        sun_r = 120 + amplitude * 60

        # Create sun with gradient
        for r in range(int(sun_r), 0, -5):
            color_mix = r / sun_r
            color = (
                int(255 * color_mix + 255 * (1 - color_mix)),
                int(150 * color_mix + 100 * (1 - color_mix)),
                int(0 * color_mix + 255 * (1 - color_mix))
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # Convert to array for warping
        arr = np.array(base).astype(np.float32)

        # Apply acid modulation
        if progress > 0.3:
            acid_intensity = (progress - 0.3) / 0.7 * amplitude

            # Create distortion field
            y_coords, x_coords = np.mgrid[0:HEIGHT, 0:WIDTH]

            # Multiple sine waves
            distort_x = np.sin(y_coords / 30 + t * 2) * acid_intensity * 20
            distort_x += np.sin(x_coords / 40 + t * 3) * acid_intensity * 15
            distort_y = np.cos(x_coords / 35 + t * 2.5) * acid_intensity * 18
            distort_y += np.cos(y_coords / 25 + t * 1.8) * acid_intensity * 12

            # Apply displacement
            new_x = np.clip(x_coords + distort_x, 0, WIDTH - 1).astype(np.int32)
            new_y = np.clip(y_coords + distort_y, 0, HEIGHT - 1).astype(np.int32)

            arr = arr[new_y, new_x]

        # Chromatic aberration
        if progress > 0.6:
            shift = int(acid_intensity * 15)
            arr[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)

        base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        base = add_glow_layer(base, radius=20, intensity=0.4 + amplitude * 0.2)
        base = add_grain(base, strength=0.03)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 14: I skovens dybe stille ro (Organic) - Tree growth ===
class TreeGrowthVisual:
    """Slow organic tree growth with breathing quality"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def _draw_branch(self, draw: ImageDraw.ImageDraw, x: float, y: float, length: float, angle: float, depth: int, max_depth: int, progress: float):
        """Recursive branch drawing"""
        if depth > max_depth or length < 3:
            return

        # Branch only grows if we've progressed enough
        branch_progress = max(0, min(1, (progress - depth * 0.15) / 0.15))
        if branch_progress <= 0:
            return

        actual_length = length * branch_progress

        x2 = x + math.cos(angle) * actual_length
        y2 = y - math.sin(angle) * actual_length

        width = max(1, int((max_depth - depth + 1) * 1.5))
        color = (40 + depth * 20, 60 + depth * 15, 20)
        draw.line([(x, y), (x2, y2)], fill=color, width=width)

        # Leaves at tips (late stage)
        if depth == max_depth and progress > 0.6:
            leaf_r = 8
            leaf_color = (20, 100 + int(progress * 50), 20, 150)
            draw.ellipse([x2 - leaf_r, y2 - leaf_r, x2 + leaf_r, y2 + leaf_r], fill=leaf_color)

        # Recurse
        if depth < max_depth:
            new_length = length * 0.7
            self._draw_branch(draw, x2, y2, new_length, angle - 0.4, depth + 1, max_depth, progress)
            self._draw_branch(draw, x2, y2, new_length, angle + 0.3, depth + 1, max_depth, progress)

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        # Dark forest background
        base = Image.new("RGBA", (WIDTH, HEIGHT), (5, 15, 10, 255))
        draw = ImageDraw.Draw(base, "RGBA")

        # Breathing sway
        sway = math.sin(t * 0.5) * 5 * progress

        # Draw tree
        trunk_x = WIDTH * 0.5 + sway
        trunk_y = HEIGHT * 0.85
        trunk_length = HEIGHT * 0.3

        max_depth = 6
        self._draw_branch(draw, trunk_x, trunk_y, trunk_length, math.pi / 2, 0, max_depth, progress)

        base = base.convert("RGB")
        base = add_glow_layer(base, radius=8, intensity=0.2)
        base = add_grain(base, strength=0.03)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 15: Hist, hvor vejen slår en bugt (Downtempo) - Curved road ===
class CurvedRoadVisual:
    """Zen curved path through minimal landscape"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        # Soft watercolor background
        base = Image.new("RGB", (WIDTH, HEIGHT), (220, 210, 190))

        # Sky gradient
        for y in range(HEIGHT // 2):
            color = (
                200 + int(y / HEIGHT * 55),
                210 + int(y / HEIGHT * 45),
                230
            )
            draw_line = ImageDraw.Draw(base)
            draw_line.line([(0, y), (WIDTH, y)], fill=color)

        draw = ImageDraw.Draw(base, "RGBA")

        # Curved road
        path_points = []
        num_points = 60
        for i in range(num_points):
            t_param = i / (num_points - 1)
            # Bezier-like curve
            x = WIDTH * (0.2 + t_param * 0.6 + math.sin(t_param * math.pi) * 0.15)
            y = HEIGHT * (0.8 - t_param * 0.5)
            path_points.append((x, y))

        # Draw road with fading based on progress
        visible_points = int(len(path_points) * min(progress / 0.7, 1.0))
        if visible_points > 1:
            visible_path = path_points[:visible_points]
            draw.line(visible_path, fill=(180, 170, 150, 200), width=40, joint="curve")
            draw.line(visible_path, fill=(200, 190, 170, 255), width=4, joint="curve")

        # Minimal distant trees (appear late)
        if progress > 0.3:
            tree_progress = (progress - 0.3) / 0.7
            num_trees = int(tree_progress * 8)
            for i in range(num_trees):
                tree_x = WIDTH * (0.15 + i * 0.1)
                tree_y = HEIGHT * 0.5
                tree_h = 30
                draw.line([(tree_x, tree_y), (tree_x, tree_y - tree_h)], fill=(100, 120, 80), width=3)
                draw.ellipse([tree_x - 10, tree_y - tree_h - 15, tree_x + 10, tree_y - tree_h],
                           fill=(120, 140, 100, 180))

        base = add_grain(base, strength=0.02)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 16: Fra Engeland til Skotland (Trip Hop) - Network nodes ===
class NetworkNodesVisual:
    """Network graph spreading northward"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope
        random.seed(420)
        self.nodes = self._generate_nodes()

    def _generate_nodes(self):
        """Generate network node positions"""
        nodes = []
        # South to north distribution
        for i in range(40):
            x = random.uniform(0.2, 0.8)
            # Bias toward north as index increases
            y = random.uniform(0.3 + i * 0.015, 0.9)
            nodes.append({"x": x, "y": y, "connections": []})

        # Create connections
        for i, node in enumerate(nodes):
            # Connect to nearby nodes
            for j, other in enumerate(nodes):
                if i != j:
                    dist = math.sqrt((node["x"] - other["x"])**2 + (node["y"] - other["y"])**2)
                    if dist < 0.2 and len(node["connections"]) < 4:
                        node["connections"].append(j)

        return nodes

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (5, 10, 25))
        draw = ImageDraw.Draw(base, "RGBA")

        # Nodes appear progressively from south to north
        visible_nodes = int(len(self.nodes) * min(progress / 0.6, 1.0))

        # Draw connections first
        for i in range(visible_nodes):
            node = self.nodes[i]
            x1 = node["x"] * WIDTH
            y1 = node["y"] * HEIGHT

            for conn_idx in node["connections"]:
                if conn_idx < visible_nodes:
                    other = self.nodes[conn_idx]
                    x2 = other["x"] * WIDTH
                    y2 = other["y"] * HEIGHT

                    # Pulsing data flow
                    pulse = (math.sin(t * 2 + i * 0.5) + 1) / 2
                    alpha = int(150 + pulse * 100)
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 150, 200, alpha), width=2)

        # Draw nodes
        for i in range(visible_nodes):
            node = self.nodes[i]
            x = node["x"] * WIDTH
            y = node["y"] * HEIGHT

            # Node size based on connections
            node_r = 8 + len(node["connections"]) * 2 + amplitude * 5

            # Glow
            draw.ellipse([x - node_r * 2, y - node_r * 2, x + node_r * 2, y + node_r * 2],
                        fill=(0, 100, 200, 50))
            # Core
            draw.ellipse([x - node_r, y - node_r, x + node_r, y + node_r],
                        fill=(100, 200, 255, 255))

        base = add_glow_layer(base, radius=15, intensity=0.35)
        base = add_grain(base, strength=0.03)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 17: Tre små fisk (Drumstep) - Synchronized fish ===
class SynchronizedFishVisual:
    """Three geometric fish in synchronized choreography"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def _draw_fish(self, draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, color: Tuple[int, int, int]):
        """Draw geometric fish"""
        # Body (triangle)
        body = [
            (x + math.cos(angle) * size, y + math.sin(angle) * size),
            (x + math.cos(angle + 2.5) * size * 0.6, y + math.sin(angle + 2.5) * size * 0.6),
            (x + math.cos(angle - 2.5) * size * 0.6, y + math.sin(angle - 2.5) * size * 0.6),
        ]
        draw.polygon(body, fill=color, outline=(255, 255, 255), width=2)

        # Tail
        tail_base_x = x - math.cos(angle) * size * 0.3
        tail_base_y = y - math.sin(angle) * size * 0.3
        tail = [
            (tail_base_x, tail_base_y),
            (tail_base_x + math.cos(angle + math.pi - 0.5) * size * 0.5,
             tail_base_y + math.sin(angle + math.pi - 0.5) * size * 0.5),
            (tail_base_x + math.cos(angle + math.pi + 0.5) * size * 0.5,
             tail_base_y + math.sin(angle + math.pi + 0.5) * size * 0.5),
        ]
        draw.polygon(tail, fill=color)

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (0, 20, 40))
        draw = ImageDraw.Draw(base, "RGBA")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.5
        fish_size = 60 + amplitude * 30

        # Three fish in formation
        fish_colors = [(0, 200, 255), (0, 255, 200), (255, 150, 0)]

        for fish_idx in range(3):
            # Different choreography patterns over time
            if progress < 0.3:
                # Simple line formation
                x = cx + (fish_idx - 1) * 150
                y = cy
                angle = 0
            elif progress < 0.6:
                # Circle formation
                formation_angle = (fish_idx / 3) * 2 * math.pi + t * 1.5
                radius = 180 + amplitude * 50
                x = cx + math.cos(formation_angle) * radius
                y = cy + math.sin(formation_angle) * radius
                angle = formation_angle + math.pi / 2
            else:
                # Figure-8 pattern
                t_param = t * 1.2 + fish_idx * 0.6
                x = cx + math.sin(t_param) * 250
                y = cy + math.sin(t_param * 2) * 150
                angle = math.atan2(math.cos(t_param * 2) * 2, math.cos(t_param))

            self._draw_fish(draw, x, y, fish_size, angle, fish_colors[fish_idx])

            # Trails multiply in final stage
            if progress > 0.6:
                trail_alpha = int((progress - 0.6) / 0.4 * 150)
                for i in range(1, 4):
                    trail_x = x - math.cos(angle) * i * 30
                    trail_y = y - math.sin(angle) * i * 30
                    trail_size = fish_size * (1 - i * 0.2)
                    trail_color = fish_colors[fish_idx] + (trail_alpha // i,)
                    draw.ellipse([trail_x - trail_size/2, trail_y - trail_size/2,
                                trail_x + trail_size/2, trail_y + trail_size/2],
                               fill=trail_color)

        base = add_glow_layer(base, radius=16, intensity=0.35)
        base = add_grain(base, strength=0.025)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 18: Tre små kinesere (Jungle) - Rowing rhythm ===
class RowingRhythmVisual:
    """Stroboscopic rowing strokes creating energy waves"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        # High contrast black/white
        base = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(base, "RGBA")

        # Rowing beat (84 BPM = 1.4 Hz)
        beat_phase = (t * 1.4) % 1.0
        is_stroke = beat_phase < 0.3

        if is_stroke:
            # Stroke moment - bright flash
            stroke_intensity = (0.3 - beat_phase) / 0.3

            # Three rowers
            for rower_idx in range(3):
                x = WIDTH * (0.25 + rower_idx * 0.25)
                y = HEIGHT * 0.5

                # Oar stroke
                oar_angle = -math.pi / 4 + beat_phase * math.pi / 2
                oar_length = 150
                oar_x = x + math.cos(oar_angle) * oar_length
                oar_y = y + math.sin(oar_angle) * oar_length

                draw.line([(x, y), (oar_x, oar_y)],
                         fill=(255, 255, 255, int(stroke_intensity * 255)),
                         width=int(6 + amplitude * 10))

                # Rower figure (simple)
                draw.ellipse([x - 20, y - 20, x + 20, y + 20],
                           fill=(255, 255, 255, int(stroke_intensity * 200)))

        # Energy waves emanate from strokes
        num_waves = int(t * 1.4)  # One per stroke
        for wave_idx in range(max(0, num_waves - 10), num_waves):
            wave_age = t - (wave_idx / 1.4)
            if wave_age > 0 and wave_age < 2:
                wave_radius = wave_age * 300
                wave_alpha = int((1 - wave_age / 2) * 100)

                for i in range(3):
                    cx = WIDTH * (0.25 + i * 0.25)
                    cy = HEIGHT * 0.5
                    draw.ellipse([cx - wave_radius, cy - wave_radius,
                                cx + wave_radius, cy + wave_radius],
                               outline=(255, 100, 100, wave_alpha), width=3)

        # Add accent colors
        if random.random() < amplitude * 0.3:
            accent_color = random.choice([(255, 0, 0), (0, 255, 255), (255, 255, 0)])
            x1, x2 = random.randint(0, WIDTH), random.randint(0, WIDTH)
            y1, y2 = random.randint(0, HEIGHT), random.randint(0, HEIGHT)
            draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                         fill=accent_color + (100,))

        base = add_grain(base, strength=0.04)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 19: Ti små cyklister (DnB) - Speed vortex ===
class SpeedVortexVisual:
    """Radial motion blur creating spiral vortex"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(base, "RGBA")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.5

        # Number of cyclists (build up to 10)
        num_cyclists = int(1 + progress * 9)

        # Spiral parameters
        spiral_speed = 2 + progress * 4 + amplitude * 3

        for cyclist_idx in range(num_cyclists):
            # Each cyclist on a spiral path
            phase = t * spiral_speed + cyclist_idx * 2 * math.pi / 10
            radius = 50 + (cyclist_idx + 1) * 35 + math.sin(phase * 2) * 20

            x = cx + math.cos(phase) * radius
            y = cy + math.sin(phase) * radius

            # Wheel trails (speed lines)
            num_trails = 12
            for trail_idx in range(num_trails):
                trail_phase = phase - trail_idx * 0.15
                trail_radius = 50 + (cyclist_idx + 1) * 35 + math.sin(trail_phase * 2) * 20
                trail_x = cx + math.cos(trail_phase) * trail_radius
                trail_y = cy + math.sin(trail_phase) * trail_radius

                trail_alpha = int(255 * (1 - trail_idx / num_trails) * (0.5 + amplitude * 0.5))

                # Color shifts through spectrum
                color_shift = (cyclist_idx + frame_idx / 10) % 3
                if color_shift < 1:
                    color = (255, int(255 * (1 - color_shift)), 255)
                elif color_shift < 2:
                    color = (int(255 * (2 - color_shift)), 255, 255)
                else:
                    color = (255, 255, int(255 * (3 - color_shift)))

                draw.ellipse([trail_x - 5, trail_y - 5, trail_x + 5, trail_y + 5],
                           fill=color + (trail_alpha,))

        # Vortex center grows
        if progress > 0.7:
            vortex_size = (progress - 0.7) / 0.3 * 100 + amplitude * 50
            draw.ellipse([cx - vortex_size, cy - vortex_size, cx + vortex_size, cy + vortex_size],
                        fill=(255, 255, 255, 50))

        base = add_glow_layer(base, radius=20, intensity=0.4)
        base = add_grain(base, strength=0.03)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === TRACK 20: Se dig for (Techstep/DnB) - Warning symbols FINALE ===
class WarningSymbolsVisual:
    """Aggressive warning symbols explosive finale"""
    def __init__(self, duration: float, envelope: Sequence[float]):
        self.duration = duration
        self.total_frames = int(math.ceil(duration * FPS))
        self.envelope = envelope

    def render_frame(self, frame_idx: int) -> Image.Image:
        total = self.total_frames
        progress = frame_idx / max(total - 1, 1)
        t = frame_idx / FPS
        amplitude = float(self.envelope[min(frame_idx, len(self.envelope) - 1)])

        base = Image.new("RGB", (WIDTH, HEIGHT), (20, 0, 0))
        draw = ImageDraw.Draw(base, "RGBA")

        cx, cy = WIDTH * 0.5, HEIGHT * 0.5

        # Warning colors
        warning_red = (255, 0, 0)
        warning_yellow = (255, 255, 0)
        white = (255, 255, 255)

        # Stage 1: Symbols appear from edges (0-40%)
        if progress < 0.4:
            num_symbols = int(progress / 0.4 * 20)
            for i in range(num_symbols):
                angle = i * 2 * math.pi / 20
                dist = WIDTH * 0.5 * (1 - progress / 0.4)
                x = cx + math.cos(angle) * dist
                y = cy + math.sin(angle) * dist

                # Draw warning triangle
                size = 30 + amplitude * 20
                triangle = [
                    (x, y - size),
                    (x - size * 0.866, y + size * 0.5),
                    (x + size * 0.866, y + size * 0.5),
                ]
                draw.polygon(triangle, fill=warning_yellow, outline=warning_red, width=3)
                draw.text((x - 5, y - 10), "!", fill=warning_red, font=None)

        # Stage 2: Aggressive pulsing (40-90%)
        elif progress < 0.9:
            # Strobe effect
            strobe = int(t * 8) % 2
            bg_color = warning_red if strobe else (100, 0, 0)
            base = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
            draw = ImageDraw.Draw(base, "RGBA")

            # Multiple symbols converging
            num_symbols = 30
            convergence = (progress - 0.4) / 0.5

            for i in range(num_symbols):
                angle = i * 2 * math.pi / num_symbols + t * 3
                dist = WIDTH * 0.6 * (1 - convergence) + amplitude * 50
                x = cx + math.cos(angle) * dist
                y = cy + math.sin(angle) * dist

                size = 40 + amplitude * 30

                # Eyes symbol
                if i % 3 == 0:
                    draw.ellipse([x - size, y - size/2, x - size/2, y + size/2], fill=white)
                    draw.ellipse([x + size/2, y - size/2, x + size, y + size/2], fill=white)
                    draw.ellipse([x - size*0.8, y - size/4, x - size*0.7, y + size/4], fill=(0, 0, 0))
                    draw.ellipse([x + size*0.7, y - size/4, x + size*0.8, y + size/4], fill=(0, 0, 0))

                # Arrows
                elif i % 3 == 1:
                    arrow = [
                        (x, y - size),
                        (x - size/2, y),
                        (x - size/4, y),
                        (x - size/4, y + size),
                        (x + size/4, y + size),
                        (x + size/4, y),
                        (x + size/2, y),
                    ]
                    draw.polygon(arrow, fill=warning_yellow)

                # Exclamation marks
                else:
                    draw.rectangle([x - size/4, y - size, x + size/4, y + size/2], fill=warning_red)
                    draw.ellipse([x - size/4, y + size*0.6, x + size/4, y + size], fill=warning_red)

        # Stage 3: FINAL CLIMAX - Explosion (90-100%)
        else:
            explosion_progress = (progress - 0.9) / 0.1

            # White flash
            if explosion_progress < 0.3:
                flash_intensity = (0.3 - explosion_progress) / 0.3
                base = Image.new("RGB", (WIDTH, HEIGHT), (int(255 * flash_intensity),) * 3)
            else:
                # Explosive shards
                base = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
                draw = ImageDraw.Draw(base, "RGBA")

                num_shards = 200
                for i in range(num_shards):
                    angle = i * 2 * math.pi / num_shards
                    speed = 300 + i % 100
                    dist = explosion_progress * speed

                    x = cx + math.cos(angle) * dist
                    y = cy + math.sin(angle) * dist

                    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                        shard_color = warning_red if i % 2 == 0 else warning_yellow
                        shard_size = 15 - explosion_progress * 10
                        if shard_size > 0:
                            draw.ellipse([x - shard_size, y - shard_size,
                                        x + shard_size, y + shard_size],
                                       fill=shard_color + (int((1 - explosion_progress) * 255),))

        base = add_glow_layer(base, radius=15, intensity=0.3)
        base = add_grain(base, strength=0.035)
        base = apply_global_fade(base, frame_idx, total)
        return base


# === Track configuration ===
@dataclass
class TrackConfig:
    key: str
    title: str
    start: float
    duration: float
    visual_class: type
    description: str


# Timing from PA-PAGOJE_TRANSITION_LOG.md (converted to seconds)
def time_to_seconds(time_str: str) -> float:
    """Convert MM:SS.SS to seconds"""
    parts = time_str.split(":")
    return float(parts[0]) * 60 + float(parts[1])


TRACKS: Dict[str, TrackConfig] = {
    "04": TrackConfig(
        key="04", title="Spørge Jørgen (Minimal techno _ tech house)",
        start=time_to_seconds("07:17.74"), duration=189.96,
        visual_class=LineArtAnimalsVisual,
        description="Single line art morphing between animals",
    ),
    "05": TrackConfig(
        key="05", title="Der sad to katte på et bord (Tech house)",
        start=time_to_seconds("10:22.07"), duration=62.48,
        visual_class=GeometricCatsVisual,
        description="Two geometric cats dancing and merging",
    ),
    "06": TrackConfig(
        key="06", title="Mariehønen Evigglad (French house)",
        start=time_to_seconds("11:22.50"), duration=156.56,
        visual_class=KaleidoscopeLadybugVisual,
        description="Kaleidoscopic ladybug pattern blooming",
    ),
    "07": TrackConfig(
        key="07", title="Lille sky gik morgentur (Future garage)",
        start=time_to_seconds("14:02.29"), duration=112.12,
        visual_class=CloudLandscapeVisual,
        description="Cloud floating over geometric terrain",
    ),
    "08": TrackConfig(
        key="08", title="Hønsefødder og gulerødder (Glitch hop)",
        start=time_to_seconds("15:19.27"), duration=60.04,
        visual_class=GlitchShapesVisual,
        description="Glitching between chicken feet and carrots",
    ),
    "09": TrackConfig(
        key="09", title="Jeg er en papegøje fra Amerika (Electroclash _ Electro house)",
        start=970.4772335600906, duration=136.04,
        visual_class=DigitalParrotVisual,
        description="Neon geometric parrot with glitch effects",
    ),
    "10": TrackConfig(
        key="10", title="Op lille Hans (Big beat _ breakbeat)",
        start=time_to_seconds("18:19.95"), duration=110.96,
        visual_class=ExplosiveSunVisual,
        description="Explosive alarm sun with aggressive rays",
    ),
    "11": TrackConfig(
        key="11", title="Oles nye autobil (EBM, Industrial Techno) (TØF TØF TØF MIX)",
        start=time_to_seconds("20:02.87"), duration=138.32,
        visual_class=IndustrialBlueprintVisual,
        description="Industrial blueprint car with firing pistons",
    ),
    "12": TrackConfig(
        key="12", title="Jeg gik mig over sø og land (Prog House)",
        start=time_to_seconds("22:11.93"), duration=231.96,
        visual_class=TopographicMapVisual,
        description="Topographic map with progressive path layers",
    ),
    "13": TrackConfig(
        key="13", title="I østen stiger solen op (acid techno, acid house)",
        start=time_to_seconds("26:10.35"), duration=218.68,
        visual_class=AcidSunVisual,
        description="Psychedelic acid-warped melting sun",
    ),
    "14": TrackConfig(
        key="14", title="I skovens dybe stille ro (Organic)",
        start=time_to_seconds("29:18.98"), duration=129.96,
        visual_class=TreeGrowthVisual,
        description="Slow organic tree growth with breathing",
    ),
    "15": TrackConfig(
        key="15", title="Hist, hvor vejen slår en bugt (Downtempo)",
        start=time_to_seconds("31:16.37"), duration=209.96,
        visual_class=CurvedRoadVisual,
        description="Zen curved road through minimal landscape",
    ),
    "16": TrackConfig(
        key="16", title="Fra Engeland til Skotland (Trip Hop)",
        start=time_to_seconds("34:32.60"), duration=172.96,
        visual_class=NetworkNodesVisual,
        description="Network nodes spreading northward",
    ),
    "17": TrackConfig(
        key="17", title="Tre små fisk (drumstep)",
        start=time_to_seconds("36:56.60"), duration=179.84,
        visual_class=SynchronizedFishVisual,
        description="Three fish in synchronized choreography",
    ),
    "18": TrackConfig(
        key="18", title="Tre små kinesere (Jungle)",
        start=time_to_seconds("39:45.69"), duration=56.04,
        visual_class=RowingRhythmVisual,
        description="Stroboscopic rowing creating energy waves",
    ),
    "19": TrackConfig(
        key="19", title="Ti små cyklister (DnB)",
        start=time_to_seconds("40:36.67"), duration=184.64,
        visual_class=SpeedVortexVisual,
        description="Radial speed vortex with cyclist trails",
    ),
    "20": TrackConfig(
        key="20", title="Se dig for (Techstep _ DnB)",
        start=time_to_seconds("43:35.85"), duration=137.96,
        visual_class=WarningSymbolsVisual,
        description="Warning symbols explosive finale",
    ),
}


def render_track(config: TrackConfig, mix_path: str, overwrite: bool = False) -> Dict:
    ensure_dirs(OUTPUT_ROOT, AUDIO_DIR, FRAMES_DIR, RENDERS_DIR, METADATA_DIR)

    slug = slugify(config.title)
    audio_path = os.path.join(AUDIO_DIR, f"{config.key}_{slug}.wav")
    frames_path = os.path.join(FRAMES_DIR, f"{config.key}_{slug}")
    output_video = os.path.join(RENDERS_DIR, f"{config.key}_{slug}.mp4")
    metadata_path = os.path.join(METADATA_DIR, f"{config.key}_{slug}.json")

    ensure_dirs(frames_path)

    # Clean old frames
    if os.path.exists(frames_path):
        for filename in os.listdir(frames_path):
            if filename.endswith(".png"):
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
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", os.path.join(frames_path, "frame_%05d.png"),
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_video,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("  ✓ Video render complete.")

    metadata = {
        "track_key": config.key,
        "title": config.title,
        "description": config.description,
        "duration_seconds": config.duration,
        "fps": FPS,
        "frames": total_frames,
        "video_path": output_video,
    }
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PA-PAGØJE remaining visuals.")
    parser.add_argument("--tracks", nargs="+", help="Track IDs to render")
    parser.add_argument("--mix", default="PA-PAGOJE_Festival_Mix2.wav", help="Path to master audio mix")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached audio segments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mix_path = args.mix

    if not os.path.exists(mix_path):
        raise FileNotFoundError(f"Mix audio not found: {mix_path}")

    track_keys = args.tracks if args.tracks else list(TRACKS.keys())

    selected = []
    for key in track_keys:
        if key not in TRACKS:
            print(f"Warning: Unknown track key '{key}', skipping")
            continue
        selected.append(TRACKS[key])

    manifest = []
    for config in selected:
        metadata = render_track(config, mix_path, overwrite=args.overwrite)
        manifest.append(metadata)

    manifest_path = os.path.join(METADATA_DIR, "remaining_tracks_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 100)
    print(f"✓ All {len(selected)} tracks rendered successfully")
    print(f"✓ Manifest: {manifest_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
