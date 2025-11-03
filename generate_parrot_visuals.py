#!/usr/bin/env python3
"""
Digital Parrot Visuals Generator
For "Jeg er en papegøje fra Amerika" (Electroclash, 128 BPM)
Theme: Neon geometric parrot, glitch aesthetic, progressive transformation
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import subprocess
import json
import math
import random

# Configuration
WIDTH, HEIGHT = 1920, 1080
FPS = 25
DURATION = 136  # 2:16 in seconds
TOTAL_FRAMES = int(DURATION * FPS)

# Color palette (neon/electroclash)
COLORS = {
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'yellow': (255, 255, 0),
    'electric_blue': (0, 200, 255),
    'hot_pink': (255, 20, 147),
    'lime': (0, 255, 100),
    'orange': (255, 150, 0),
}

def get_audio_amplitude(audio_file, frame_num, total_frames):
    """Extract amplitude for this frame (simplified - returns time-based modulation)"""
    # Simulate beat at 128 BPM = 2.13 beats per second
    t = frame_num / FPS
    beat_freq = 128 / 60.0

    # Primary beat pulse
    beat = abs(math.sin(t * beat_freq * 2 * math.pi))

    # Add some variation
    sub_beat = abs(math.sin(t * beat_freq * 8 * math.pi)) * 0.3

    # Overall energy curve (builds throughout)
    energy = 0.5 + 0.5 * (frame_num / total_frames)

    return beat * 0.7 + sub_beat * 0.3 + energy * 0.2

def create_parrot_shape(draw, center_x, center_y, size, rotation, progress):
    """Draw a geometric/digital parrot shape"""

    # Body (main triangle pointing right, like a bird body)
    body_points = [
        (center_x - size*0.3, center_y - size*0.4),
        (center_x - size*0.3, center_y + size*0.4),
        (center_x + size*0.5, center_y),
    ]

    # Head (circle)
    head_x = center_x + size * 0.5
    head_y = center_y - size * 0.2
    head_r = size * 0.25

    # Beak (small triangle)
    beak_points = [
        (head_x + head_r, head_y - size*0.05),
        (head_x + head_r, head_y + size*0.05),
        (head_x + head_r + size*0.2, head_y),
    ]

    # Tail (fan of lines)
    tail_lines = []
    for i in range(5):
        angle = rotation + (i - 2) * 0.3
        tail_lines.append([
            (center_x - size*0.3, center_y),
            (center_x - size*0.3 - size*0.4*math.cos(angle),
             center_y + size*0.4*math.sin(angle))
        ])

    return {
        'body': body_points,
        'head': (head_x, head_y, head_r),
        'beak': beak_points,
        'tail': tail_lines
    }

def add_glitch_effect(img, intensity):
    """Add digital glitch/scan line effects"""
    pixels = np.array(img)
    height, width = pixels.shape[:2]

    # RGB shift
    if intensity > 0.3:
        shift = int(10 * intensity)
        r_channel = np.roll(pixels[:,:,0], shift, axis=1)
        b_channel = np.roll(pixels[:,:,2], -shift, axis=1)
        pixels[:,:,0] = r_channel
        pixels[:,:,2] = b_channel

    # Scan lines
    for y in range(0, height, 4):
        if random.random() < intensity * 0.5:
            pixels[y:y+2, :] = pixels[y:y+2, :] * 0.7

    # Random horizontal glitch bars
    if random.random() < intensity * 0.3:
        glitch_y = random.randint(0, height - 50)
        glitch_h = random.randint(5, 30)
        shift_x = random.randint(-50, 50)
        glitch_section = pixels[glitch_y:glitch_y+glitch_h, :]
        pixels[glitch_y:glitch_y+glitch_h, :] = np.roll(glitch_section, shift_x, axis=1)

    return Image.fromarray(pixels)

def generate_frame(frame_num):
    """Generate a single frame"""

    # Progress through the song (0.0 to 1.0)
    progress = frame_num / TOTAL_FRAMES

    # Time in seconds
    t = frame_num / FPS

    # Get audio reactivity
    amplitude = get_audio_amplitude("song_papegoje.wav", frame_num, TOTAL_FRAMES)

    # Background color (shifts over time)
    bg_hue = int((progress * 60 + amplitude * 20) % 360)
    bg_color = tuple([int(c * 0.05) for c in COLORS['cyan']])  # Very dark

    # Create image
    img = Image.new('RGB', (WIDTH, HEIGHT), bg_color)
    draw = ImageDraw.Draw(img, 'RGBA')

    # === VISUAL STAGES ===

    # Stage 1 (0-25%): Particle emergence
    if progress < 0.25:
        stage_progress = progress / 0.25
        num_particles = int(100 + stage_progress * 200)

        for i in range(num_particles):
            # Particles slowly coalescing toward center-right (where parrot will be)
            target_x = WIDTH * 0.55
            target_y = HEIGHT * 0.5

            # Start scattered, move toward target
            scatter = (1 - stage_progress) * 400
            px = target_x + random.gauss(0, scatter)
            py = target_y + random.gauss(0, scatter)

            # Color cycles through palette
            color_idx = (i + int(t * 10)) % len(COLORS)
            color = list(COLORS.values())[color_idx]

            # Size reacts to audio
            size = 2 + amplitude * 8

            # Draw particle with glow
            draw.ellipse([px-size, py-size, px+size, py+size], fill=color)

    # Stage 2 (25-50%): Parrot formation
    elif progress < 0.5:
        stage_progress = (progress - 0.25) / 0.25

        # Parrot appears and solidifies
        center_x = WIDTH * 0.55
        center_y = HEIGHT * 0.5
        size = 150 * stage_progress + amplitude * 30

        rotation = math.sin(t * 2) * 0.3

        parrot = create_parrot_shape(draw, center_x, center_y, size, rotation, stage_progress)

        # Color shifts
        color1 = COLORS['cyan'] if int(t * 2) % 2 == 0 else COLORS['magenta']
        color2 = COLORS['hot_pink'] if int(t * 2) % 2 == 0 else COLORS['yellow']

        # Draw parrot with opacity based on stage_progress
        opacity = int(255 * stage_progress)

        # Body
        draw.polygon(parrot['body'], fill=color1 + (opacity,), outline=color2 + (255,), width=3)

        # Head
        hx, hy, hr = parrot['head']
        draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=color2 + (opacity,), outline=color1 + (255,), width=3)

        # Beak
        draw.polygon(parrot['beak'], fill=COLORS['orange'] + (opacity,))

        # Tail feathers
        for line in parrot['tail']:
            tail_color = list(COLORS.values())[random.randint(0, len(COLORS)-1)]
            draw.line(line, fill=tail_color + (opacity,), width=int(4 + amplitude * 4))

    # Stage 3 (50-75%): Glitch transformation
    elif progress < 0.75:
        stage_progress = (progress - 0.5) / 0.25

        # Fully formed parrot, now glitching and pulsing
        center_x = WIDTH * 0.5
        center_y = HEIGHT * 0.5
        size = 180 + amplitude * 50 + math.sin(t * 4) * 20

        rotation = math.sin(t * 3) * 0.5

        parrot = create_parrot_shape(draw, center_x, center_y, size, rotation, 1.0)

        # Rapid color cycling
        colors_list = list(COLORS.values())
        color_idx = int(t * 8) % len(colors_list)
        color1 = colors_list[color_idx]
        color2 = colors_list[(color_idx + 2) % len(colors_list)]

        # Draw with full opacity, heavy outlines
        draw.polygon(parrot['body'], fill=color1, outline=color2, width=5)

        hx, hy, hr = parrot['head']
        draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=color2, outline=color1, width=5)

        draw.polygon(parrot['beak'], fill=COLORS['orange'])

        for i, line in enumerate(parrot['tail']):
            tail_color = colors_list[(color_idx + i) % len(colors_list)]
            draw.line(line, fill=tail_color, width=int(6 + amplitude * 6))

        # Add geometric accents (triangles around the parrot)
        num_accents = 8
        for i in range(num_accents):
            angle = (t * 2 + i * 2 * math.pi / num_accents)
            dist = 250 + amplitude * 80
            ax = center_x + math.cos(angle) * dist
            ay = center_y + math.sin(angle) * dist
            tri_size = 20 + amplitude * 15

            tri_points = [
                (ax, ay - tri_size),
                (ax - tri_size*0.866, ay + tri_size*0.5),
                (ax + tri_size*0.866, ay + tri_size*0.5),
            ]
            accent_color = colors_list[(i + int(t*4)) % len(colors_list)]
            draw.polygon(tri_points, fill=accent_color + (200,))

    # Stage 4 (75-100%): Prismatic explosion
    else:
        stage_progress = (progress - 0.75) / 0.25

        # Parrot shatters into shards flying outward
        center_x = WIDTH * 0.5
        center_y = HEIGHT * 0.5

        num_shards = 150
        for i in range(num_shards):
            # Each shard flies outward
            angle = (i * 2 * math.pi / num_shards) + random.gauss(0, 0.2)
            speed = 200 + i * 2
            dist = stage_progress * speed + amplitude * 30

            sx = center_x + math.cos(angle) * dist
            sy = center_y + math.sin(angle) * dist

            # Shards rotate
            shard_rot = t * 5 + i
            shard_size = 15 + random.gauss(0, 5)

            # Shard shape (small triangle)
            shard_points = [
                (sx + math.cos(shard_rot) * shard_size,
                 sy + math.sin(shard_rot) * shard_size),
                (sx + math.cos(shard_rot + 2.1) * shard_size,
                 sy + math.sin(shard_rot + 2.1) * shard_size),
                (sx + math.cos(shard_rot + 4.2) * shard_size,
                 sy + math.sin(shard_rot + 4.2) * shard_size),
            ]

            colors_list = list(COLORS.values())
            shard_color = colors_list[i % len(colors_list)]
            opacity = int(255 * (1 - stage_progress * 0.7))

            draw.polygon(shard_points, fill=shard_color + (opacity,))

    # === POST-PROCESSING ===

    # Add glow/bloom
    img_glow = img.filter(ImageFilter.GaussianBlur(radius=10))
    img = Image.blend(img, img_glow, 0.3)

    # Enhance saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.5)

    # Add glitch effects (intensity varies with progress and audio)
    glitch_intensity = amplitude * 0.3 + (0.5 if 0.5 < progress < 0.75 else 0.2)
    img = add_glitch_effect(img, glitch_intensity)

    # Add scan line overlay
    overlay = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for y in range(0, HEIGHT, 3):
        overlay_draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, 30))
    img.paste(overlay, (0, 0), overlay)

    return img

def main():
    print("="*80)
    print("Generating Digital Parrot Visuals")
    print(f"Duration: {DURATION}s | FPS: {FPS} | Total frames: {TOTAL_FRAMES}")
    print("="*80)

    # Create frames directory
    import os
    frames_dir = "frames_parrot"
    os.makedirs(frames_dir, exist_ok=True)

    # Generate frames
    for frame_num in range(TOTAL_FRAMES):
        if frame_num % 25 == 0:
            progress_pct = (frame_num / TOTAL_FRAMES) * 100
            print(f"Progress: {frame_num}/{TOTAL_FRAMES} frames ({progress_pct:.1f}%)")

        img = generate_frame(frame_num)
        img.save(f"{frames_dir}/frame_{frame_num:05d}.png")

    print("\n✓ All frames generated!")
    print(f"Frames saved to: {frames_dir}/")

    # Combine with audio using ffmpeg
    print("\n" + "="*80)
    print("Rendering final video with audio...")
    print("="*80)

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', f'{frames_dir}/frame_%05d.png',
        '-i', 'song_papegoje.wav',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        'parrot_visual_full.mp4'
    ]

    subprocess.run(cmd)

    print("\n" + "="*80)
    print("✓ Video complete: parrot_visual_full.mp4")
    print("="*80)

if __name__ == '__main__':
    main()
