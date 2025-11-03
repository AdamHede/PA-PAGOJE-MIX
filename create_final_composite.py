#!/usr/bin/env python3
"""
PA-PAGØJE Final Video Compositor
Combines all 20 track videos into the complete 45-minute festival mix video
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict


def find_all_videos() -> List[Dict[str, str]]:
    """Find all rendered track videos"""
    videos = []

    search_dirs = [
        "visuals_phase1/renders",
        "visuals_all_tracks/renders",
        "parrot_visual_full.mp4",  # Track 9 rendered separately
    ]

    for search_path in search_dirs:
        if os.path.isfile(search_path):
            # Single file
            videos.append({
                "path": search_path,
                "track": "09",  # Parrot
            })
        elif os.path.isdir(search_path):
            # Directory of videos
            for filename in os.listdir(search_path):
                if filename.endswith(".mp4"):
                    # Extract track number from filename
                    track_num = filename.split("_")[0]
                    videos.append({
                        "path": os.path.join(search_path, filename),
                        "track": track_num,
                    })

    # Sort by track number
    videos.sort(key=lambda x: int(x["track"]))

    return videos


def create_concat_file(videos: List[Dict[str, str]], output_file: str = "concat_list.txt"):
    """Create ffmpeg concat file"""
    with open(output_file, "w") as f:
        for video in videos:
            # Use absolute path
            abs_path = os.path.abspath(video["path"])
            f.write(f"file '{abs_path}'\n")

    print(f"Created concat file with {len(videos)} videos")
    return output_file


def render_final_video(
    concat_file: str,
    output_video: str = "PA-PAGOJE_Festival_Mix_FINAL_VIDEO.mp4",
    audio_mix: str = "PA-PAGOJE_Festival_Mix2.wav",
):
    """Render final composite video"""
    print("=" * 100)
    print("RENDERING FINAL COMPOSITE VIDEO")
    print("=" * 100)
    print(f"Output: {output_video}")
    print(f"Audio: {audio_mix}")
    print("=" * 100)

    # Step 1: Concatenate all videos (no audio yet)
    temp_video = "temp_concatenated_video.mp4"

    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        temp_video,
    ]

    print("\nStep 1: Concatenating videos...")
    result = subprocess.run(cmd_concat, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR concatenating videos:")
        print(result.stderr)
        return False

    print("✓ Videos concatenated")

    # Step 2: Add master audio mix
    cmd_audio = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", audio_mix,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "320k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_video,
    ]

    print("\nStep 2: Adding master audio mix...")
    result = subprocess.run(cmd_audio, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR adding audio:")
        print(result.stderr)
        return False

    print("✓ Audio mixed")

    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)

    # Get final file info
    size_mb = os.path.getsize(output_video) / 1e6

    print("\n" + "=" * 100)
    print("✓✓✓ FINAL VIDEO COMPLETE ✓✓✓")
    print("=" * 100)
    print(f"File: {output_video}")
    print(f"Size: {size_mb:.1f} MB ({size_mb / 1024:.2f} GB)")
    print("=" * 100)

    return True


def verify_videos(videos: List[Dict[str, str]]):
    """Verify all expected videos exist"""
    print("\nVerifying track videos...")
    print("=" * 100)

    expected_tracks = [f"{i:02d}" for i in range(1, 21)]
    found_tracks = [v["track"] for v in videos]

    missing = []
    for track in expected_tracks:
        if track not in found_tracks:
            missing.append(track)

    if missing:
        print(f"⚠ WARNING: Missing tracks: {', '.join(missing)}")
        print(f"Found: {len(found_tracks)}/20 tracks")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False
    else:
        print(f"✓ All 20 tracks found")

    # Show all videos
    print("\nTrack videos:")
    for video in videos:
        size_mb = os.path.getsize(video["path"]) / 1e6
        print(f"  Track {video['track']}: {os.path.basename(video['path'])} ({size_mb:.1f} MB)")

    total_size = sum(os.path.getsize(v["path"]) for v in videos) / 1e6
    print(f"\nTotal input size: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")
    print("=" * 100)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create final composite PA-PAGØJE video")
    parser.add_argument("--output", default="PA-PAGOJE_Festival_Mix_FINAL_VIDEO.mp4", help="Output video filename")
    parser.add_argument("--audio", default="PA-PAGOJE_Festival_Mix2.wav", help="Master audio mix")
    parser.add_argument("--force", action="store_true", help="Skip verification prompts")
    args = parser.parse_args()

    # Find all videos
    videos = find_all_videos()

    if not videos:
        print("ERROR: No track videos found!")
        print("Make sure you've rendered the tracks first using render_queue_manager.py")
        return 1

    # Verify
    if not args.force:
        if not verify_videos(videos):
            print("Aborted.")
            return 1

    # Create concat file
    concat_file = create_concat_file(videos)

    # Render final video
    success = render_final_video(concat_file, args.output, args.audio)

    # Cleanup concat file
    if os.path.exists(concat_file):
        os.remove(concat_file)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
