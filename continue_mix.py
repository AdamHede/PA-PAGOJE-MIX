#!/usr/bin/env python3
import subprocess
import json
import os

# Remaining tracks to mix (tracks 12-20)
REMAINING_TRACKS = [
    "Jeg gik mig over sø og land (Prog House).mp3",
    "I østen stiger solen op (acid techno, acid house).mp3",
    "I skovens dybe stille ro (Organic).mp3",
    "Hist, hvor vejen slår en bugt (Downtempo).mp3",
    "Fra Engeland til Skotland (Trip Hop).mp3",
    "Tre små fisk (drumstep).mp3",
    "Tre små kinesere (Jungle).mp3",
    "Ti små cyklister (DnB).mp3",
    "Se dig for (Techstep _ DnB).mp3",
]

def get_duration(filepath):
    """Get track duration in seconds using ffprobe"""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', filepath],
        capture_output=True,
        text=True
    )
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def calculate_transition_bars(duration1, duration2):
    """Calculate optimal transition length based on shorter track"""
    min_duration = min(duration1, duration2)
    if min_duration < 70:
        return 2, 2
    elif min_duration < 100:
        return 3, 3
    elif min_duration < 150:
        return 4, 4
    elif min_duration < 200:
        return 6, 6
    else:
        return 8, 8

def run_docker_crossfade(track1, track2, output, len_ts, len_cf):
    """Run pyCrossfade docker command for two tracks"""
    cmd = [
        'docker', 'run', '--rm',
        '-v', '/Users/adamhede/Downloads/tracks:/app/audios',
        '-v', f'{os.path.expanduser("~")}/.pycrossfade/annotations:/app/pycrossfade_annotations',
        '-e', 'ANNOTATIONS_DIRECTORY=/app/pycrossfade_annotations',
        '-e', 'BASE_AUDIO_DIRECTORY=/app/audios/',
        'ghcr.io/oguzhan-yilmaz/pycrossfade:latest',
        'crossfade',
        '-t', str(len_ts),
        '-c', str(len_cf),
        '-o', output,
        track1,
        track2
    ]

    print(f"\nMixing: {track1} + {track2}")
    print(f"Transition: {len_ts}bars/{len_cf}bars → {output}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

print("="*80)
print("Continuing PA-PAGØJE Mix from temp_mix_10.mp3")
print("="*80)

current_mix = "temp_mix_10.mp3"
counter = 11

for i, next_track in enumerate(REMAINING_TRACKS):
    dur1 = get_duration(current_mix) if os.path.exists(current_mix) else 180
    dur2 = get_duration(next_track)
    len_ts, len_cf = calculate_transition_bars(dur1, dur2)

    if i == len(REMAINING_TRACKS) - 1:
        output = "PA-PAGOJE_Festival_Mix_FINAL.wav"
    else:
        output = f"temp_mix_{counter}.mp3"
        counter += 1

    success = run_docker_crossfade(current_mix, next_track, output, len_ts, len_cf)

    if not success:
        print(f"\n❌ ERROR at track {i+12}/20")
        break

    # Convert to MP3 if not final
    if output != "PA-PAGOJE_Festival_Mix_FINAL.wav" and output.endswith('.mp3'):
        pass  # already MP3
    elif output != "PA-PAGOJE_Festival_Mix_FINAL.wav":
        print(f"Converting {output} to MP3...")
        mp3_output = output.replace('.wav', '.mp3')
        subprocess.run(['ffmpeg', '-i', output, '-b:a', '192k', mp3_output, '-y'],
                      capture_output=True)
        os.remove(output)
        current_mix = mp3_output
    else:
        current_mix = output

    # Clean up old temp
    if i > 0 and f"temp_mix_{counter-2}.mp3" != "temp_mix_10.mp3":
        old_file = f"temp_mix_{counter-2}.mp3"
        if os.path.exists(old_file):
            os.remove(old_file)

    print(f"✓ Progress: {i+11}/19 transitions complete")

print("\n" + "="*80)
print("✓ Mix complete!")
print("="*80)
