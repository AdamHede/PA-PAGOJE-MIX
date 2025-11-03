#!/usr/bin/env python3
import subprocess
import json
import os

# PA-PAGØJE Festival Set Order - OPTIMIZED
# Based on BPM analysis, harmonic mixing (Camelot wheel), and energy flow
SETLIST = [
    # WARM-UP / OPENING (120-125 BPM)
    "Se den lille kattekilling (Deep House).mp3",
    "I en kælder sort som kul (Dub techno).mp3",
    "Hundred' mus med haler på (Microhouse).mp3",
    "Spørge Jørgen (Minimal techno _ tech house).mp3",

    # FIRST BUILD (125-127 BPM)
    "Der sad to katte på et bord (Tech house).mp3",
    "Mariehønen Evigglad (French house).mp3",
    "Lille sky gik morgentur (Future garage).mp3",
    "Hønsefødder og gulerødder (Glitch hop).mp3",

    # FIRST PEAK (128-133 BPM)
    "Jeg er en papegøje fra Amerika (Electroclash _ Electro house).mp3",
    "Op lille Hans (Big beat _ breakbeat).mp3",
    "Oles nye autobil (EBM, Industrial Techno) (TØF TØF TØF MIX).mp3",
    "Jeg gik mig over sø og land (Prog House).mp3",
    "I østen stiger solen op (acid techno, acid house).mp3",

    # MID-SET BREATHER (98-99 BPM)
    "I skovens dybe stille ro (Organic).mp3",
    "Hist, hvor vejen slår en bugt (Downtempo).mp3",

    # SECOND BUILD (150-178 BPM)
    "Fra Engeland til Skotland (Trip Hop).mp3",
    "Tre små fisk (drumstep).mp3",

    # FINAL PEAK (168-172 BPM DnB)
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

    if min_duration < 70:      # < 70s: very short
        return 2, 2
    elif min_duration < 100:   # 70-100s: short
        return 3, 3
    elif min_duration < 150:   # 100-150s: medium-short
        return 4, 4
    elif min_duration < 200:   # 150-200s: medium
        return 6, 6
    else:                      # > 200s: long
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

    print(f"\n{'='*80}")
    print(f"Mixing: {track1}")
    print(f"   and: {track2}")
    print(f"Transition: {len_ts} bars time-stretch, {len_cf} bars crossfade")
    print(f"Output: {output}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

# Main execution
print("PA-PAGØJE Festival Mix Generator")
print("="*80)
print(f"Total tracks: {len(SETLIST)}")
print("\nAnalyzing track durations...")

# Get all durations
durations = {}
for track in SETLIST:
    durations[track] = get_duration(track)
    mins = int(durations[track] // 60)
    secs = int(durations[track] % 60)
    print(f"  {mins:02d}:{secs:02d} - {track}")

total_duration = sum(durations.values())
print(f"\nTotal raw duration: {int(total_duration // 60)}:{int(total_duration % 60):02d}")

# Build mix progressively
print("\n" + "="*80)
print("Building festival mix...")
print("="*80)

current_mix = SETLIST[0]
temp_counter = 1

for i in range(1, len(SETLIST)):
    next_track = SETLIST[i]

    # Calculate transition length
    if current_mix.startswith("temp_mix_"):
        # For intermediate mixes, estimate duration from original tracks
        dur1 = 180  # reasonable estimate
    else:
        dur1 = durations[current_mix]

    dur2 = durations[next_track]
    len_ts, len_cf = calculate_transition_bars(dur1, dur2)

    # Output filename
    if i == len(SETLIST) - 1:
        output = "PA-PAGOJE_Festival_Mix_FINAL.wav"
    else:
        output = f"temp_mix_{temp_counter}.wav"
        temp_counter += 1

    # Run crossfade
    success = run_docker_crossfade(current_mix, next_track, output, len_ts, len_cf)

    if not success:
        print(f"\n❌ ERROR: Failed to mix tracks!")
        break

    # Clean up old temp file
    if current_mix.startswith("temp_mix_") and os.path.exists(current_mix):
        os.remove(current_mix)
        print(f"✓ Cleaned up {current_mix}")

    current_mix = output
    print(f"✓ Progress: {i}/{len(SETLIST)-1} transitions complete")

print("\n" + "="*80)
print("✓ Festival mix complete!")
print(f"Output: PA-PAGOJE_Festival_Mix_FINAL.wav")

# Get final duration
if os.path.exists("PA-PAGOJE_Festival_Mix_FINAL.wav"):
    final_dur = get_duration("PA-PAGOJE_Festival_Mix_FINAL.wav")
    mins = int(final_dur // 60)
    secs = int(final_dur % 60)
    print(f"Final duration: {mins}:{secs:02d}")
print("="*80)
