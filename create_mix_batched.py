#!/usr/bin/env python3
import subprocess
import json
import os

# Split into batches to avoid large file issues
BATCH_1 = [
    "Se den lille kattekilling (Deep House).mp3",
    "I en kælder sort som kul (Dub techno).mp3",
    "Hundred' mus med haler på (Microhouse).mp3",
    "Spørge Jørgen (Minimal techno _ tech house).mp3",
    "Der sad to katte på et bord (Tech house).mp3",
    "Mariehønen Evigglad (French house).mp3",
    "Lille sky gik morgentur (Future garage).mp3",
    "Hønsefødder og gulerødder (Glitch hop).mp3",
    "Jeg er en papegøje fra Amerika (Electroclash _ Electro house).mp3",
    "Op lille Hans (Big beat _ breakbeat).mp3",
]

BATCH_2 = [
    "Oles nye autobil (EBM, Industrial Techno) (TØF TØF TØF MIX).mp3",
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
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', filepath],
        capture_output=True, text=True
    )
    return float(json.loads(result.stdout)['format']['duration'])

def calculate_transition_bars(duration1, duration2):
    min_duration = min(duration1, duration2)
    if min_duration < 70: return 4, 4
    elif min_duration < 100: return 4, 4
    elif min_duration < 150: return 4, 4
    elif min_duration < 200: return 6, 6
    else: return 8, 8

def crossfade(track1, track2, output, bars_ts, bars_cf):
    cmd = [
        'docker', 'run', '--rm',
        '-v', '/Users/adamhede/Downloads/tracks:/app/audios',
        '-v', '/Users/adamhede/.pycrossfade/annotations:/app/pycrossfade_annotations',
        '-e', 'ANNOTATIONS_DIRECTORY=/app/pycrossfade_annotations',
        '-e', 'BASE_AUDIO_DIRECTORY=/app/audios/',
        'ghcr.io/oguzhan-yilmaz/pycrossfade:latest',
        'crossfade', '-t', str(bars_ts), '-c', str(bars_cf), '-o', output, track1, track2
    ]
    print(f"  Mixing: {track1} + {track2} → {output}")
    return subprocess.run(cmd, capture_output=True).returncode == 0

def mix_batch(tracks, output_name):
    print(f"\n{'='*80}")
    print(f"BATCH: {output_name}")
    print(f"{'='*80}")

    current = tracks[0]
    for i in range(1, len(tracks)):
        d1 = get_duration(current) if i > 1 else get_duration(current)
        d2 = get_duration(tracks[i])
        bars_ts, bars_cf = calculate_transition_bars(d1, d2)

        temp_out = f"batch_temp_{i}.wav"
        if not crossfade(current, tracks[i], temp_out, bars_ts, bars_cf):
            print(f"ERROR at track {i+1}")
            return None

        if i > 1 and os.path.exists(current):
            os.remove(current)
        current = temp_out
        print(f"  ✓ {i}/{len(tracks)-1} transitions")

    os.rename(current, output_name)
    return output_name

# Mix batch 1
batch1_out = mix_batch(BATCH_1, "PA-PAGOJE_BATCH_1.wav")
if not batch1_out:
    print("Batch 1 failed")
    exit(1)

print(f"\n✓ Batch 1 complete: {int(get_duration(batch1_out)//60)}:{int(get_duration(batch1_out)%60):02d}")

# Mix batch 2
batch2_out = mix_batch(BATCH_2, "PA-PAGOJE_BATCH_2.wav")
if not batch2_out:
    print("Batch 2 failed")
    exit(1)

print(f"\n✓ Batch 2 complete: {int(get_duration(batch2_out)//60)}:{int(get_duration(batch2_out)%60):02d}")

# Combine batches using simple ffmpeg concat
print(f"\n{'='*80}")
print("COMBINING BATCHES")
print(f"{'='*80}")

# Use ffmpeg to concat (no crossfade, just append)
with open("concat_list.txt", "w") as f:
    f.write(f"file '{batch1_out}'\n")
    f.write(f"file '{batch2_out}'\n")

cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'concat_list.txt', '-c', 'copy', 'PA-PAGOJE_COMPLETE_RAW.wav']
subprocess.run(cmd, capture_output=True)

final_dur = get_duration("PA-PAGOJE_COMPLETE_RAW.wav")
print(f"\n✓ COMPLETE MIX: {int(final_dur//60)}:{int(final_dur%60):02d}")
print(f"✓ Output: PA-PAGOJE_COMPLETE_RAW.wav")
