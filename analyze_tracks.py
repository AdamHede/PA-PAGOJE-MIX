#!/usr/bin/env python3
import subprocess
import re
import json

TRACKS = [
    "Se den lille kattekilling (Deep House).mp3",
    "Hundred' mus med haler på (Microhouse).mp3",
    "Spørge Jørgen (Minimal techno _ tech house).mp3",
    "Der sad to katte på et bord (Tech house).mp3",
    "I skovens dybe stille ro (Organic).mp3",
    "Mariehønen Evigglad (French house).mp3",
    "Jeg gik mig over sø og land (Prog House).mp3",
    "I østen stiger solen op (acid techno, acid house).mp3",
    "Jeg er en papegøje fra Amerika (Electroclash _ Electro house).mp3",
    "Op lille Hans (Big beat _ breakbeat).mp3",
    "Oles nye autobil (EBM, Industrial Techno) (TØF TØF TØF MIX).mp3",
    "I en kælder sort som kul (Dub techno).mp3",
    "Hist, hvor vejen slår en bugt (Downtempo).mp3",
    "Fra Engeland til Skotland (Trip Hop).mp3",
    "Lille sky gik morgentur (Future garage).mp3",
    "Hønsefødder og gulerødder (Glitch hop).mp3",
    "Tre små fisk (drumstep).mp3",
    "Tre små kinesere (Jungle).mp3",
    "Ti små cyklister (DnB).mp3",
    "Se dig for (Techstep _ DnB).mp3",
]

def extract_metadata(track):
    """Extract BPM, key, danceability using pyCrossfade"""
    cmd = [
        'docker', 'run', '--rm',
        '-v', '/Users/adamhede/Downloads/tracks:/app/audios',
        '-v', f'{subprocess.os.path.expanduser("~")}/.pycrossfade/annotations:/app/pycrossfade_annotations',
        '-e', 'ANNOTATIONS_DIRECTORY=/app/pycrossfade_annotations',
        '-e', 'BASE_AUDIO_DIRECTORY=/app/audios/',
        'ghcr.io/oguzhan-yilmaz/pycrossfade:latest',
        'extract',
        track
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # Parse output
    bpm = None
    key = None
    danceability = None
    duration = None

    for line in output.split('\n'):
        if 'BPM (rounded)' in line:
            match = re.search(r'(\d+)', line.split('BPM (rounded)')[-1])
            if match:
                bpm = int(match.group(1))
        elif 'Key/Scale estimation (edma)' in line:
            match = re.search(r'\[conf\.: [\d.]+\]\s+(.+)', line)
            if match:
                key = match.group(1).strip()
        elif 'Danceability' in line:
            match = re.search(r'([\d.]+)/3\.00', line)
            if match:
                danceability = float(match.group(1))
        elif 'Duration (seconds)' in line:
            match = re.search(r'([\d.]+)', line.split('Duration (seconds)')[-1])
            if match:
                duration = float(match.group(1))

    return {
        'track': track,
        'bpm': bpm,
        'key': key,
        'danceability': danceability,
        'duration': duration
    }

print("PA-PAGØJE Track Analysis")
print("="*100)
print(f"{'Track':<50} {'BPM':<6} {'Key':<15} {'Dance':<7} {'Duration'}")
print("="*100)

results = []
for track in TRACKS:
    print(f"Analyzing: {track[:50]:<50}", end=' ', flush=True)
    data = extract_metadata(track)
    results.append(data)

    mins = int(data['duration'] // 60) if data['duration'] else 0
    secs = int(data['duration'] % 60) if data['duration'] else 0

    print(f"{data['bpm'] or 'N/A':<6} {data['key'] or 'N/A':<15} {data['danceability'] or 'N/A':<7} {mins:02d}:{secs:02d}")

# Save to JSON
with open('track_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*100)
print("✓ Analysis complete! Data saved to track_analysis.json")
print("\nBPM Range:", min(r['bpm'] for r in results if r['bpm']), "-", max(r['bpm'] for r in results if r['bpm']))
