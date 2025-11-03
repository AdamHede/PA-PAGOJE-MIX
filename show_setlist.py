#!/usr/bin/env python3
import json

# Load analysis data
with open('track_analysis.json', 'r') as f:
    analysis = json.load(f)

# Create lookup dictionary
track_data = {item['track']: item for item in analysis}

# Camelot wheel mapping
CAMELOT = {
    'G major': '9B', 'Bb minor': '3A', 'Ab minor': '1A', 'E minor': '9A',
    'G minor': '6A', 'D minor': '7A', 'A minor': '8A', 'C major': '8B',
    'Bb major': '6B', 'F# major': '2B', 'F minor': '4A', 'F major': '7B',
    'Eb minor': '12A'
}

# Optimized setlist
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

PHASES = {
    0: ("WARM-UP / OPENING", "Establish the groove"),
    4: ("FIRST BUILD", "Energy rising"),
    8: ("FIRST PEAK", "High energy explosion"),
    13: ("MID-SET BREATHER", "Come down, rebuild"),
    15: ("SECOND BUILD", "Tension rising again"),
    17: ("FINAL PEAK", "Maximum energy - DnB finale"),
}

print("="*120)
print("PA-PAGØJE: OPTIMIZED FESTIVAL SET ORDER 🎧")
print("="*120)
print()

total_duration = 0
current_phase = None

for i, track in enumerate(SETLIST):
    # Check if new phase
    if i in PHASES:
        phase_name, phase_desc = PHASES[i]
        if current_phase:
            print()
        print(f"\n{'─'*120}")
        print(f"  {phase_name} — {phase_desc}")
        print(f"{'─'*120}\n")
        current_phase = phase_name

    data = track_data[track]
    mins = int(data['duration'] // 60)
    secs = int(data['duration'] % 60)
    total_duration += data['duration']

    camelot = CAMELOT.get(data['key'], '??')

    # Extract genre from filename
    genre = track.split('(')[1].split(')')[0] if '(' in track else 'Unknown'
    track_name = track.split('(')[0].strip()

    # BPM transition indicator
    if i > 0:
        prev_bpm = track_data[SETLIST[i-1]]['bpm']
        bpm_diff = data['bpm'] - prev_bpm
        if bpm_diff > 0:
            bpm_arrow = f"↑+{bpm_diff}"
        elif bpm_diff < 0:
            bpm_arrow = f"↓{bpm_diff}"
        else:
            bpm_arrow = "→"
    else:
        bpm_arrow = " "

    print(f"  {i+1:2d}. {track_name:<45} │ {data['bpm']:3d} BPM {bpm_arrow:>5} │ {camelot:>4} ({data['key']:<12}) │ {mins:02d}:{secs:02d}")

print(f"\n{'='*120}")
total_mins = int(total_duration // 60)
total_secs = int(total_duration % 60)
print(f"Total Duration (raw): {total_mins}:{total_secs:02d}")
print(f"Estimated Mix (with transitions): ~{total_mins - 6}:{total_secs:02d} - {total_mins - 4}:{total_secs:02d}")
print(f"{'='*120}")
