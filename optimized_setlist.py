#!/usr/bin/env python3
"""
PA-PAGØJE: Optimized Festival Set
Based on BPM, harmonic (Camelot wheel), and energy analysis

Strategy:
- Harmonic mixing using Camelot wheel (stay within 1-2 steps)
- Smooth BPM transitions (avoid big jumps except at breaks)
- Festival energy arc: Build → Peak → Breather → Build → Final Peak
- Consider track length for transition planning
"""

# Camelot wheel reference for harmonic mixing
CAMELOT = {
    'G major': '9B', 'Bb minor': '3A', 'Ab minor': '1A', 'E minor': '9A',
    'G minor': '6A', 'D minor': '7A', 'A minor': '8A', 'C major': '8B',
    'Bb major': '6B', 'F# major': '2B', 'F minor': '4A', 'F major': '7B',
    'Eb minor': '12A'
}

# Optimized setlist with reasoning
OPTIMIZED_SETLIST = [
    # === WARM-UP / OPENING (120-125 BPM) ===
    # Start welcoming, build foundation
    {
        "track": "Se den lille kattekilling (Deep House).mp3",
        "bpm": 123, "key": "9B", "phase": "WARM-UP",
        "reason": "Warm, welcoming Deep House opener - sets the mood"
    },
    {
        "track": "I en kælder sort som kul (Dub techno).mp3",
        "bpm": 120, "key": "4A", "phase": "WARM-UP",
        "reason": "Deep, echoing Dub - establishes darker foundation"
    },
    {
        "track": "Hundred' mus med haler på (Microhouse).mp3",
        "bpm": 125, "key": "3A", "phase": "WARM-UP",
        "reason": "Minimal groove building, 3A→4A smooth harmonic flow"
    },
    {
        "track": "Spørge Jørgen (Minimal techno _ tech house).mp3",
        "bpm": 125, "key": "1A", "phase": "WARM-UP",
        "reason": "Hypnotic minimal, 1A→3A adjacent on wheel"
    },

    # === FIRST BUILD (125-133 BPM) ===
    # Rising energy, groove intensifies
    {
        "track": "Der sad to katte på et bord (Tech house).mp3",
        "bpm": 125, "key": "9A", "phase": "BUILD-1",
        "reason": "Tech house groove locked in, short track (62s) keeps momentum"
    },
    {
        "track": "Mariehønen Evigglad (French house).mp3",
        "bpm": 125, "key": "7A", "phase": "BUILD-1",
        "reason": "Happy French house filter disco! 7A→9A smooth, energy rising"
    },
    {
        "track": "Lille sky gik morgentur (Future garage).mp3",
        "bpm": 126, "key": "2B", "phase": "BUILD-1",
        "reason": "Atmospheric clouds, slight tempo bump, genre variety"
    },
    {
        "track": "Hønsefødder og gulerødder (Glitch hop).mp3",
        "bpm": 127, "key": "6B", "phase": "BUILD-1",
        "reason": "Playful wonky energy, 6B close to 2B, short (60s) transitional piece"
    },

    # === FIRST PEAK (128-133 BPM) ===
    # High energy explosion
    {
        "track": "Jeg er en papegøje fra Amerika (Electroclash _ Electro house).mp3",
        "bpm": 128, "key": "8B", "phase": "PEAK-1",
        "reason": "Sassy electroclash peak! 8B→6B smooth transition"
    },
    {
        "track": "Op lille Hans (Big beat _ breakbeat).mp3",
        "bpm": 128, "key": "6B", "phase": "PEAK-1",
        "reason": "FAT breaks! Morning energy, stays at 128, 6B perfect match"
    },
    {
        "track": "Oles nye autobil (EBM, Industrial Techno) (TØF TØF TØF MIX).mp3",
        "bpm": 130, "key": "2B", "phase": "PEAK-1",
        "reason": "TØF TØF mechanical power! 2B↔6B = 4 steps but genre supports it"
    },
    {
        "track": "Jeg gik mig over sø og land (Prog House).mp3",
        "bpm": 132, "key": "8A", "phase": "PEAK-1",
        "reason": "Epic progressive build! Long track (232s), energy intensifies"
    },
    {
        "track": "I østen stiger solen op (acid techno, acid house).mp3",
        "bpm": 133, "key": "1A", "phase": "PEAK-1",
        "reason": "303 squelch sunrise! Peak climax, 1A→8A transition"
    },

    # === BREATHER / BREAKDOWN (98-99 BPM) ===
    # Come down, reset for final push
    {
        "track": "I skovens dybe stille ro (Organic).mp3",
        "bpm": 98, "key": "6A", "phase": "BREATHER",
        "reason": "Nature sounds, organic break - big tempo drop resets energy"
    },
    {
        "track": "Hist, hvor vejen slår en bugt (Downtempo).mp3",
        "bpm": 99, "key": "7B", "phase": "BREATHER",
        "reason": "Winding road journey, cinematic moment, 6A→7B adjacent"
    },

    # === SECOND BUILD (125-150 BPM) ===
    # Tension rising again toward finale
    {
        "track": "Fra Engeland til Skotland (Trip Hop).mp3",
        "bpm": 178, "key": "12A", "phase": "BUILD-2",
        "reason": "Cinematic trip hop - BPM detected at 178 (double-time feel), builds tension"
    },
    {
        "track": "Tre små fisk (drumstep).mp3",
        "bpm": 150, "key": "4A", "phase": "BUILD-2",
        "reason": "Aquatic wobbles, drumstep bridge to DnB finale, 4A→12A flow"
    },

    # === FINAL PEAK (168-172 BPM) ===
    # Maximum energy, DnB finale
    {
        "track": "Tre små kinesere (Jungle).mp3",
        "bpm": 84, "key": "7B", "phase": "PEAK-2",
        "reason": "Amen breaks! 84 BPM = half-time detection (feels 168), jungle energy"
    },
    {
        "track": "Ti små cyklister (DnB).mp3",
        "bpm": 172, "key": "6B", "phase": "PEAK-2",
        "reason": "Full speed cycling madness! 6B→7B smooth, 172 BPM DnB peak"
    },
    {
        "track": "Se dig for (Techstep _ DnB).mp3",
        "bpm": 172, "key": "2B", "phase": "PEAK-2",
        "reason": "Dark technical closer - leaves them wanting more! 2B↔6B = 4 steps but both DnB"
    },
]

# Print setlist
print("="*100)
print("PA-PAGØJE: OPTIMIZED FESTIVAL SET ORDER 🎧")
print("="*100)
print()

phases = {}
for item in OPTIMIZED_SETLIST:
    if item['phase'] not in phases:
        phases[item['phase']] = []
    phases[item['phase']].append(item)

phase_names = {
    'WARM-UP': 'WARM-UP / OPENING',
    'BUILD-1': 'FIRST BUILD',
    'PEAK-1': 'FIRST PEAK',
    'BREATHER': 'MID-SET BREATHER',
    'BUILD-2': 'SECOND BUILD',
    'PEAK-2': 'FINAL PEAK'
}

for phase_key in ['WARM-UP', 'BUILD-1', 'PEAK-1', 'BREATHER', 'BUILD-2', 'PEAK-2']:
    if phase_key in phases:
        print(f"\n{'='*100}")
        print(f"{phase_names[phase_key]}")
        print(f"{'='*100}")
        for item in phases[phase_key]:
            print(f"\n  {item['track']}")
            print(f"    BPM: {item['bpm']} | Key: {item['key']}")
            print(f"    → {item['reason']}")

print("\n" + "="*100)
print(f"Total tracks: {len(OPTIMIZED_SETLIST)}")
print("="*100)

# Export just the track list
print("\n\nTRACK LIST FOR SCRIPT:")
print("="*100)
for i, item in enumerate(OPTIMIZED_SETLIST, 1):
    print(f'{i:2d}. "{item["track"]}"')
