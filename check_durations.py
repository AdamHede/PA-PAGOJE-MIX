#!/usr/bin/env python3
import subprocess
import json
import glob

tracks = sorted(glob.glob("*.mp3"))

print(f"{'Duration':<12} {'File'}")
print("="*80)

for track in tracks:
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', track],
        capture_output=True,
        text=True
    )
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"{minutes:02d}:{seconds:02d} ({duration:6.1f}s)  {track}")
