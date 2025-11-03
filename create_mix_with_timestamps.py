#!/usr/bin/env python3
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --- Configuration -----------------------------------------------------------------

SETLIST: List[str] = [
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

WORKDIR = Path("mix_workspace")
WORKDIR.mkdir(exist_ok=True)

FINAL_WAV = Path("PA-PAGOJE_Festival_Mix.wav")
FINAL_MP3 = Path("PA-PAGOJE_Festival_Mix.mp3")
TIMESTAMP_JSON = Path("PA-PAGOJE_TIMESTAMPS.json")
TRANSITION_LOG = Path("PA-PAGOJE_TRANSITION_LOG.md")

DOCKER_IMAGE = "ghcr.io/oguzhan-yilmaz/pycrossfade:latest"
HOST_AUDIO_ROOT = Path("/Users/adamhede/Downloads/tracks")
HOST_ANNOTATIONS = Path("/Users/adamhede/.pycrossfade/annotations")


# --- Utility helpers ----------------------------------------------------------------

def run_cmd(cmd: Sequence[str], purpose: str, log_buffer: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, capture output, and append to log buffer."""
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    log_buffer.append(f"$ {cmd_str}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        log_buffer.append(result.stdout.strip())
    if result.stderr:
        log_buffer.append(result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(f"{purpose} failed (exit {result.returncode})")
    return result


def get_duration(path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def convert_to_wav(src: Path, dst: Path, log_buffer: List[str]) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    run_cmd(cmd, f"convert {src.name} to WAV", log_buffer)


def slice_audio(src: Path, dst: Path, start: Optional[float], end: Optional[float], log_buffer: List[str]) -> None:
    """Extract a segment from src. Start/end are absolute seconds; end=None means to EOF."""
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if start is not None and start > 0:
        cmd.extend(["-ss", f"{start:.6f}"])
    if end is not None:
        duration = end - (start or 0.0)
        if duration <= 0:
            raise ValueError("Invalid slice range (non-positive duration).")
        cmd.extend(["-t", f"{duration:.6f}"])
    cmd.extend(["-c", "copy", str(dst)])
    run_cmd(cmd, f"slice {src.name}", log_buffer)


def concat_audios(paths: Sequence[Path], dst: Path, log_buffer: List[str]) -> None:
    """Concatenate WAV files in order."""
    concat_list = WORKDIR / "concat_list.txt"
    with concat_list.open("w") as fh:
        for path in paths:
            fh.write(f"file '{path.resolve().as_posix()}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(dst),
    ]
    run_cmd(cmd, f"concat {len(paths)} segments", log_buffer)
    concat_list.unlink(missing_ok=True)


def load_track_metadata() -> Dict[str, Dict[str, float]]:
    meta_path = Path("track_analysis.json")
    if not meta_path.exists():
        return {}
    with meta_path.open() as fh:
        data = json.load(fh)
    metadata: Dict[str, Dict[str, float]] = {}
    for item in data:
        metadata[item["track"]] = item
    return metadata


def calculate_transition_bars(duration1: float, duration2: float) -> Sequence[int]:
    """Heuristic transition selection based on shorter track length."""
    min_duration = min(duration1, duration2)
    if min_duration < 70:
        return 2, 2
    if min_duration < 100:
        return 3, 3
    if min_duration < 150:
        return 4, 4
    if min_duration < 200:
        return 6, 6
    return 8, 8


def parse_crossfade_output(output: str) -> Dict[str, float]:
    meta: Dict[str, float] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[") or line.startswith("WARNING"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        value_token = parts[-1]
        try:
            meta[key] = float(value_token)
        except ValueError:
            continue
    return meta


def run_crossfade(master: Path, slave: Path, output: Path, len_ts: int, len_cf: int, log_buffer: List[str]) -> Dict[str, float]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{HOST_AUDIO_ROOT}:/app/audios",
        "-v",
        f"{HOST_ANNOTATIONS}:/app/pycrossfade_annotations",
        "-e",
        "ANNOTATIONS_DIRECTORY=/app/pycrossfade_annotations",
        "-e",
        "BASE_AUDIO_DIRECTORY=/app/audios/",
        DOCKER_IMAGE,
        "crossfade",
        "--verbose",
        "-t",
        str(len_ts),
        "-c",
        str(len_cf),
        "-o",
        output.as_posix(),
        master.as_posix(),
        slave.as_posix(),
    ]
    result = run_cmd(cmd, f"crossfade {master.name} ➜ {slave.name}", log_buffer)
    metadata = parse_crossfade_output("\n".join([result.stdout, result.stderr]))
    if "slave_start_seconds" not in metadata:
        raise RuntimeError("crossfade metadata missing slave_start_seconds")
    return metadata


def convert_wav_to_mp3(src: Path, dst: Path, bitrate: str, log_buffer: List[str]) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-b:a", bitrate, str(dst)]
    run_cmd(cmd, f"encode {dst.name}", log_buffer)


def write_transition_log(entries: List[Dict[str, float]], metadata: Dict[str, Dict[str, float]], total_duration: float) -> None:
    lines = [
        "# PA-PAGØJE Mix Transition Log",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| # | Start | Track | BPM | Duration |",
        "|---|-------|-------|-----|----------|",
    ]
    for idx, entry in enumerate(entries, start=1):
        track = entry["track"]
        bpm = metadata.get(track, {}).get("bpm", "—")
        duration = entry["duration"]
        lines.append(
            f"| {idx} | {format_time(entry['start'])} | {track} | {bpm} | {duration:0.2f}s |"
        )
    lines.append("")
    lines.append(f"Total mix duration: {format_time(total_duration)}")
    TRANSITION_LOG.write_text("\n".join(lines))


def main() -> None:
    log_buffer: List[str] = []
    metadata = load_track_metadata()
    durations = {track: metadata.get(track, {}).get("duration") for track in SETLIST}
    HOST_ANNOTATIONS.mkdir(parents=True, exist_ok=True)

    # Re-measure durations via ffprobe for accuracy
    for track in SETLIST:
        durations[track] = get_duration(Path(track))

    first_track = SETLIST[0]
    current_mix = WORKDIR / "mix_current.wav"
    convert_to_wav(Path(first_track), current_mix, log_buffer)

    timestamp_log: List[Dict[str, float]] = [
        {
            "track": first_track,
            "start": 0.0,
            "duration": durations[first_track],
            "end": durations[first_track],
        }
    ]

    print("=" * 80)
    print("PA-PAGØJE Festival Mix Generator (memory-safe incremental mode)")
    print("=" * 80)
    print(f"Tracks to mix: {len(SETLIST)}\n")
    print(f"Track 01 start @ 00:00.00 — {first_track}")

    for idx, next_track in enumerate(SETLIST[1:], start=2):
        outgoing_entry = timestamp_log[-1]
        outgoing_track = outgoing_entry["track"]
        outgoing_start = outgoing_entry["start"]

        master_segment = WORKDIR / f"segment_{idx:02d}_master.wav"
        head_segment = WORKDIR / f"segment_{idx:02d}_head.wav"
        crossfade_segment = WORKDIR / f"segment_{idx:02d}_xf.wav"
        next_mix = WORKDIR / "mix_next.wav"

        slice_audio(current_mix, master_segment, outgoing_start, None, log_buffer)
        if outgoing_start > 1e-3:
            slice_audio(current_mix, head_segment, None, outgoing_start, log_buffer)

        len_ts, len_cf = calculate_transition_bars(durations[outgoing_track], durations[next_track])
        print(f"\nTransition {idx-1:02d}: {outgoing_track} ➜ {next_track}")
        print(f"  Using {len_ts} bars time-stretch / {len_cf} bars crossfade")

        xf_meta = run_crossfade(master_segment, Path(next_track), crossfade_segment, len_ts, len_cf, log_buffer)
        slave_start = xf_meta["slave_start_seconds"]
        next_start = outgoing_start + slave_start

        concat_parts: List[Path] = []
        if head_segment.exists():
            concat_parts.append(head_segment)
        concat_parts.append(crossfade_segment)
        concat_audios(concat_parts, next_mix, log_buffer)

        current_mix.unlink(missing_ok=True)
        next_mix.rename(current_mix)

        master_segment.unlink(missing_ok=True)
        head_segment.unlink(missing_ok=True)
        crossfade_segment.unlink(missing_ok=True)

        timestamp_log.append(
            {
                "track": next_track,
                "start": next_start,
                "duration": durations[next_track],
                "end": next_start + durations[next_track],
            }
        )

        print(f"  {next_track} starts @ {format_time(next_start)}")

    if FINAL_WAV.exists():
        FINAL_WAV.unlink()
    os.replace(current_mix, FINAL_WAV)

    convert_wav_to_mp3(FINAL_WAV, FINAL_MP3, "320k", log_buffer)

    final_duration = get_duration(FINAL_WAV)
    timestamp_log[-1]["end"] = final_duration

    with TIMESTAMP_JSON.open("w") as fh:
        json.dump(timestamp_log, fh, indent=2)

    write_transition_log(timestamp_log, metadata, final_duration)

    log_path = WORKDIR / f"mix_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(log_buffer))

    print("\n" + "=" * 80)
    print(f"✓ Mix complete — duration {format_time(final_duration)}")
    print(f"✓ WAV saved to {FINAL_WAV}")
    print(f"✓ MP3 saved to {FINAL_MP3}")
    print(f"✓ Detailed timestamps written to {TIMESTAMP_JSON}")
    print(f"✓ Transition log written to {TRANSITION_LOG}")
    print(f"✓ Command log saved to {log_path}")


if __name__ == "__main__":
    main()
