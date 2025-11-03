# PA-PAGØJE Festival Visuals - Rendering Guide

## 🎨 What You Have

Complete visual generators for all 20 tracks based on `VISUAL_PLAN_COMPLETE.md`:

### Phase 1: Warm-up (Tracks 1-3)
- **Track 01**: Liquid kitten silhouette with flowing particles
- **Track 02**: Descending wireframe architecture (basement)
- **Track 03**: Particle trail mice forming mandalas

### Phase 2: First Build (Tracks 4-8)
- **Track 04**: Single-line morphing animals (Jørgen)
- **Track 05**: Geometric cats dancing and merging
- **Track 06**: Kaleidoscopic ladybug blooming
- **Track 07**: Cloud floating over geometric landscape
- **Track 08**: Glitching chicken feet/carrot shapes

### Phase 3: First Peak (Tracks 9-13)
- **Track 09**: Digital parrot (already rendered separately!)
- **Track 10**: Explosive alarm sun with aggressive rays
- **Track 11**: Industrial blueprint car with pistons (TØF TØF TØF)
- **Track 12**: Topographic map with progressive pathfinding
- **Track 13**: Psychedelic acid-warped melting sun

### Phase 4: Breather (Tracks 14-15)
- **Track 14**: Slow organic tree growth
- **Track 15**: Zen curved road through minimal landscape

### Phase 5: Second Build (Tracks 16-17)
- **Track 16**: Network nodes spreading northward
- **Track 17**: Three synchronized geometric fish

### Phase 6: Final Peak (Tracks 18-20)
- **Track 18**: Stroboscopic rowing creating energy waves
- **Track 19**: Radial speed vortex with cyclist trails
- **Track 20**: Warning symbols explosive FINALE

---

## 🚀 Quick Start - Overnight Rendering

### Option 1: Automatic (Recommended)
```bash
chmod +x START_OVERNIGHT_RENDER.sh
./START_OVERNIGHT_RENDER.sh
```

This will:
1. Render all 20 tracks sequentially (~4-6 hours)
2. Save progress to `render_queue.json`
3. Create final 45-minute composite video
4. Log everything to `render_queue_*.log`

### Option 2: Manual Control

**Render all tracks:**
```bash
python3 render_queue_manager.py
```

**Render specific tracks only:**
```bash
python3 render_queue_manager.py --tracks 04 05 06 10
```

**Resume after interruption:**
```bash
python3 render_queue_manager.py --resume
```

**Check queue status:**
```bash
python3 render_queue_manager.py --status
```

**Create final composite:**
```bash
python3 create_final_composite.py
```

---

## 📊 Technical Details

### Rendering Specs
- **Resolution**: 1280x720 (HD)
- **Frame Rate**: 24 fps
- **Video Codec**: H.264 (libx264, CRT 18, slow preset)
- **Audio Codec**: AAC 192kbps (per track) / 320kbps (final)
- **Grain**: Applied to all tracks for texture
- **Fades**: 1.5-second fade in/out on every track

### Performance
- **CPU Usage**: 100% on one core per track
- **RAM Usage**: ~400-800 MB per track
- **Disk Space**:
  - Individual tracks: 400-600 MB each (~10 GB total)
  - Final composite: ~3-4 GB
  - Frames (temp): ~50 GB (auto-cleaned per track)
- **Estimated Time**:
  - Short tracks (60s): ~6 minutes
  - Medium tracks (120s): ~12 minutes
  - Long tracks (230s): ~23 minutes
  - **Total**: 4-6 hours for all 20

### Output Structure
```
visuals_phase1/
├── audio_segments/          # Extracted audio per track
├── frames/                  # PNG frames (cleaned after encoding)
├── renders/                 # Final track videos (.mp4)
└── metadata/                # JSON metadata

visuals_all_tracks/
├── audio_segments/
├── frames/
├── renders/
└── metadata/

PA-PAGOJE_Festival_Mix_FINAL_VIDEO.mp4  # ← Final 45-minute video
render_queue_*.log                        # Detailed render log
render_queue.json                         # Queue state (for resume)
```

---

## 🎬 Visual Features

### Smooth Transitions
All tracks have 1.5s fade in/out for smooth crossfading in final composite.

### Audio Reactivity
Every visual responds to:
- **Amplitude**: Particle size, glow intensity, motion speed
- **BPM**: Rhythm of animations and pulses
- **Energy**: Progressive build throughout each track

### Grain Texture
All tracks include film grain (strength 0.02-0.04) for cohesive analog feel.

### Color Progression
- **Warm-up**: Cool blues, purples, teals
- **Build 1**: Warming greens, yellows
- **Peak 1**: Hot reds, oranges, neons
- **Breather**: Natural greens, browns
- **Build 2**: Cool tension blues, purples
- **Peak 2**: Maximum contrast primaries, white strobes

---

## 🔧 Troubleshooting

### If a track fails:
1. Check the log file: `render_queue_*.log`
2. Look for the error message for that track
3. Re-render just that track:
   ```bash
   python3 render_queue_manager.py --tracks 07
   ```

### If you run out of disk space:
Individual track frames are cleaned after encoding. If you need more space:
```bash
# Remove old frame directories
rm -rf visuals_*/frames/*
```

### If rendering is too slow:
You can reduce quality in the generator files:
- Change `"-preset", "slow"` to `"-preset", "medium"` (line ~730)
- Change `"-crf", "18"` to `"-crf", "20"` (slightly smaller files)

### If you want to preview:
After rendering a few tracks, you can preview them:
```bash
# Play track 4
ffplay visuals_all_tracks/renders/04_*.mp4

# Or create a quick composite of completed tracks
python3 create_final_composite.py --force
```

---

## 📝 Notes

- **Track 09 (Parrot)** is already rendered as `parrot_visual_full.mp4` from earlier testing
- The queue manager will automatically find and use it
- All visual designs follow `VISUAL_PLAN_COMPLETE.md` specifications
- Transitions are smooth thanks to fade in/out on each track
- Final composite will be perfectly synced to `PA-PAGOJE_Festival_Mix2.wav`

---

## ⏱️ Estimated Timeline

If you start tonight:

| Time    | Status                              |
|---------|-------------------------------------|
| 00:00   | Start rendering                     |
| 01:30   | ~6 tracks done (Phase 1-2)          |
| 03:00   | ~12 tracks done (halfway!)          |
| 04:30   | ~18 tracks done (almost there!)     |
| 05:30   | All 20 tracks complete              |
| 05:35   | Final composite rendering           |
| 05:40   | ✓ DONE! 45-minute video ready       |

Wake up to a complete festival visual! 🎉

---

## 🎥 Final Result

You'll have:
- **20 individual track videos** (each synced to its audio segment)
- **1 master composite video** (45 minutes, full quality)
- **Full audio-visual sync** with the PA-PAGOJE_Festival_Mix2.wav
- **Stage-ready visuals** at 1280x720 HD

Ready for projection! 🚀
