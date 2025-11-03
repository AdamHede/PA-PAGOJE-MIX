#!/usr/bin/env python3
"""
PA-PAGØJE Render Queue Manager
Orchestrates overnight batch rendering of all 20 tracks with progress tracking
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class RenderJob:
    track_id: str
    script: str
    estimated_minutes: int
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None


class QueueManager:
    def __init__(self, queue_file: str = "render_queue.json"):
        self.queue_file = queue_file
        self.jobs: List[RenderJob] = []
        self.log_file = f"render_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(self, message: str):
        """Log to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def create_queue(self):
        """Define all rendering jobs"""
        # Phase 1 tracks (already have script)
        phase1_tracks = [
            ("01", "generate_phase1_visuals.py", 15),  # Se den lille kattekilling
            ("02", "generate_phase1_visuals.py", 18),  # I en kælder
            ("03", "generate_phase1_visuals.py", 13),  # Hundred' mus
        ]

        # Remaining tracks (new script)
        remaining_tracks = [
            ("04", "generate_remaining_visuals.py", 19),  # Spørge Jørgen
            ("05", "generate_remaining_visuals.py", 6),   # Der sad to katte
            ("06", "generate_remaining_visuals.py", 16),  # Mariehønen
            ("07", "generate_remaining_visuals.py", 11),  # Lille sky
            ("08", "generate_remaining_visuals.py", 6),   # Hønsefødder
            ("09", "generate_remaining_visuals.py", 11),  # Jeg er en papegøje (PARROT)
            ("10", "generate_remaining_visuals.py", 11),  # Op lille Hans
            ("11", "generate_remaining_visuals.py", 14),  # Oles nye autobil
            ("12", "generate_remaining_visuals.py", 23),  # Jeg gik mig over
            ("13", "generate_remaining_visuals.py", 22),  # I østen stiger
            ("14", "generate_remaining_visuals.py", 13),  # I skovens dybe
            ("15", "generate_remaining_visuals.py", 21),  # Hist, hvor vejen
            ("16", "generate_remaining_visuals.py", 17),  # Fra Engeland
            ("17", "generate_remaining_visuals.py", 18),  # Tre små fisk
            ("18", "generate_remaining_visuals.py", 6),   # Tre små kinesere
            ("19", "generate_remaining_visuals.py", 18),  # Ti små cyklister
            ("20", "generate_remaining_visuals.py", 14),  # Se dig for
        ]

        for track_id, script, est_min in phase1_tracks + remaining_tracks:
            self.jobs.append(RenderJob(track_id, script, est_min))

        self.log(f"Created queue with {len(self.jobs)} jobs")
        total_est = sum(job.estimated_minutes for job in self.jobs)
        self.log(f"Total estimated time: {total_est // 60}h {total_est % 60}m")

    def save_queue(self):
        """Save queue state to JSON"""
        data = {
            "jobs": [
                {
                    "track_id": job.track_id,
                    "script": job.script,
                    "estimated_minutes": job.estimated_minutes,
                    "status": job.status,
                    "start_time": job.start_time,
                    "end_time": job.end_time,
                    "error": job.error,
                }
                for job in self.jobs
            ],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.queue_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_queue(self):
        """Load queue state from JSON"""
        if not os.path.exists(self.queue_file):
            return False

        with open(self.queue_file, "r") as f:
            data = json.load(f)

        self.jobs = []
        for job_data in data["jobs"]:
            self.jobs.append(
                RenderJob(
                    track_id=job_data["track_id"],
                    script=job_data["script"],
                    estimated_minutes=job_data["estimated_minutes"],
                    status=job_data["status"],
                    start_time=job_data.get("start_time"),
                    end_time=job_data.get("end_time"),
                    error=job_data.get("error"),
                )
            )

        self.log(f"Loaded queue with {len(self.jobs)} jobs")
        return True

    def get_next_job(self) -> Optional[RenderJob]:
        """Get next pending job"""
        for job in self.jobs:
            if job.status == "pending":
                return job
        return None

    def run_job(self, job: RenderJob) -> bool:
        """Execute a single rendering job"""
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        self.save_queue()

        self.log("=" * 100)
        self.log(f"Starting Track {job.track_id}")
        self.log(f"Script: {job.script}")
        self.log(f"Estimated time: {job.estimated_minutes} minutes")
        self.log("=" * 100)

        try:
            cmd = ["python3", job.script, "--tracks", job.track_id]

            # Run with live output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Stream output to log
            for line in process.stdout:
                print(line, end="")
                with open(self.log_file, "a") as f:
                    f.write(line)

            process.wait()

            if process.returncode == 0:
                job.status = "completed"
                job.end_time = datetime.now().isoformat()
                self.log(f"✓ Track {job.track_id} completed successfully")
                return True
            else:
                job.status = "failed"
                job.error = f"Exit code {process.returncode}"
                job.end_time = datetime.now().isoformat()
                self.log(f"✗ Track {job.track_id} failed with exit code {process.returncode}")
                return False

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = datetime.now().isoformat()
            self.log(f"✗ Track {job.track_id} failed with exception: {e}")
            return False

        finally:
            self.save_queue()

    def print_status(self):
        """Print current queue status"""
        completed = sum(1 for job in self.jobs if job.status == "completed")
        failed = sum(1 for job in self.jobs if job.status == "failed")
        running = sum(1 for job in self.jobs if job.status == "running")
        pending = sum(1 for job in self.jobs if job.status == "pending")

        self.log("\n" + "=" * 100)
        self.log("QUEUE STATUS")
        self.log("=" * 100)
        self.log(f"Completed: {completed}/{len(self.jobs)}")
        self.log(f"Failed:    {failed}/{len(self.jobs)}")
        self.log(f"Running:   {running}/{len(self.jobs)}")
        self.log(f"Pending:   {pending}/{len(self.jobs)}")
        self.log("=" * 100)

        if failed > 0:
            self.log("\nFailed tracks:")
            for job in self.jobs:
                if job.status == "failed":
                    self.log(f"  Track {job.track_id}: {job.error}")

    def run_all(self, resume: bool = False):
        """Run all jobs in queue"""
        if resume and self.load_queue():
            self.log("Resuming from saved queue state")
        else:
            self.create_queue()
            self.save_queue()

        start_time = time.time()

        while True:
            next_job = self.get_next_job()
            if not next_job:
                break

            self.run_job(next_job)

            # Brief pause between jobs
            time.sleep(2)

        end_time = time.time()
        elapsed = (end_time - start_time) / 60

        self.log("\n" + "=" * 100)
        self.log("ALL JOBS COMPLETED")
        self.log(f"Total time: {int(elapsed // 60)}h {int(elapsed % 60)}m")
        self.log("=" * 100)

        self.print_status()

        # Check outputs
        self.verify_outputs()

    def verify_outputs(self):
        """Verify all output videos exist"""
        self.log("\n" + "=" * 100)
        self.log("VERIFYING OUTPUTS")
        self.log("=" * 100)

        output_dirs = ["visuals_phase1/renders", "visuals_all_tracks/renders"]

        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                videos = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
                total_size = sum(
                    os.path.getsize(os.path.join(output_dir, f)) for f in videos
                )
                self.log(f"\n{output_dir}:")
                self.log(f"  Videos: {len(videos)}")
                self.log(f"  Total size: {total_size / 1e9:.2f} GB")

                for video in sorted(videos):
                    size_mb = os.path.getsize(os.path.join(output_dir, video)) / 1e6
                    self.log(f"    {video}: {size_mb:.1f} MB")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render queue manager for PA-PAGØJE visuals")
    parser.add_argument("--resume", action="store_true", help="Resume from saved queue state")
    parser.add_argument("--status", action="store_true", help="Show queue status only")
    parser.add_argument("--tracks", nargs="+", help="Only render specific tracks")
    args = parser.parse_args()

    manager = QueueManager()

    if args.status:
        if manager.load_queue():
            manager.print_status()
        else:
            print("No queue file found")
        return

    if args.tracks:
        # Custom track list
        manager.create_queue()
        manager.jobs = [job for job in manager.jobs if job.track_id in args.tracks]
        manager.log(f"Filtered to {len(manager.jobs)} tracks: {', '.join(args.tracks)}")

    manager.run_all(resume=args.resume)


if __name__ == "__main__":
    main()
