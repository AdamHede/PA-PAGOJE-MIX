#!/bin/bash
# PA-PAGØJE Overnight Render Starter
# Run this script to start the full overnight rendering process

set -e

echo "=========================================="
echo "PA-PAGØJE OVERNIGHT RENDERING"
echo "=========================================="
echo ""
echo "This will render all 20 tracks."
echo "Estimated time: 4-6 hours"
echo ""
echo "The process will:"
echo "  1. Render all track visuals sequentially"
echo "  2. Save progress (can resume if interrupted)"
echo "  3. Create final 45-minute composite video"
echo ""
read -p "Press ENTER to start, or Ctrl+C to cancel..."

# Make scripts executable
chmod +x generate_phase1_visuals.py
chmod +x generate_remaining_visuals.py
chmod +x render_queue_manager.py
chmod +x create_final_composite.py

# Start queue manager
echo ""
echo "Starting render queue..."
echo "Log file: render_queue_*.log"
echo ""

python3 render_queue_manager.py

# Check if all succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ALL TRACKS RENDERED!"
    echo "=========================================="
    echo ""
    read -p "Create final composite video now? (y/n): " response

    if [ "$response" = "y" ]; then
        echo ""
        echo "Creating final composite..."
        python3 create_final_composite.py

        if [ $? -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "✓✓✓ COMPLETE! ✓✓✓"
            echo "=========================================="
            echo ""
            echo "Your final video is ready:"
            ls -lh PA-PAGOJE_Festival_Mix_FINAL_VIDEO.mp4
            echo ""
            echo "Total render outputs:"
            du -sh visuals_*/renders
        fi
    else
        echo ""
        echo "You can create the composite later with:"
        echo "  python3 create_final_composite.py"
    fi
else
    echo ""
    echo "Some tracks failed. Check the log file for details."
    echo "You can resume with:"
    echo "  python3 render_queue_manager.py --resume"
fi
