#!/bin/bash
cd "$(dirname "$0")"
echo "Running run_live_face_detector..."
wine "run_live_face_detector" || ./"run_live_face_detector" "$@"
