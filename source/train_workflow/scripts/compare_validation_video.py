"""Add a synchronized PMX panel to an already rendered acceptance video (CPU)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.validation_reference import (
    add_reference_video_arguments, reference_video_from_args, compose_reference_video)
from source.train_workflow.utils.motion.resolve import resolve_dance_h5_by_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True, help="Existing validation MP4 with its JSON sidecar")
    parser.add_argument("--dance", help="Dance name for the configured default PMX video")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--output-height", type=int, help="Even output height; defaults to validation video height")
    add_reference_video_arguments(parser)
    args = parser.parse_args()
    motion = resolve_dance_h5_by_name(args.dance, compile_from_csv=False)[0] if args.dance else None
    reference = reference_video_from_args(args, motion)
    if reference is None:
        parser.error("No reference MP4 configured; supply --validation_reference_video or configure --dance")
    # Import here to keep the resolver and compositor themselves simulator/torch-free.
    from source.train_workflow.utils.validation_recording import find_ffmpeg
    encoder = str(args.ffmpeg) if args.ffmpeg else find_ffmpeg()
    if encoder is None:
        parser.error("ffmpeg unavailable; install imageio-ffmpeg or supply --ffmpeg")
    compose_reference_video(args.validation, reference, ffmpeg=encoder, output=args.output, output_height=args.output_height)


if __name__ == "__main__":
    main()
