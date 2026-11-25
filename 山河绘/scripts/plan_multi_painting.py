"""CLI for subject-aware offline design. No robot imports or execution flags."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_painting import PLANNERS, build_plan, save_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description='多画种离线笔画规划；只生成 JSON 与路径预览')
    parser.add_argument('--subject', required=True, choices=tuple(PLANNERS))
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='输出计划 JSON')
    parser.add_argument('--preview', required=True, type=Path, help='输出路径 PNG')
    parser.add_argument('--work-size', type=int, default=1000)
    args = parser.parse_args(argv)
    if not 64 <= args.work_size <= 2400:
        parser.error('--work-size must be between 64 and 2400')
    plan, work = build_plan(args.image, args.subject, args.work_size)
    save_outputs(plan, work, args.out, args.preview)
    counts = {name: module['stroke_count'] for name, module in plan['modules'].items()}
    print(f"{args.subject}: {counts}; status={plan['status']}; ready_for_execution=false")
    print(args.out.resolve())
    print(args.preview.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
