"""Dispatch the supported project workflows without loading Isaac Sim."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
G1 = Path("source/train_workflow")
PMX = Path("source/pmx_prototype")

# Register public entry points explicitly. Workers, test runners and archived
# source files may also be executable, but are not public modes.
WORKFLOWS = {
    "g1-replay": (G1 / "g1_vmd_0_replay.py", "G1 回放、映射调试和 H5 导出"),
    "g1-train": (G1 / "g1_vmd_1_train.py", "G1 PPO 训练和续训"),
    "g1-eval": (G1 / "g1_vmd_2_eval.py", "G1 checkpoint 验证、播放和验收录像"),
    "g1-pipeline": (G1 / "g1_vmd_pipeline.py", "G1 VMD 转换、H5 和视频流水线"),
    "g1-motion": (G1 / "scripts/g1_ground_motion.py", "G1 贴地 IK、避碰平滑和动作验收"),
    "g1-diagnostics": (G1 / "scripts/g1_diagnostics.py", "G1 跳跃和转向离线诊断"),
    "g1-prepare-motion": (G1 / "scripts/compare_dance_auto.py", "从 VMD 生成并验收 G1 训练参考 H5，支持三列复核录像"),
    "pmx-convert": (PMX / "convert_pmx.py", "将 PMX 角色转换为 USD"),
    "pmx-bake": (PMX / "load_vmd.py", "将 VMD 动作烘焙到 USD 角色"),
    "pmx-view": (PMX / "view_character.py", "查看 USD 角色资产"),
    "pmx-preview": (PMX / "tools/preview_dance.py", "按角色和舞蹈预览 PMX"),
    "pmx-physx": (PMX / "physx_dance.py", "预览或录制原生 PhysX 发丝和裙摆"),
    "pmx-realtime": (PMX / "realtime_dance.py", "实时播放 XPBD 舞蹈"),
    "pmx-record": (PMX / "record_physx_video.py", "录制 PMX PhysX 舞蹈视频"),
    "pmx-locomotion": (PMX / "train_anon_locomotion.py", "爱音策略训练、评估和参考动作导出"),
    "pmx-policy-play": (PMX / "play_anon_policy.py", "播放爱音运动策略"),
    "pmx-policy-inspect": (PMX / "inspect_anon_policy.py", "交互检查爱音运动策略"),
    "pmx-experiments": (PMX / "experiments/__main__.py", "列出或复现已登记的 PMX 实验"),
    "compare-g1-pmx": (PMX / "play_g1_anon_comparison.py", "用现有动作播放或录制 G1 与 PMX 对比"),
}

WORKFLOW_ALIASES = {"compare-dance": "g1-prepare-motion"}

# Tools in these groups use their original script names without the .py suffix.
TOOL_GROUPS = {
    "g1-tools": ("G1 转换、重定向、网格检查、训练诊断和录像工具", G1 / "scripts", (
        "analyze_auto_jump",
        "analyze_jump_diagnostics",
        "audit_full_meshes",
        "benchmark_auto_checkpoints",
        "calibrate_quality_profile",
        "compare_tracking_diagnostics",
        "compare_validation_video",
        "compose_phase_checkpoint",
        "diagnose_auto_tail",
        "diagnose_turn",
        "edit_csv_root_z",
        "h5_motion_start_weight",
        "plot_motion_start_weight",
        "prepare_jump_checkpoint",
        "recheck_motion_feedback",
        "record_checkpoint_comparison",
        "record_h5_without_collision",
        "regress_ground_motion",
        "render_validation_video",
        "repair_gesture_collision",
        "repair_h5_self_collision",
        "retarget_lower_body",
        "retarget_upper_body",
        "summarize_auto_resume",
        "train_turn_smoothing",
        "validate_parallel_checkpoint",
        "vmd_2_csv",
    )),
    "pmx-motion": ("PMX 动作库、重定向和次级物理烘焙", PMX, (
        "motion/adapt_garment_reference",
        "motion/balance_full_motion",
        "motion/balance_running_reference",
        "motion/build_continuous_walk",
        "motion/build_continuous_wrists",
        "motion/build_full_motion",
        "motion/build_motion_bank",
        "motion/build_running_wrists",
        "motion/build_turn_motion",
        "motion/build_unified_walk_run",
        "motion/export_full_motion_reference",
        "motion/mirror_full_motion",
        "motion/mocap_walk_reference",
        "motion/retarget_100style",
        "motion/smooth_full_motion",
        "runtime/secondary_motion",
    )),
    "pmx-diagnostics": ("PMX 缓存、物理、步态和策略验收", PMX / "diagnostics", (
        "audit_anon_gait",
        "audit_gait_controller",
        "audit_running_arms",
        "audit_unified_rollout",
        "character_cache",
        "check_live_music",
        "check_physx_quality",
        "check_realtime",
        "compare_amp_coordination",
        "compare_stance_slip",
        "compose_gait_diagnostic",
        "diagnose_pivot_failure",
        "find_mocap_cycles",
        "locomotion_tools",
        "measure_contiguous_variation",
        "measure_gait_slip",
        "probe_lateral_symmetry",
        "probe_unified_nominal",
        "record_gait_diagnostic",
        "summarize_anon_training_time",
        "validate_anon_runtime",
        "validate_full_motion",
        "validate_imitation_heading",
        "validate_imitation_idle",
        "validate_physical_character",
        "validate_pivot_training",
        "verify_articulated_model",
    )),
    "pmx-tools": ("PMX 资产准备、录像、蒸馏、打包和数据下载", PMX, (
        "prepare_mutsumi_physx",
        "record_anon_locomotion",
        "record_cached_secondary",
        "record_g1_anon_comparison",
        "record_mutsumi_references",
        "tools/collect_unified_teachers",
        "tools/compare_locomotion_video",
        "tools/distill_unified_actor",
        "tools/download_100style",
        "tools/package_anon_policy",
        "tools/package_unified_walk_run",
        "tools/record_look_comparison",
        "training/build_anon_v4",
    )),
}

# These standalone tools use built-in inputs and have no argument parser.
# Handle help here so asking for it cannot run the underlying job.
NO_ARGUMENT_TOOLS = {
    "build_anon_v4",
    "build_continuous_wrists",
    "build_running_wrists",
    "check_live_music",
    "check_physx_quality",
    "check_realtime",
    "download_100style",
    "find_mocap_cycles",
    "mocap_walk_reference",
    "retarget_100style",
    "validate_full_motion",
    "validate_imitation_heading",
    "validate_imitation_idle",
    "verify_articulated_model",
}


def build_parser():
    descriptions = {name: description for name, (_, description) in WORKFLOWS.items()}
    descriptions.update({name: group[0] for name, group in TOOL_GROUPS.items()})
    parser = argparse.ArgumentParser(
        description="Isaac-Robot-MMD 正式工作流入口。",
        usage="%(prog)s --mode MODE [原脚本参数 ...]",
        epilog="可用模式：\n" + "\n".join(
            f"  {name:22} {description}" for name, description in sorted(descriptions.items())
        ) + "\n\n运行 %(prog)s --mode MODE --help 查看该模式的用法。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--mode", required=True, choices=(*descriptions, *WORKFLOW_ALIASES), metavar="MODE")
    return parser


def select_tool(mode, arguments):
    description, directory, registered = TOOL_GROUPS[mode]
    tools = {Path(name).name: directory / (name + ".py") for name in registered}
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} --mode {mode}",
        description=description,
        usage="%(prog)s TOOL [原脚本参数 ...]",
        epilog="可用工具：\n" + "\n".join(f"  {name}" for name in sorted(tools)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("tool", choices=tools, metavar="TOOL")
    if not arguments:
        parser.print_help()
        return None
    options = parser.parse_args(arguments[:1])
    return tools[options.tool]


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 0

    # Parse only the leading selector. The tail can contain its own --mode,
    # --help or --, and belongs entirely to the selected script.
    selector_size = 2 if arguments[0] == "--mode" else 1
    mode = parser.parse_args(arguments[:selector_size]).mode
    mode = WORKFLOW_ALIASES.get(mode, mode)
    remaining = arguments[selector_size:]
    if mode in TOOL_GROUPS:
        entry = select_tool(mode, remaining)
        if entry is None:
            return 0
        tool = remaining[0]
        remaining = remaining[1:]
        if tool in NO_ARGUMENT_TOOLS and remaining:
            if remaining in (["--help"], ["-h"]):
                print(f"用法：{parser.prog} --mode {mode} {tool}")
                print(f"{entry.as_posix()} 使用内置输入和输出路径，无命令行参数。")
                return 0
            parser.error(f"{mode} {tool} 不接收命令行参数")
    else:
        entry, _ = WORKFLOWS[mode]

    script = ROOT / entry
    if not script.is_file():
        parser.error(f"入口文件不存在：{script}")
    # Keep the caller's working directory, environment and console. Ctrl+C
    # reaches the child directly; wait for its own shutdown handler to finish.
    try:
        with subprocess.Popen([sys.executable, str(script), *remaining]) as child:
            try:
                return child.wait()
            except KeyboardInterrupt:
                return child.wait()
    except KeyboardInterrupt:
        return 130
    except OSError as error:
        print(f"{parser.prog}: 无法启动 {entry.as_posix()}：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
