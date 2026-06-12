"""
VMD/VPD 到 CSV 的转换工具。

功能概览：
1) 解析 VMD 二进制骨骼关键帧；
2) 解析 VPD 单帧姿态文本；
3) 导出四元数列格式的 CSV（frame, bone, pos_*, quat_*）。
"""
import argparse
import re
import struct
import csv
from pathlib import Path
from typing import Iterator


def _read_text_with_fallback(file_path: str, encodings: tuple[str, ...]) -> list[str]:
    """按候选编码读取文本文件，返回全部行。"""
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("", b"", 0, 0, f"无法用 {'/'.join(encodings)} 解码文件")


def read_vmd_bones(file_path: str) -> list[dict]:
    """读取 VMD 文件中的骨骼数据，返回原始记录列表"""
    with open(file_path, 'rb') as f:
        # 1. 跳过文件头 (30 bytes) 和 模型名 (20 bytes)
        f.seek(50)

        # 2. 读取总帧数 (4 bytes, unsigned int)
        count_data = f.read(4)
        if not count_data:
            return []
        frame_count = struct.unpack('<I', count_data)[0]

        bones_data = []
        for _ in range(frame_count):
            # 每帧包含: 骨骼名(15), 帧序号(4), 坐标(12), 旋转(16), 插值(64)
            # 总共 111 字节
            data = f.read(111)
            if len(data) < 111:
                break

            # 解析骨骼名 (Shift-JIS 编码)
            name = data[:15].split(b'\x00')[0].decode('shift_jis', errors='ignore')

            # 解析帧序号
            frame_idx = struct.unpack('<I', data[15:19])[0]

            # 解析坐标 (x, y, z) - float32
            pos = struct.unpack('<fff', data[19:31])

            # 解析旋转 (qx, qy, qz, qw) - float32
            rot = struct.unpack('<ffff', data[31:47])

            bones_data.append({
                "bone": name,
                "frame": frame_idx,
                "position": pos,
                "quaternion": rot
            })

        return bones_data


def read_vpd_pose(file_path: str) -> list[dict]:
    """
    读取 VPD 文件中的单帧姿态数据，返回与 VMD 兼容的骨骼记录列表。
    所有骨骼的 frame 均为 0，可直接用于 export_to_csv、iter_frames 等函数。
    VPD 为 Shift-JIS 编码的文本格式。
    """
    lines = _read_text_with_fallback(file_path, ("shift_jis", "cp932", "utf-8"))

    bones_data: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = re.match(r"Bone\d+\{(.+)", line.strip())
        if not m:
            i += 1
            continue

        bone_name = m.group(1).strip()

        # 下一行: trans x,y,z
        i += 1
        if i >= len(lines):
            break
        pos_str = lines[i].split(";")[0].strip()
        pos = tuple(float(x.strip()) for x in pos_str.split(","))

        # 再下一行: quaternion x,y,z,w
        i += 1
        if i >= len(lines):
            break
        quat_str = lines[i].split(";")[0].strip()
        quat = tuple(float(x.strip()) for x in quat_str.split(","))

        bones_data.append({
            "bone": bone_name,
            "frame": 0,
            "position": pos,
            "quaternion": quat,
        })

        i += 1  # 跳过 }

    return bones_data


def get_frames_sorted(bones_data: list[dict]) -> list[dict]:
    """按帧数顺序排序骨骼数据"""
    return sorted(bones_data, key=lambda x: (x["frame"], x["bone"]))


def iter_frames(bones_data: list[dict]) -> Iterator[tuple[int, list[dict]]]:
    """
    按帧数顺序迭代，每帧返回 (frame_idx, [该帧的所有骨骼数据])
    """
    sorted_data = get_frames_sorted(bones_data)
    if not sorted_data:
        return

    current_frame = sorted_data[0]["frame"]
    frame_bones = []
    for record in sorted_data:
        if record["frame"] != current_frame:
            yield current_frame, frame_bones
            current_frame = record["frame"]
            frame_bones = []
        frame_bones.append(record)
    yield current_frame, frame_bones


# 导出 CSV 时仅保留的骨骼（上半身、下半身、头、腰、中心、手臂、手指、足）
_EXPORT_BONE_WHITELIST: set[str] = {
    "上半身", "上半身2", "下半身", "下半身先",
    "頭", "頭先", "首", "腰", "腰飾り",
    "センター", "センター先", "グルーブ",
    "右肩", "左肩", "右腕", "左腕", "右ひじ", "左ひじ",
    "右手首", "左手首", "右手先", "左手先",
    "右足", "左足", "右ひざ", "左ひざ", "右足首", "左足首",
}
_EXPORT_ARM_PREFIXES: tuple[str, ...] = (
    "右腕捩", "左腕捩", "右手捩", "左手捩",  # 腕捩/手捩 及其 1,2,3,先 等变体
)
_EXPORT_FINGER_PREFIXES: tuple[str, ...] = (
    "右親指", "左親指", "右人指", "左人指", "右人差指", "左人差指",
    "右中指", "左中指", "右薬指", "左薬指", "右小指", "左小指",
)


def _should_export_bone(bone_name: str) -> bool:
    """判断骨骼是否在导出白名单内"""
    if bone_name in _EXPORT_BONE_WHITELIST:
        return True
    return any(bone_name.startswith(p) for p in _EXPORT_ARM_PREFIXES + _EXPORT_FINGER_PREFIXES)


def export_to_csv(
    bones_data: list[dict],
    output_path: str,
    *,
    sorted_by_frame: bool = True,
    bone_filter: bool = True,
) -> str:
    """
    将骨骼数据导出为 CSV 文件。

    格式说明：每行一条骨骼记录，列包括：
    - frame: 帧号
    - bone: 骨骼名称
    - pos_x, pos_y, pos_z: 位置（与 MMD/PMX 一致浮点，与 VMD 二进制一致；非实际毫米）
    - quat_x, quat_y, quat_z, quat_w: 旋转四元数

    默认仅保留：上半身、下半身、头、颈、腰、中心、手臂、手指、足。
    若 bone_filter=False 则导出全部骨骼。
    """
    data = get_frames_sorted(bones_data) if sorted_by_frame else bones_data
    if bone_filter:
        data = [r for r in data if _should_export_bone(r["bone"])]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame', 'bone',
            'pos_x', 'pos_y', 'pos_z',
            'quat_x', 'quat_y', 'quat_z', 'quat_w'
        ])
        for record in data:
            pos = record["position"]
            quat = record["quaternion"]
            writer.writerow([
                record["frame"],
                record["bone"],
                f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                f"{quat[0]:.6f}", f"{quat[1]:.6f}", f"{quat[2]:.6f}", f"{quat[3]:.6f}",
            ])

    return output_path


def read_and_export(
    vmd_path: str,
    output_path: str | None = None,
    *,
    bone_filter: bool = True,
) -> str:
    """
    读取 VMD 文件并导出为 CSV。若未指定 output_path，则使用与 vmd 同名的 .csv 文件。
    bone_filter=True 时仅导出上半身、下半身、头、颈、腰、中心、手臂、手指、足。
    """
    return read_motion_and_export(vmd_path, output_path=output_path, bone_filter=bone_filter)


def read_vpd_and_export(
    vpd_path: str,
    output_path: str | None = None,
    *,
    bone_filter: bool = True,
) -> str:
    """
    读取 VPD 文件并导出为 CSV。若未指定 output_path，则使用与 vpd 同名的 .csv 文件。
    bone_filter=True 时仅导出上半身、下半身、头、颈、腰、中心、手臂、手指、足。
    """
    return read_motion_and_export(vpd_path, output_path=output_path, bone_filter=bone_filter)


def read_motion(input_path: str) -> list[dict]:
    """
    统一读取入口：根据后缀自动分发 VMD / VPD 读取逻辑。
    支持 .vmd / .vpd（大小写不敏感）。
    """
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".vmd":
        return read_vmd_bones(str(path))
    if suffix == ".vpd":
        return read_vpd_pose(str(path))
    raise ValueError(f"不支持的文件类型: {path.suffix}，仅支持 .vmd / .vpd")


def read_motion_and_export(
    input_path: str,
    output_path: str | None = None,
    *,
    bone_filter: bool = True,
    hand_suffix: bool = False,
) -> str:
    """
    统一转换入口：VMD/VPD -> CSV。
    - 根据输入后缀自动选择解析器
    - 未提供 output_path 时，默认输出为同名 .csv
    """
    path = Path(input_path)
    if output_path is None:
        if hand_suffix:
            output_path = str(path.with_name(f"{path.stem}_hand.csv"))
        else:
            output_path = str(path.with_suffix(".csv"))
    bones_data = read_motion(str(path))
    exported = export_to_csv(bones_data, output_path, bone_filter=bone_filter)
    if hand_suffix:
        finger_count = sum(1 for r in bones_data if _should_export_bone(r["bone"]) and any(
            r["bone"].startswith(p) for p in _EXPORT_FINGER_PREFIXES
        ))
        print(f"[INFO] hand 导出模式: {exported} (finger_records={finger_count})")
    return exported


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VMD/VPD -> CSV")
    p.add_argument("input_path", type=str, help="输入 .vmd 或 .vpd")
    p.add_argument("-o", "--output", type=str, default=None, help="输出 CSV 路径")
    p.add_argument(
        "--no-bone-filter",
        action="store_true",
        help="关闭骨骼白名单过滤，导出全部骨骼",
    )
    p.add_argument(
        "--with-hand",
        action="store_true",
        help="手指导出模式：默认输出使用 *_hand.csv",
    )
    return p


# 使用示例
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    output = args.output
    if output is not None and args.with_hand and not str(output).lower().endswith("_hand.csv"):
        print("[WARN] --with-hand 启用，但 --output 未使用 _hand.csv 后缀")
    read_motion_and_export(
        args.input_path,
        output_path=output,
        bone_filter=not bool(args.no_bone_filter),
        hand_suffix=bool(args.with_hand),
    )

