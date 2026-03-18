"""
VMD 文件解析器 - 读取 MMD 骨骼动画数据
支持按帧顺序输出骨骼数据，并导出为 CSV 文件
支持将四元数 CSV 转换为 roll/pitch/yaw 欧拉角 CSV
"""
import math
import struct
import csv
from pathlib import Path
from typing import Iterator


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


def _quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """四元数转欧拉角 (XYZ 顺序)，返回 (roll, pitch, yaw) 弧度"""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-10:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = max(-1, min(1, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def convert_csv_to_euler(csv_path: str) -> str:
    """
    将四元数格式的骨骼 CSV 转换为 roll/pitch/yaw 欧拉角格式。
    输入 CSV 需包含: frame, bone, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w
    输出文件命名为: 原名_euler.csv，列变为: frame, bone, pos_x, pos_y, pos_z, roll, pitch, yaw
    欧拉角单位为弧度。
    """
    path = Path(csv_path)
    if path.suffix.lower() == ".vmd":
        path = path.with_suffix(".csv")
        if not path.exists():
            raise FileNotFoundError(
                f"convert_csv_to_euler 需要 CSV 文件。请先从 VMD 导出 CSV，或传入 .csv 路径，例如: {path}"
            )
        csv_path = str(path)
    out_path = path.parent / (path.stem + "_euler.csv")

    rows = None
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            with open(csv_path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        raise UnicodeDecodeError("", b"", 0, 0, "无法用 utf-8/cp932/shift_jis 解码 CSV")

    if not rows:
        with open(out_path, "w", encoding="utf-8", newline="") as out:
            out.write("frame,bone,pos_x,pos_y,pos_z,roll,pitch,yaw\n")
        return str(out_path)

    fieldnames = ["frame", "bone", "pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            qx = float(row["quat_x"])
            qy = float(row["quat_y"])
            qz = float(row["quat_z"])
            qw = float(row["quat_w"])
            roll, pitch, yaw = _quat_to_euler(qx, qy, qz, qw)
            writer.writerow({
                "frame": row["frame"],
                "bone": row["bone"],
                "pos_x": row["pos_x"],
                "pos_y": row["pos_y"],
                "pos_z": row["pos_z"],
                "roll": f"{roll:.6f}",
                "pitch": f"{pitch:.6f}",
                "yaw": f"{yaw:.6f}",
            })

    print(f"已生成欧拉角 CSV: {out_path}")
    return str(out_path)


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
    - pos_x, pos_y, pos_z: 位置 (mm)
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
    vmd_path = Path(vmd_path)
    if output_path is None:
        output_path = str(vmd_path.with_suffix('.csv'))

    bones_data = read_vmd_bones(str(vmd_path))
    return export_to_csv(bones_data, output_path, bone_filter=bone_filter)


# 使用示例
if __name__ == "__main__":
    import sys

    # vmd_file = Path("I:/robot_isaac/robot_mmd/media/333.vmd")
    # output_csv = str(vmd_file.with_suffix(".csv"))
    # bones_data = read_vmd_bones(vmd_file)
    # out = export_to_csv(bones_data, output_csv)
    # print(f"已导出到: {out}")

    convert_csv_to_euler("robot_mmd/media/333.csv")

