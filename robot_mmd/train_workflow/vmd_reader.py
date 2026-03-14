"""
VMD 文件解析器 - 读取 MMD 骨骼动画数据
支持按帧顺序输出骨骼数据，并导出为 CSV 文件
"""
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


def export_to_csv(
    bones_data: list[dict],
    output_path: str,
    *,
    sorted_by_frame: bool = True,
) -> str:
    """
    将骨骼数据导出为 CSV 文件。

    格式说明：每行一条骨骼记录，列包括：
    - frame: 帧号
    - bone: 骨骼名称
    - pos_x, pos_y, pos_z: 位置 (mm)
    - quat_x, quat_y, quat_z, quat_w: 旋转四元数

    这种「长格式」便于按帧筛选、按骨骼筛选，也适合后续分析或导入其他工具。
    """
    data = get_frames_sorted(bones_data) if sorted_by_frame else bones_data

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


def read_and_export(vmd_path: str, output_path: str | None = None) -> str:
    """
    读取 VMD 文件并导出为 CSV。若未指定 output_path，则使用与 vmd 同名的 .csv 文件。
    """
    vmd_path = Path(vmd_path)
    if output_path is None:
        output_path = str(vmd_path.with_suffix('.csv'))

    bones_data = read_vmd_bones(str(vmd_path))
    return export_to_csv(bones_data, output_path)


# 使用示例
if __name__ == "__main__":
    vmd_file = Path("I:/robot_isaac/robot_mmd/media/you_are_important.vmd")
    # output_csv 与 vmd_file 同名
    output_csv = str(vmd_file.with_suffix('.csv'))

    # 方式1: 读取并按帧迭代
    bones_data = read_vmd_bones(vmd_file)
    for frame_idx, frame_bones in iter_frames(bones_data):
        if frame_idx <= 2:  # 仅打印前几帧示例
            print(f"帧 {frame_idx}: {len(frame_bones)} 个骨骼")
            for b in frame_bones[:3]:
                print(f"  - {b['bone']}: pos={b['position']}, quat={b['quaternion']}")

    # 方式2: 导出为 CSV
    out = export_to_csv(bones_data, output_csv)
    print(f"\n已导出到: {out}")

    # 方式3: 一步完成
    # read_and_export(vmd_file)
