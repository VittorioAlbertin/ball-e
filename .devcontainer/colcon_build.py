#!/usr/bin/env python3
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

# === CONFIGURATION ===
# Map group names to python interpreter
GROUP_INTERPRETER = {
    "system": "/usr/bin/python3",
    "ml": "/home/ubuntu/ml_env/bin/python3"
}

# Workspace path
WS_PATH = Path(__file__).parent.parent / "ros2_ws"

# === STEP 1: Scan packages ===
packages_by_group = {}
for xml_file in (WS_PATH / "src").rglob("package.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    group_tag = root.find(".//group_depend")
    group = group_tag.text if group_tag is not None else "system"
    packages_by_group.setdefault(group, []).append(xml_file.parent.name)

# === STEP 2: Build each group with correct interpreter ===
for group, packages in packages_by_group.items():
    interpreter = GROUP_INTERPRETER.get(group, "/usr/bin/python3")
    print(f"Building group '{group}' with interpreter {interpreter}: {packages}")

    # Run colcon build
    subprocess.run(
        f"{interpreter} -m colcon build "
        f"--merge-install "
        f"--packages-select {' '.join(packages)} "
        f"--cmake-args -DPYTHON_EXECUTABLE={interpreter}",
        cwd=WS_PATH,
        shell=True,
        check=True,
    )