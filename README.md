note to self:
- from vscode --> `>Dev Containers: Rebuild and Reopen in container`
even if the venv is not active, its sourced as the main python for colcon --> error when `colcon build`.
- (if you tried to colcon build and got an error, remeber to clear your mess --> `cd ros2_ws && rm -rf install build log`)
- deactivate venv with `source ~/ros2_ws/install/setup.bash`
- run `colcon build`
- for each node that needs to run in the venv:
    `~/ros2_ws/install/[pkg_name]/lib/[pkg_name]/[node_name]`
    change the first line from `#!/usr/bin/python3` to `#!/home/ubuntu/ml_env/bin/python3`

- to activate the venv (if needed)
    `source ~/ml_env/bin/activate`