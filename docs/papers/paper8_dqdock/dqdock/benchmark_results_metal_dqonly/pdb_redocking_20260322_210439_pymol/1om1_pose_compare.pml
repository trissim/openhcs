reinitialize
load /home/ts/code/projects/openhcs-sequential/benchmark_results_metal_dqonly/pdb_redocking_20260322_210439_poses/1om1_receptor.pdb, receptor
load /home/ts/code/projects/openhcs-sequential/benchmark_results_metal_dqonly/pdb_redocking_20260322_210439_poses/1om1_native_ligand.pdb, native
hide everything, all
remove hydro
select pocket_wall, byres (receptor within 6.0 of native)
show sticks, native
color black, native
load /home/ts/code/projects/openhcs-sequential/benchmark_results_metal_dqonly/pdb_redocking_20260322_210439_poses/1om1_dq_dock_pose.pdb, dq_dock
show sticks, dq_dock
color tv_orange, dq_dock
show surface, pocket_wall
set transparency, 0.14, pocket_wall
set surface_color, lightblue, pocket_wall
show sticks, byres ((receptor within 4.0 of native) and sidechain)
color gray60, byres ((receptor within 4.0 of native) and sidechain)
set surface_quality, 1
set stick_radius, 0.22
set ray_opaque_background, on
set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set two_sided_lighting, on
set antialias, 2
bg_color white
set_name native, native_1om1
group poses, dq_dock
orient (pocket_wall or native_1om1 or dq_dock)
zoom (pocket_wall or native_1om1 or dq_dock), 2
python
from pymol import cmd
import numpy as np
view_dir = np.array([0.35194654, 0.66814376, 0.65552845], dtype=float)
view = np.array(cmd.get_view()[:9], dtype=float).reshape(3, 3)
view_dir_camera = view.T @ view_dir
yaw = -np.degrees(np.arctan2(view_dir_camera[0], view_dir_camera[2]))
cmd.turn('y', float(yaw))
view = np.array(cmd.get_view()[:9], dtype=float).reshape(3, 3)
view_dir_camera = view.T @ view_dir
pitch = np.degrees(np.arctan2(view_dir_camera[1], view_dir_camera[2]))
cmd.turn('x', float(pitch))
cmd.turn('z', 15.0)
cmd.zoom('(pocket_wall or native_1om1 or dq_dock)', 2.0)
python end
viewport 1600, 1200
png /home/ts/code/projects/openhcs-sequential/benchmark_results_metal_dqonly/pdb_redocking_20260322_210439_pymol/1om1_pose_compare_3d.png, width=1600, height=1200, dpi=200, ray=1
turn y, 90
zoom (pocket_wall or native_1om1 or dq_dock), 2
png /home/ts/code/projects/openhcs-sequential/benchmark_results_metal_dqonly/pdb_redocking_20260322_210439_pymol/1om1_pose_compare_3d_side.png, width=1600, height=1200, dpi=200, ray=1
save /home/ts/code/projects/openhcs-sequential/benchmark_results_metal_dqonly/pdb_redocking_20260322_210439_pymol/1om1_pose_compare.pse
quit
