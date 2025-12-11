import time
from pathlib import Path
from typing import Tuple, TypedDict
import enum
import numpy as np
import pickle
import torch
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation as R
import yourdfpy
from PIL import Image


class CameraMode(enum.Enum):
    user = 0
    sync = 1
    track = 2


def main():
    output_root = Path("record_viser/test")
    output_root.mkdir(exist_ok=True)

    urdf = yourdfpy.URDF.load(
        "/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/LAFAN1_Retargeting_Dataset/robot_description/g1/g1_29dof_rev_1_0.urdf",
        mesh_dir="/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/LAFAN1_Retargeting_Dataset/robot_description/g1/meshes",
    )

    robot_motion_file_list = [
        "data/motions/g1/g1_backflip.pkl",
        "data/motions/g1/g1_cartwheel.pkl",
        ]
    color_list = [
        (0.3, 0.8, 0.3),
        None,
        ]
    assert len(robot_motion_file_list) == len(color_list), 'number of motion files must match number of colors'

    robot_offset = np.array([3.0, 0.0, 0.0])
    cam_delta = np.array([0.35, -1.1, 1.0])
    cam_toward_delta = np.array([0.0, 0.0, 0.8])
    camera_mode = CameraMode.sync

    server = viser.ViserServer()

    robot_instances = []
    max_num_timesteps = 0
    for i, (robot_motion_file, color) in enumerate(zip(robot_motion_file_list, color_list)):
        with open(robot_motion_file, "rb") as filestream:
            in_dict = pickle.load(filestream)
            robot_motion = np.array(in_dict['frames'])
            
            num_timesteps = robot_motion.shape[0]
            if num_timesteps > max_num_timesteps:
                max_num_timesteps = num_timesteps
            
            joints = robot_motion[:, 6:]
            Ts_world_root = robot_motion[:, :6] # xyz, rx, ry, rz
            Ts_world_root[:, :3] += i * robot_offset
            root_pos = Ts_world_root[:, 0:3]
            root_rotvec = Ts_world_root[:, 3:6]
            root_quat = R.from_rotvec(root_rotvec).as_quat()[..., [3, 0, 1, 2]]    # xyzw to wxyz
            motion_name = robot_motion_file.split("/")[-1].replace(".pkl", "")
            save_dir = output_root / motion_name
            save_dir.mkdir(exist_ok=True)

        frame_name = f"/{motion_name}_{i}"
        base_frame = server.scene.add_frame(frame_name, show_axes=False)
        urdf_vis = ViserUrdf(server, urdf, root_node_name=frame_name)

        if color is not None:
            pass    # TODO: set color

        robot_instances.append({
            "frame": base_frame,
            "vis": urdf_vis,
            "num_timesteps": num_timesteps,
            "joints": joints,
            "root_pos": root_pos,
            "root_quat": root_quat, # wxyz
            "save_dir": save_dir,
        })

    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=20,
        height=20,
        position=(0.0, 0.0, 0.0),
        plane="xy"
    )
    playing = server.gui.add_checkbox("playing", True)
    recording = server.gui.add_checkbox("recording", False)
    timestep_slider = server.gui.add_slider("timestep", 0, max_num_timesteps - 1, 1, 0)

    ref_pos = [0, 0, 0]
    ref_toward_pos = [0, 0, 0]

    while True:
        clients = list(server.get_clients().values())
        client = clients[0] if len(clients) > 0 else None

        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % max_num_timesteps

            for robot in robot_instances:
                local_tstep = min(timestep_slider.value, robot['num_timesteps'] - 1)
                root_pos = robot['root_pos'][local_tstep]
                root_quat = robot['root_quat'][local_tstep]
                robot['frame'].position = np.array(root_pos)
                robot['frame'].wxyz = np.array(root_quat)
                robot['vis'].update_cfg(np.array(robot['joints'][local_tstep]))

        if recording.value and client is not None:
            for i, robot in enumerate(robot_instances):
                if local_tstep >= robot['num_timesteps']:
                    continue

                root_pos = robot['root_pos'][local_tstep]
                root_quat = robot['root_quat'][local_tstep]
                if camera_mode != CameraMode.user:
                    if camera_mode == CameraMode.sync:
                        if i == 0:
                            ref_pos = np.array([root_pos[0], root_pos[1], 0]) + cam_delta
                            ref_toward_pos = np.array([root_pos[0], root_pos[1], 0]) + cam_toward_delta
                        cam_world_pos = ref_pos + i * robot_offset
                        cam_toward_pos = ref_toward_pos + i * robot_offset
                        client.camera.up_direction = client.camera.up_direction
                    elif camera_mode == CameraMode.track:
                        cam_world_pos = np.array([root_pos[0], root_pos[1], 0]) + cam_delta
                        cam_toward_pos = np.array([root_pos[0], root_pos[1], 0]) + cam_toward_delta
                    client.camera.position = cam_world_pos
                    client.camera.look_at = cam_toward_pos
                else:
                    pass    # user controlled
                client.camera.up_direction =(0.0, 0.0, 1.0)
                img_array = client.camera.get_render(height=1080, width=1080)
                img = Image.fromarray(img_array)
                file_name = f"frame_{local_tstep:04d}.png"
                img.save(robot['save_dir'] / file_name)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
