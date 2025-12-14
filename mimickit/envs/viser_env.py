import envs.deepmimic_env as deepmimic_env
import envs.amp_env as amp_env
import envs.add_env as add_env
import envs.static_objects_env as static_objects_env
import envs.static_objects_amp_env as static_objects_amp_env
import envs.static_objects_add_env as static_objects_add_env
import envs.task_location_env as task_location_env
import envs.task_steering_env as task_steering_env

import util.torch_util as torch_util

from scipy.spatial.transform import Rotation as R
import viser
from viser.extras import ViserUrdf
import yourdfpy

import os
import numpy as np
import copy
import torch
import time
import enum
import isaacgym.gymapi as gymapi


ENV_CLASS_MAP = {
    'deepmimic': deepmimic_env.DeepMimicEnv,
    'amp': amp_env.AMPEnv,
    'add': add_env.ADDEnv,
    'static_objects': static_objects_env.StaticObjectsEnv,
    'static_objects_amp': static_objects_amp_env.StaticObjectsEnv,
    'static_objects_add': static_objects_add_env.StaticObjectsEnv,
    'task_location': task_location_env.TaskLocationEnv,
    'task_steering': task_steering_env.TaskSteeringEnv,
}


class ViserCameraMode(enum.Enum):
    user = 0
    track = 1
    sync = 2


class ViserMixin:
    def __init__(self, *args, **kwargs):
        self.server = viser.ViserServer()
        self.viser_base_env_name = kwargs.get('config', {}).get('env_name', 'unknown')
        self.viser_cam_config = kwargs.get('config', {}).get('viser_cam', {})
        self.viser_cam_config_handles = {}
        with self.server.gui.add_folder("Viser Camera Config"):
            self.viser_cam_config_handles["viser_cam_mode"] = self.server.gui.add_dropdown(
                label="viser_cam_mode",
                options =["user", "track", "sync"],
                initial_value=ViserCameraMode(self.viser_cam_config.get('viser_cam_mode', 0)).name,
            )
            self.viser_cam_config_handles["resolution_width"] = self.server.gui.add_number(
                label="resolution_width",
                initial_value=self.viser_cam_config.get('cam_resolution', [1080, 1080])[0],
                step=100,
            )
            self.viser_cam_config_handles["resolution_height"] = self.server.gui.add_number(
                label="resolution_height",
                initial_value=self.viser_cam_config.get('cam_resolution', [1080, 1080])[1],
                step=100,
            )
            self.viser_cam_config_handles["cam_delta_x"] = self.server.gui.add_number(
                label="cam_delta_x",
                initial_value=self.viser_cam_config.get('cam_delta', [0.35, -1.1, 1.0])[0],
                step=0.1,
            )
            self.viser_cam_config_handles["cam_delta_y"] = self.server.gui.add_number(
                label="cam_delta_y",
                initial_value=self.viser_cam_config.get('cam_delta', [0.35, -1.1, 1.0])[1],
                step=0.1,
            )
            self.viser_cam_config_handles["cam_delta_z"] = self.server.gui.add_number(
                label="cam_delta_z",
                initial_value=self.viser_cam_config.get('cam_delta', [0.35, -1.1, 1.0])[2],
                step=0.1,
            )
            self.viser_cam_config_handles["cam_toward_delta_x"] = self.server.gui.add_number(
                label="cam_toward_delta_x",
                initial_value=self.viser_cam_config.get('cam_toward_delta', [0.0, 0.0, 0.8])[0],
                step=0.1,
            )
            self.viser_cam_config_handles["cam_toward_delta_y"] = self.server.gui.add_number(
                label="cam_toward_delta_y",
                initial_value=self.viser_cam_config.get('cam_toward_delta', [0.0, 0.0, 0.8])[1],
                step=0.1,
            )
            self.viser_cam_config_handles["cam_toward_delta_z"] = self.server.gui.add_number(
                label="cam_toward_delta_z",
                initial_value=self.viser_cam_config.get('cam_toward_delta', [0.0, 0.0, 0.8])[2],
                step=0.1,
            )

        self.viser_robot_urdf = yourdfpy.URDF.load("data/assets/viser_g1/g1_29dof_rev_1_0.urdf", mesh_dir="data/assets/viser_g1/meshes")
        self.viser_ref_robot_urdf = yourdfpy.URDF.load("data/assets/viser_g1/g1_29dof_rev_1_0_colored.urdf", mesh_dir="data/assets/viser_g1/meshes")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid(
            "/grid",
            width=20,
            height=20,
            position=(0.0, 0.0, 0.0),
            plane="xy"
        )
        self._viser_playing = self.server.gui.add_checkbox("playing", True)
        self._viser_recording = self.server.gui.add_checkbox("recording", False)
        self._viser_ref_robot_instances = []
        self._viser_char_robot_instances = []

        super().__init__(*args, **kwargs)

    # for initializing ref robot
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        if (self._enable_ref_char()):
            ref_frame_name = f"/ref/env_{env_id}"
            ref_root_frame = self.server.scene.add_frame(
                name=ref_frame_name,
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                show_axes=False
            )
            ref_urdf_vis = ViserUrdf(self.server, self.viser_ref_robot_urdf, root_node_name=ref_frame_name)
            self._viser_ref_robot_instances.append({
                "frame": ref_root_frame,
                "vis": ref_urdf_vis,
            })

        return

    # for updating ref robot
    def _update_ref_char(self):
        super()._update_ref_char()
        root_pos = self._ref_root_pos + self._ref_char_offset
        for i, ref_robot in enumerate(self._viser_ref_robot_instances):
            frame = ref_robot['frame']
            vis = ref_robot['vis']
            root_pos = self._ref_root_pos[i, :].cpu().numpy() + self._ref_char_offset.cpu().numpy()
            root_quat = self._ref_root_rot[i, :].cpu().numpy()  # xyzw
            root_quat = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])  # wxyz
            frame.position  = root_pos
            frame.wxyz = root_quat
            dof_pos = self._ref_dof_pos[i, :].cpu().numpy()
            vis.update_cfg(dof_pos)

        return
    
    # for initializing char robot
    def _build_character(self, env_id, config, color=None):
        result = super()._build_character(env_id, config, color=color)
        char_frame_name = f"/char/env_{env_id}"
        char_root_frame = self.server.scene.add_frame(
            name=char_frame_name,
            position=np.array([0.0, 0.0, 0.0]),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            show_axes=False
        )
        char_urdf_vis = ViserUrdf(self.server, self.viser_robot_urdf, root_node_name=char_frame_name)
        self._viser_char_robot_instances.append({
            "frame": char_root_frame,
            "vis": char_urdf_vis,
        })
        return result


    # for updating char robot
    def step(self, action):
        results = super().step(action)
        char_id = self._get_char_id()
        root_pos = self._engine._root_pos[:, char_id, :].cpu().numpy()
        root_quat = self._engine._root_rot[:, char_id, :].cpu().numpy()
        dof_pos = self._engine._obj_dof_pos[char_id].cpu().numpy()
        for i, char_robot in enumerate(self._viser_char_robot_instances):
            frame = char_robot['frame']
            vis = char_robot['vis']
            frame.position = root_pos[i]
            frame.wxyz = np.array([root_quat[i][3], root_quat[i][0], root_quat[i][1], root_quat[i][2]])
            vis.update_cfg(dof_pos[i])

        self._update_viser_camera()

        return results

    # for static_objects envs
    def _build_static_object(self, env_id, config):
        super()._build_static_object(env_id, config)
        objs_config = config["env"].get("objects", [])
        for i, obj_config in enumerate(objs_config):
            asset_file = obj_config["file"].replace('.xml', '.urdf').replace('.usd', '.urdf')
            pos = obj_config["pos"]
            quat = obj_config.get("rot", [0.0, 0.0, 0.0, 1.0])   # (x,y,z,w)
            root_frame_name = f"/static_objects/obj_{env_id}_{i}"
            static_obj_root_frame = self.server.scene.add_frame(
                name=root_frame_name,
                position=np.array(pos),
                wxyz=np.array([quat[3], quat[0], quat[1], quat[2]]),
                show_axes=False
            )
            static_obj_urdf = yourdfpy.URDF.load(asset_file)
            static_obj_vis = ViserUrdf(self.server, static_obj_urdf, root_node_name=root_frame_name)

        return

    # for task_location env
    def _build_marker(self, env_id):
        marker_id = super()._build_marker(env_id)
        if not hasattr(self, '_viser_marker_frames'):
            self._viser_marker_frames = [None] * self._engine._num_envs
        asset_file = "data/assets/objects/location_marker.urdf"       
        
        marker_frame_name = f"/markers/env_{env_id}"
        marker_frame_handle = self.server.scene.add_frame(
            name=marker_frame_name,
            position=np.array([0, 0, 0]),
            wxyz=np.array([1, 0, 0, 0]), # Identity rotation
            show_axes=False
        )
        marker_urdf = yourdfpy.URDF.load(asset_file)
        marker_vis = ViserUrdf(self.server, marker_urdf, root_node_name=marker_frame_name)
        self._viser_marker_frames[env_id] = marker_frame_handle
    
        return marker_id

    # for task_steering env
    def _build_markers(self, env_id):
        tar_marker_id, face_marker_id = super()._build_markers(env_id)
        if not hasattr(self, '_viser_marker_frames'):
            self._viser_tar_marker_frames = [None] * self._engine._num_envs
            self._viser_face_marker_frames = [None] * self._engine._num_envs
        asset_file = "data/assets/objects/steering_marker.urdf"
        marker_urdf = yourdfpy.URDF.load(asset_file)
        tar_name = f"/markers/env_{env_id}/steering_target"
        tar_frame_handle = self.server.scene.add_frame(
                name=tar_name,
                position=np.array([0,0,0]),
                wxyz=np.array([1,0,0,0]),
                show_axes=False
            )
        face_name = f"/markers/env_{env_id}/steering_face"
        face_frame_handle = self.server.scene.add_frame(
                name=face_name,
                position=np.array([0,0,0]),
                wxyz=np.array([1,0,0,0]),
                show_axes=False
            )
        self._viser_tar_marker_frames[env_id] = tar_frame_handle
        self._viser_face_marker_frames[env_id] = face_frame_handle

        return tar_marker_id, face_marker_id
    
    # for task_location & task_steering env
    def _update_marker(self, env_ids):
        super()._update_marker(env_ids)
        if 'task_location' in self.viser_base_env_name:
            for env_id in env_ids:
                marker_frame_handle = self._viser_marker_frames[env_id]
                target_pos = self._tar_pos[env_id].detach().cpu().numpy()
                marker_frame_handle.position = target_pos

        if 'task_steering' in self.viser_base_env_name:
            tar_dist_min = 1.0
            tar_dist_max = 1.5

            char_id = self._get_char_id()
            
            root_pos = self._engine.get_root_pos(char_id)
            root_pos = root_pos[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_dir = self._tar_dir[env_ids]
            face_dir = self._face_dir[env_ids]

            tar_dist = (tar_speed - self._tar_speed_min) / (self._tar_speed_max - self._tar_speed_min)
            tar_dist = (tar_dist_max - tar_dist_min) * tar_dist + tar_dist_min
            tar_dist = tar_dist.unsqueeze(-1)

            marker_pos = root_pos.clone()
            marker_pos[..., 0:2] += tar_dist * tar_dir
            marker_pos[..., 2] = 0.0

            tar_theta = torch.atan2(tar_dir[..., 1], tar_dir[..., 0])
            tar_axis = torch.zeros_like(root_pos)
            tar_axis[..., -1] = 1.0
            marker_rot = torch_util.axis_angle_to_quat(tar_axis, tar_theta)
            
            face_marker_pos = root_pos.clone()
            face_marker_pos[..., 0:2] += tar_dist_min * face_dir
            face_marker_pos[..., 2] = 0.01

            face_theta = torch.atan2(face_dir[..., 1], face_dir[..., 0])
            face_axis = torch.zeros_like(root_pos)
            face_axis[..., -1] = 1.0
            face_marker_rot = torch_util.axis_angle_to_quat(tar_axis, face_theta)

            np_marker_pos = marker_pos.detach().cpu().numpy()
            np_marker_rot = marker_rot.detach().cpu().numpy()   # xyzw
            np_face_pos = face_marker_pos.detach().cpu().numpy()
            np_face_rot = face_marker_rot.detach().cpu().numpy()    # xyzw

            for i, env_id in enumerate(env_ids):
                tar_marker_frame_handle = self._viser_tar_marker_frames[env_id]
                face_marker_frame_handle = self._viser_face_marker_frames[env_id]
                tar_marker_frame_handle.position = np_marker_pos[i]
                face_marker_frame_handle.position = np_face_pos[i]
                tar_marker_frame_handle.wxyz = np.array([np_marker_rot[i][3], np_marker_rot[i][0], np_marker_rot[i][1], np_marker_rot[i][2]])
                face_marker_frame_handle.wxyz = np.array([np_face_rot[i][3], np_face_rot[i][0], np_face_rot[i][1], np_face_rot[i][2]])

        return

    def _init_camera(self):
        pass

    def _render(self):
        pass

    def _update_viser_camera(self):
        current_cam_mode = ViserCameraMode[self.viser_cam_config_handles["viser_cam_mode"].value]
        cam_delta = np.array([
                self.viser_cam_config_handles["cam_delta_x"].value,
                self.viser_cam_config_handles["cam_delta_y"].value,
                self.viser_cam_config_handles["cam_delta_z"].value,
            ])
        cam_toward_delta = np.array([
                self.viser_cam_config_handles["cam_toward_delta_x"].value,
                self.viser_cam_config_handles["cam_toward_delta_y"].value,
                self.viser_cam_config_handles["cam_toward_delta_z"].value,
            ])
        if current_cam_mode == ViserCameraMode.user:
            pass
        elif current_cam_mode == ViserCameraMode.track:
            char_id = self._get_char_id()
            root_pos = self._engine._root_pos[0, char_id, :].cpu().numpy()  # 1st env
            clients = list(self.server.get_clients().values())
            client = clients[0] if len(clients) > 0 else None
            if client is not None:
                # char
                cam_world_pos = np.array([root_pos[0], root_pos[1], 0]) + cam_delta
                cam_toward_pos = np.array([root_pos[0], root_pos[1], 0]) +  cam_toward_delta
                client.camera.position = cam_world_pos
                client.camera.look_at = cam_toward_pos
                client.camera.up_direction =(0.0, 0.0, 1.0)
                time.sleep(0.1)
                # ref
                if (self._enable_ref_char()):
                    root_pos = self._ref_root_pos[0, :].cpu().numpy() + self._ref_char_offset.cpu().numpy()   # 1st env
                    ref_cam_world_pos = np.array([root_pos[0], root_pos[1], 0]) + cam_delta
                    ref_cam_toward_pos = np.array([root_pos[0], root_pos[1], 0]) +  cam_toward_delta
                    client.camera.position = ref_cam_world_pos
                    client.camera.look_at = ref_cam_toward_pos
                    client.camera.up_direction = (0.0, 0.0, 1.0)
                time.sleep(0.1)

        elif current_cam_mode == ViserCameraMode.sync:
            if not (self._enable_ref_char()):
                print("Warning: sync camera mode requires reference character enabled.")
                print('Switching to user mode.')
                self.viser_cam_config_handles["viser_cam_mode"].value = ViserCameraMode.user.name
            else:
                char_id = self._get_char_id()
                ref_root_pos = self._ref_root_pos[0, :].cpu().numpy()   # 1st env
                clients = list(self.server.get_clients().values())
                client = clients[0] if len(clients) > 0 else None
                if client is not None:
                    # char
                    cam_world_pos = np.array([ref_root_pos[0], ref_root_pos[1], 0]) + cam_delta
                    cam_toward_pos = np.array([ref_root_pos[0], ref_root_pos[1], 0]) + cam_toward_delta
                    client.camera.position = cam_world_pos
                    client.camera.look_at = cam_toward_pos
                    client.camera.up_direction = (0.0, 0.0, 1.0)
                    time.sleep(0.1)
                    # ref
                    ref_cam_world_pos = np.array([ref_root_pos[0], ref_root_pos[1], 0]) + cam_delta + self._ref_char_offset.cpu().numpy()
                    ref_cam_toward_pos = np.array([ref_root_pos[0], ref_root_pos[1], 0]) + cam_toward_delta + self._ref_char_offset.cpu().numpy()
                    client.camera.position = ref_cam_world_pos
                    client.camera.look_at = ref_cam_toward_pos
                    client.camera.up_direction = (0.0, 0.0, 1.0)
                    time.sleep(0.1)

        return

    # TODO
    def _save_img(self, client):
        pass




# dynamic Env builder for viser envs
def ViserEnvBuilder(env_name, config, num_envs, device, visualize):
    base_name = env_name.replace("_viser", "")
    base_class = ENV_CLASS_MAP.get(base_name)
    if base_class is None:
        raise ValueError(f"Unsupported base env for recording: {base_name}")

    config_base = copy.deepcopy(config)
    config_base['env_name'] = base_name

    new_class_name = f"Viser{base_class.__name__}"
    DynamicClass = type(new_class_name, (ViserMixin, base_class), {})

    return DynamicClass(config=config_base, num_envs=num_envs, device=device, visualize=visualize)

