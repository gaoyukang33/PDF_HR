import envs.deepmimic_env as deepmimic_env
import envs.amp_env as amp_env
import envs.add_env as add_env
import envs.static_objects_env as static_objects_env
import envs.static_objects_amp_env as static_objects_amp_env
import envs.static_objects_add_env as static_objects_add_env
import envs.task_location_env as task_location_env
import envs.task_steering_env as task_steering_env

from envs.char_env import CameraMode
import engines.engine as engine
import util.torch_util as torch_util
from scipy.spatial.transform import Rotation as R
import pickle

import os
import numpy as np
import copy
import isaacgym.gymapi as gymapi


ENV_CLASS_MAP = {
    'deepmimic': deepmimic_env.DeepMimicEnv,
    'amp': amp_env.AMPEnv,
    'add': add_env.ADDEnv,
    'static_objects': static_objects_env.StaticObjectsEnv,
    'static_objects_amp': static_objects_amp_env.StaticObjectsEnv,
    'static_objects_add': static_objects_add_env.StaticObjectsEnv,
    'task_location_env': task_location_env.TaskLocationEnv,
    'task_steering_env': task_steering_env.TaskSteeringEnv,
}


class RecordMixin:
    def __init__(self, *args, **kwargs):
        self.record_cfg = kwargs.get('config', {}).get('env', {}).get('record_config', {})
        self.record_frame_path = self.record_cfg.get('record_frame_path', None)
        assert self.record_frame_path is not None, "record_frame_path must be specified in record_config"
        self.record_resolution = self.record_cfg.get('record_resolution', [1960, 1080])
        self.record_cam_mode = CameraMode(self.record_cfg.get('record_cam_mode', 0))  # 0: still, 1: track
        self.record_cam_delta = gymapi.Vec3(*self.record_cfg.get('record_cam_delta', [0.0, 0.0, 0.0]))
        self.record_cam_toward_delta = gymapi.Vec3(*self.record_cfg.get('record_cam_toward_delta', [0.0, 0.0, 1.0]))
        self.light_color = gymapi.Vec3(*self.record_cfg.get('light_color', [0.8, 0.8, 0.8]))
        self.light_ambient = gymapi.Vec3(*self.record_cfg.get('light_ambient', [0.4, 0.4, 0.4]))
        self.light_direction = gymapi.Vec3(0.1, -0.1, 0.5)
        self.bg_objects = self.record_cfg.get('bg_objects', [])
        self.record_motion_pkl = self.record_cfg.get('record_motion_pkl', False)
        self.record_motion_img = self.record_cfg.get('record_motion_img', False)

        self.record_motion_ref_data = {
            'loop_mode': 0,
            'fps': kwargs.get('config', {}).get('engine', {}).get("control_freq", 30),
            'frames': []
        }
        self.record_motion_char_data = {
            'loop_mode': 0,
            'fps': kwargs.get('config', {}).get('engine', {}).get("control_freq", 30),
            'frames': []
        }

        super().__init__(*args, **kwargs)

    # re-define the functions needed for recording
    def _init_camera(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_pos = char_root_pos[0].cpu().numpy()
        ref_id = self._get_ref_char_id()
        ref_root_pos = self._engine.get_root_pos(ref_id)
        ref_pos = ref_root_pos[0].cpu().numpy()

        cam_pos = [char_pos[0], char_pos[1] - 5.0, 3.0]
        cam_target = [char_pos[0], char_pos[1], 0.0]

        self._engine.update_camera(cam_pos, cam_target)
        self._cam_prev_char_pos = char_pos

        self.frame_id = 0
        os.makedirs(self.record_frame_path, exist_ok=True)
        os.makedirs(os.path.join(self.record_frame_path, 'char'), exist_ok=True)
        os.makedirs(os.path.join(self.record_frame_path, 'ref'), exist_ok=True)
        record_cam_props = gymapi.CameraProperties()
        record_cam_props.width = self.record_resolution[0]
        record_cam_props.height = self.record_resolution[1]
        # self.record_cam_pos_char = gymapi.Vec3(char_pos[0], char_pos[1], 0) + self.record_cam_delta
        # self.record_cam_target_char = gymapi.Vec3(char_pos[0], char_pos[1], 0) + self.record_cam_toward_delta
        # self.record_cam_pos_ref = self.record_cam_pos_char + gymapi.Vec3(*self._ref_char_offset)
        # self.record_cam_target_ref = self.record_cam_target_char + gymapi.Vec3(*self._ref_char_offset)
        self.record_cam_pos_ref = gymapi.Vec3(ref_pos[0], ref_pos[1], 0) + self.record_cam_delta
        self.record_cam_target_ref = gymapi.Vec3(ref_pos[0], ref_pos[1], 0) + self.record_cam_toward_delta
        self.record_cam_pos_char = self.record_cam_pos_ref - gymapi.Vec3(*self._ref_char_offset)
        self.record_cam_target_char = self.record_cam_target_ref - gymapi.Vec3(*self._ref_char_offset)
        self._engine.record_camera_char_handle = self._engine._gym.create_camera_sensor(self._engine.get_env(0), record_cam_props)
        self._engine.record_camera_ref_handle = self._engine._gym.create_camera_sensor(self._engine.get_env(0), record_cam_props)
        # self._engine._gym.attach_camera_to_body(self._engine.record_camera_char_handle,
        #                                         self._engine.get_env(0),
        #                                         self._char_ids[0],
        #                                         gymapi.Transform(p=self.record_cam_pos_char),
        #                                         gymapi.FOLLOW_TRANSFORM)
        self._engine._gym.set_camera_location(self._engine.record_camera_char_handle, self._engine.get_env(0), self.record_cam_pos_char, self.record_cam_target_char)
        self._engine._gym.set_camera_location(self._engine.record_camera_ref_handle, self._engine.get_env(0), self.record_cam_pos_ref, self.record_cam_target_ref)

        self._engine._gym.set_light_parameters(self._engine._sim, 0, self.light_color, self.light_ambient, self.light_direction)

        return


    def _update_camera(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_pos = char_root_pos[0].cpu().numpy()
        ref_id = self._get_ref_char_id()
        ref_root_pos = self._engine.get_root_pos(ref_id)
        ref_pos = ref_root_pos[0].cpu().numpy()

        cam_pos = self._engine.get_camera_pos()
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = np.array([char_pos[0], char_pos[1], 1.0])
        new_cam_pos = np.array([char_pos[0] + cam_delta[0], 
                                char_pos[1] + cam_delta[1], 
                                cam_pos[2]])

        if (self._camera_mode is CameraMode.still):
            pass
        elif (self._camera_mode is CameraMode.track):
            self._engine.update_camera(new_cam_pos, new_cam_target)

            self._cam_prev_char_pos[:] = char_pos
        else:
            assert(False), "Unsupported camera mode {}".format(self._camera_mode)


        frame_filename_char = os.path.join(self.record_frame_path, 'char', "frame_{:05d}_char.png".format(self.frame_id))
        frame_filename_ref = os.path.join(self.record_frame_path, 'ref', "frame_{:05d}_ref.png".format(self.frame_id))
        if self.record_cam_mode == CameraMode.track:
            # self.record_cam_pos_char = gymapi.Vec3(char_pos[0], char_pos[1], 0.0) + self.record_cam_delta
            # self.record_cam_target_char = gymapi.Vec3(char_pos[0], char_pos[1], 0.0) + self.record_cam_toward_delta
            # self.record_cam_pos_ref = self.record_cam_pos_char + gymapi.Vec3(*self._ref_char_offset)
            # self.record_cam_target_ref = self.record_cam_target_char + gymapi.Vec3(*self._ref_char_offset)
            self.record_cam_pos_ref = gymapi.Vec3(ref_pos[0], ref_pos[1], 0.0) + self.record_cam_delta
            self.record_cam_target_ref = gymapi.Vec3(ref_pos[0], ref_pos[1], 0.0) + self.record_cam_toward_delta
            self.record_cam_pos_char = self.record_cam_pos_ref - gymapi.Vec3(*self._ref_char_offset)
            self.record_cam_target_char = self.record_cam_target_ref - gymapi.Vec3(*self._ref_char_offset)

        self._engine._gym.set_camera_location(
            self._engine.record_camera_char_handle, 
            self._engine.get_env(0), 
            self.record_cam_pos_char,
            self.record_cam_target_char,
        )
        self._engine._gym.set_camera_location(
            self._engine.record_camera_ref_handle, 
            self._engine.get_env(0), 
            self.record_cam_pos_ref,
            self.record_cam_target_ref,
        )
        self._engine._gym.render_all_camera_sensors(self._engine._sim)
        if self.record_motion_img:
            self._engine._gym.write_camera_image_to_file(self._engine._sim,
                                                        self._engine.get_env(0),
                                                        self._engine.record_camera_char_handle,
                                                        gymapi.IMAGE_COLOR,
                                                        frame_filename_char)
            self._engine._gym.write_camera_image_to_file(self._engine._sim,
                                                        self._engine.get_env(0),
                                                        self._engine.record_camera_ref_handle,
                                                        gymapi.IMAGE_COLOR,
                                                        frame_filename_ref)
        self.frame_id += 1

        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        self._build_bg_object()
        return
    
    def _build_bg_object(self):
        color = np.array([1.0, 1.0, 1.0])

        for i, obj_config in enumerate(self.bg_objects):
            asset_file = obj_config["file"]
            pos = obj_config["pos"]
            rot = obj_config.get("rot", [0.0, 0.0, 0.0, 1.0])

            pos = np.array(pos)
            rot = np.array(rot)

            obj_name = "static_object{:d}".format(i)
            self._engine.create_obj(env_id=0,
                                    obj_type=engine.ObjType.rigid,
                                    asset_file=asset_file,
                                    name=obj_name,
                                    start_pos=pos,
                                    start_rot=rot,
                                    fix_root=True,
                                    color=color)
        return


    def step(self, action):
        self._pre_physics_step(action)

        self._physics_step()
        
        # compute observations, rewards, resets, ...
        self._post_physics_step()

        if (self._visualize):
            self._render()
        
        if self.record_motion_pkl:
            self.record_motion_ref_data['frames'].append(self._get_ref_motion_data(0))
            self.record_motion_char_data['frames'].append(self._get_char_motion_data(0))

            if self._done_buf[0] != 0:
                print('recording motion pkl')
                record_motion_ref_path = os.path.join(self.record_frame_path, "ref.pkl")
                record_motion_char_path = os.path.join(self.record_frame_path, "char.pkl")
                print(record_motion_ref_path, record_motion_char_path)
                with open(record_motion_ref_path, "wb") as f:
                    pickle.dump(self.record_motion_ref_data, f)
                with open(record_motion_char_path, "wb") as f:
                    pickle.dump(self.record_motion_char_data, f)

        return self._obs_buf, self._reward_buf, self._done_buf, self._info

    def _enable_ref_char(self):
        return True

    def _get_ref_motion_data(self, env_id):
        # ref_id = self._get_ref_char_id()
        root_pos = self._ref_root_pos[env_id].cpu().numpy()
        root_quat = self._ref_root_rot[env_id].cpu().numpy()    # xyzw
        root_rotvec = R.from_quat(root_quat).as_rotvec()
        dof_pos = self._ref_dof_pos[env_id].cpu().numpy()

        return np.concatenate([root_pos, root_rotvec, dof_pos], axis=0)

    def _get_char_motion_data(self, env_id):
        char_id = self._get_char_id()
        root_pos = self._engine._root_pos[env_id, char_id, :].cpu().numpy()
        root_quat = self._engine._root_rot[env_id, char_id, :].cpu().numpy()    # xyzw
        root_rotvec = R.from_quat(root_quat).as_rotvec()
        dof_pos = self._engine._obj_dof_pos[char_id][env_id, :].cpu().numpy()

        return np.concatenate([root_pos, root_rotvec, dof_pos], axis=0)


# dynamic Env builder for record envs
def RecordEnvBuilder(env_name, config, num_envs, device, visualize):
    base_name = env_name.replace("_record", "")
    base_class = ENV_CLASS_MAP.get(base_name)
    if base_class is None:
        raise ValueError(f"Unsupported base env for recording: {base_name}")

    config_base = copy.deepcopy(config)
    config_base['env_name'] = base_name

    new_class_name = f"Record{base_class.__name__}"
    DynamicClass = type(new_class_name, (RecordMixin, base_class), {})

    return DynamicClass(config=config_base, num_envs=num_envs, device=device, visualize=visualize)



