import argparse
import os
import time

import imageio.v3 as iio
import numpy as np
import torch
import viser
import yourdfpy
import xml.etree.ElementTree as ET

from viser.extras import ViserUrdf
from typing import List, Tuple, Optional, Union, Iterable, Dict

from PDF_net import PDFHR_Adapter


def read_urdf_joint_limits(urdf: Union[str, bytes], *, from_string: bool = False, include_unlimited: bool = False) -> List[Tuple[str, Optional[float], Optional[float]]]:
    root = ET.fromstring(urdf) if from_string else ET.parse(urdf).getroot()
    results = []
    def _iter_local(root, name):
        for el in root.iter():
            if el.tag.rsplit('}', 1)[-1] == name: yield el
    for j in _iter_local(root, "joint"):
        jtype = j.attrib.get("type", "")
        if jtype not in ("revolute", "prismatic"): continue
        limit_el = next((el for el in j if el.tag.rsplit('}', 1)[-1] == "limit"), None)
        lower = upper = vel = 0.0 
        if limit_el is not None:
            try:
                lower = float(limit_el.attrib.get("lower", 0.0))
                upper = float(limit_el.attrib.get("upper", 0.0))
                vel = float(limit_el.attrib.get("velocity", 0.0))
            except ValueError: pass
        if lower > upper: lower, upper = upper, lower
        results.append((lower, upper, abs(vel)))
    while len(results) < 29: results.append((0.0, 0.0, 0.0))
    return np.array(results, dtype=np.float32)


def project_random_pose_to_manifold(
    pose_model: torch.nn.Module,
    pose_low: torch.Tensor,
    pose_high: torch.Tensor,
    device: torch.device,
    num_steps: int = 1000,
    record_interval: int = 10,
    vis_interval: int = 50,
    lr: float = 0.1
):
    """
    Projects a randomly generated pose onto the learned pose manifold using gradient descent.
    """
    dof = pose_low.shape[0]
    random_noise = torch.rand(1, dof, device=device) 
    current_pose = pose_low + random_noise * (pose_high - pose_low)
    
    original_random_pose = current_pose.clone().detach()
    
    pose_list_np = [] 
    loss_history = []
    
    pose_model.eval() 
    pose_list_np.append(current_pose.detach().cpu().numpy().flatten()) 

    for i in range(num_steps):
        current_pose.requires_grad_(True)
        dist = pose_model(current_pose)
        
        grad = torch.autograd.grad(dist, current_pose)[0]
        
        with torch.no_grad():
            step = lr * dist * (grad / (grad.norm(dim=-1, keepdim=True) + 1e-8))
            current_pose = current_pose - step
            current_pose = torch.max(torch.min(current_pose, pose_high), pose_low)
            
        if i % record_interval == 0:
            pose_np = current_pose.detach().cpu().numpy().flatten()
            pose_list_np.append(pose_np)
        
        loss_history.append(dist.item())
        
        if i % vis_interval == 0:
            print(f"Iter {i:03d} | Manifold Dist: {dist.item():.6f}")

    final_pose = current_pose.detach()
    pose_list_np.append(final_pose.cpu().numpy().flatten())
    
    print(f"Done. Final Dist: {loss_history[-1]:.6f}")
    
    return original_random_pose, final_pose, loss_history, pose_list_np


def setup_camera_control_ui(server: viser.ViserServer):
    """
    Sets up the camera control UI elements in the Viser viewer.
    """
    with server.gui.add_folder("Camera Control"):
        sync_btn = server.gui.add_button("Get Current View", icon=viser.Icon.REFRESH)
        
        server.gui.add_markdown("**Camera Position (Eye)**")
        cam_x = server.gui.add_number("Pos X", initial_value=2.0, step=0.1)
        cam_y = server.gui.add_number("Pos Y", initial_value=2.0, step=0.1)
        cam_z = server.gui.add_number("Pos Z", initial_value=1.0, step=0.1)
        
        server.gui.add_markdown("**Look At Target**")
        look_x = server.gui.add_number("Look X", initial_value=0.0, step=0.1)
        look_y = server.gui.add_number("Look Y", initial_value=0.0, step=0.1)
        look_z = server.gui.add_number("Look Z", initial_value=0.5, step=0.1) 
        
        fov_slider = server.gui.add_slider("FOV (deg)", min=10.0, max=120.0, step=1.0, initial_value=75.0)

        look_at_marker = server.scene.add_mesh_simple(
            name="/camera_target_marker",
            vertices=np.array([
                [-0.05, -0.05, -0.05], [0.05, -0.05, -0.05], [0.05, 0.05, -0.05], [-0.05, 0.05, -0.05],
                [-0.05, -0.05, 0.05], [0.05, -0.05, 0.05], [0.05, 0.05, 0.05], [-0.05, 0.05, 0.05]
            ]) * 0.5, 
            faces=np.array([
                [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]
            ]),
            color=(1.0, 0.0, 0.0), 
            opacity=0.5,
        )

    def get_client():
        clients = server.get_clients()
        if not clients:
            return None
        return list(clients.values())[0]

    def sync_ui_from_camera(_):
        client = get_client()
        if client is None: return
        pos = client.camera.position
        look = client.camera.look_at
        fov = client.camera.fov
        
        cam_x.value, cam_y.value, cam_z.value = pos
        look_x.value, look_y.value, look_z.value = look
        fov_slider.value = np.degrees(fov)
        
    sync_btn.on_click(sync_ui_from_camera)

    def update_camera_from_ui(_):
        client = get_client()
        target_pos = np.array([look_x.value, look_y.value, look_z.value])
        look_at_marker.position = target_pos 
        if client is None: return
        client.camera.position = np.array([cam_x.value, cam_y.value, cam_z.value])
        client.camera.look_at = target_pos
        client.camera.fov = np.radians(fov_slider.value)

    for control in [cam_x, cam_y, cam_z, look_x, look_y, look_z, fov_slider]:
        control.on_update(update_camera_from_ui)


def run_projection_demo(urdf_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    urdf_bounds = read_urdf_joint_limits(urdf_path) 
    pose_low = torch.from_numpy(urdf_bounds[:, 0]).float().to(device)
    pose_high = torch.from_numpy(urdf_bounds[:, 1]).float().to(device)

    print(f"Loading PDFHR Model from {model_path}...")
    pose_model = PDFHR_Adapter(device=device).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("dfnet.lin0", "dfnet.layers.0")
        new_key = new_key.replace("dfnet.lin1", "dfnet.layers.1")
        new_key = new_key.replace("dfnet.lin2", "dfnet.layers.2")
        new_key = new_key.replace("dfnet.lin3", "dfnet.layers.3")
        new_state_dict[new_key] = value

    pose_model.load_state_dict(new_state_dict)

    
    pose_model.eval() 

    num_steps = 200 
    record_interval = 1
    
    _, _, _, pose_sequence = project_random_pose_to_manifold(
        pose_model=pose_model,
        pose_low=pose_low,
        pose_high=pose_high,
        device=device,
        num_steps=num_steps,  
        record_interval=record_interval,
        lr=0.1
    )

    server = viser.ViserServer(port=9191)
    
    ground_grid = server.scene.add_grid(
        "grid", width=10, height=10, position=(0.0, 0.0, 0.0), plane="xy"
    )
    server.scene.add_frame("/frame", show_axes=False)

    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=os.path.dirname(urdf_path))
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/frame")

    setup_camera_control_ui(server)

    def on_capture_click(_):
        clients = server.get_clients()
        if not clients:
            print("No client connected via browser!")
            return
        client = list(clients.values())[0]

        ground_grid.visible = False
        time.sleep(0.1) 
        
        img_binary = client.camera.get_render(height=1080, width=1920, transport_format='png')
        ground_grid.visible = True
        
        timestamp = int(time.time())
        save_path = f"robot_capture_{timestamp}.png"
        iio.imwrite(save_path, img_binary)
        print(f"Image saved to: {os.path.abspath(save_path)}")

    capture_btn = server.gui.add_button("Capture RGBA", icon=viser.Icon.CAMERA)
    capture_btn.on_click(on_capture_click)

    total_frames = len(pose_sequence)
    frame_slider = server.gui.add_slider(
        "Optimization Step",
        min=0, max=total_frames - 1, step=1, initial_value=0,
    )
    play_button = server.gui.add_button("Play Sequence")

    def update_robot(frame_idx):
        if 0 <= frame_idx < total_frames:
            pose_data = pose_sequence[int(frame_idx)]
            if len(pose_data) == len(urdf.actuated_joints):
                urdf_vis.update_cfg(pose_data)

    frame_slider.on_update(lambda _: update_robot(frame_slider.value))

    @play_button.on_click
    def _(_):
        for i in range(total_frames):
            frame_slider.value = i
            time.sleep(0.05)

    update_robot(0)
    print("Viser is running at http://localhost:9191")
    
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDFHR Projection Demo")
    
    # Update these default paths to match the structure of your GitHub repository
    parser.add_argument(
        "--urdf_path", 
        type=str, 
        default="../data/assets/g1_29dof_rev_1_0.urdf", 
        help="Path to the URDF file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="../prior_ckpts/PDFHR_epoch50.pt", 
        help="Path to the model checkpoint"
    )
    
    args = parser.parse_args()

    run_projection_demo(urdf_path=args.urdf_path, model_path=args.model_path)