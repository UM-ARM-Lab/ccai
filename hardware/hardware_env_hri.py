import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import pathlib
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
if __name__ == '__main__':
    from allegro_ros import RosNode
else:
    from .allegro_ros import RosNode
import torch
import yaml
from tf.transformations import euler_from_quaternion
import tf
import pytorch_kinematics as pk
from typing import Tuple, Optional

urdf_path = "/home/roboguest/Documents/git_packages/isaacgym-arm-envs/isaac_victor_envs/assets/xela_models/victor_allegro_stalk.urdf"
CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

BASE_FRAME = 'world'

class ObjectPoseReader:
    def __init__(self, obj='valve', mode='relative', device='cpu') -> None:
        if __name__ == '__main__':
            rospy.init_node('object_pose_reader')
        self.mode = mode
        self.obj = obj
        self.listener = tf.TransformListener()
        self.device = device
        self.screwdriver_pose_in_arm = None


        self.object_to_hand_matrix = torch.tensor([[[ 5.9605e-08,  1.0000e+00,  0.0000e+00,  9.5000e-02],
         [-7.6604e-01,  5.9605e-08,  6.4279e-01, -9.3431e-03],
         [ 6.4279e-01,  0.0000e+00,  7.6604e-01, -1.1135e-02],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]], device=device) #scene_trans
                                                                                
        self.object_to_hand_trans = pk.Transform3d(matrix=self.object_to_hand_matrix)
        self.hand_to_object_trans = self.object_to_hand_trans.inverse().get_matrix()

    def update_obj_pose_from_tf(self) -> None:
        """
        Updates self.obj_trans_ and self.obj_euler_ with the current pose of the screwdriver
        in the BASE_FRAME frame using tf.
        """
        try:
            self.listener.waitForTransform(BASE_FRAME, 'Umich_screwdriver', rospy.Time(0), rospy.Duration(1.0))
            trans, rot = self.listener.lookupTransform(BASE_FRAME, 'Umich_screwdriver', rospy.Time(0))
            self.obj_trans_ = np.array(trans)
            self.obj_euler_ = np.array(euler_from_quaternion(rot, axes='rxyz'))
            
            # self.obj_euler_[1] += .05
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"TF lookup failed for Umich_screwdriver in {BASE_FRAME}.")
            self.obj_trans_ = None
            self.obj_euler_ = None

    def get_state(self):
        self.update_obj_pose_from_tf()
        self.obj_euler_[0] += np.pi/2
        return self.obj_trans_, self.obj_euler_
      
    def update_screwdriver_pose(self) -> None:
        """
        Updates the pose of the Umich_screwdriver in the BASE_FRAME.
        """
        try:
            self.listener.waitForTransform(BASE_FRAME, 'Umich_screwdriver', rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform(BASE_FRAME, 'Umich_screwdriver', rospy.Time(0))
            pose = self._to_homogeneous_matrix(trans, rot)
            self.screwdriver_pose_in_arm = torch.tensor(pose, device=self.device, dtype=torch.float32)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"TF lookup failed for Umich_screwdriver in {BASE_FRAME}.")

    def get_state_world_frame_pos(self) -> Optional[torch.Tensor]:
        """
        Computes and returns the hand's pose in the arm's frame.

        Returns:
            Optional[torch.Tensor]: 4x4 transformation matrix of the hand in the arm's frame, or None if unavailable.
        """
        self.update_screwdriver_pose()
        if self.screwdriver_pose_in_arm is not None:
            hand_pose = self.screwdriver_pose_in_arm @ torch.linalg.inv(self.hand_to_object_trans)
            hand_pose_np = hand_pose.cpu().numpy()
            trans = hand_pose_np[0, :3, 3]
            rot_mat = hand_pose_np[0, :3, :3]
            euler = R.from_matrix(rot_mat).as_euler('xyz')
            return trans, euler
        return None

    @staticmethod
    def _to_homogeneous_matrix(trans: Tuple[float, float, float], rot: Tuple[float, float, float, float]) -> torch.Tensor:
        """
        Converts translation and quaternion rotation to a 4x4 homogeneous transformation matrix.

        Args:
            trans (tuple): Translation (x, y, z).
            rot (tuple): Quaternion (x, y, z, w).

        Returns:
            torch.Tensor: 4x4 transformation matrix.
        """
        import numpy as np
        import tf.transformations as tft
        mat = tft.quaternion_matrix(rot)
        mat[:3, 3] = trans
        return mat
    
    def get_target_IK_pose(self):

        return (self.hand_to_arm_mocap)


# tensor([[[ 7.9061e-01, -4.7014e-01,  3.9230e-01,  2.2960e-01],
#          [-3.4561e-04,  6.4034e-01,  7.6809e-01, -1.9502e-01],
#          [-6.1232e-01, -6.0740e-01,  5.0610e-01,  7.3183e-01],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])  self.hand_to_object_trans <- GPT says this one
class HardwareEnv:
    def __init__(self, default_pos, num_repeat=1, gradual_control=False, finger_list=['index', 'middle', 'ring', 'thumb'], kp=4, obj='valve', ori_only=True, mode='relative', device='cuda:0'):
        self.__all_finger_list = ['index', 'middle', 'ring', 'thumb']
        self.obj = obj
        self.__finger_list = finger_list
        self.__ros_node = RosNode(kp=kp, num_repeat=num_repeat, gradual_control=gradual_control)

        self.obj_reader = ObjectPoseReader(obj=obj, mode=mode)

        self.device = device
        self.default_dof_pos = default_pos.clone()
        self.ori_only = ori_only
    
    def get_state(self):
        # rospy.sleep(0.5)
        try:
            robot_state = self.__ros_node.allegro_joint_pos.float()
        except:
            print('No robot state received. Using default state.')
            robot_state = self.default_dof_pos.clone().squeeze(0)
        robot_state = robot_state.to(self.device)
        index, mid, ring, thumb = torch.chunk(robot_state, chunks=4, dim=-1)
        state = {}
        state['index'] = index
        state['middle'] = mid
        state['ring'] = ring
        state['thumb'] = thumb
        q = []
        for finger_name in self.__finger_list:
            q.append(state[finger_name])
        if self.obj == 'valve':
            ori = self.obj_reader.get_state()
            ori = torch.tensor([ori]).float().to(self.device)
            q.append(ori)
        elif 'screwdriver' in self.obj:
            pos, ori = self.obj_reader.get_state()
            pos = torch.tensor(pos).float().to(self.device)
            ori = torch.tensor(ori).float().to(self.device)
            # ori = ori * 0 # debug
            if self.ori_only:
                q.append(ori)
                q.append(torch.zeros(1).float().to(self.device)) # add the screwdriver cap angle
            else:
                raise NotImplementedError
        all_state = torch.cat((robot_state, ori), dim=-1)
        state['all_state'] = all_state
        state['q'] = torch.cat(q).unsqueeze(0)
        state['theta'] = ori
        return state
    def step(self, action):
        action = self.partial_to_full_state(action)
        # action[:, -2] += 0.25
        if len(action.shape) == 2:
            action = action.squeeze(0)
        self.__ros_node.apply_action(action)
        return self.get_state()
    def reset(self):
        return self.__ros_node.apply_action(self.default_dof_pos.squeeze(0))
    def partial_to_full_state(self, partial):
        """
        :params partial: B x 8 joint configurations for index and thumb
        :return full: B x 16 joint configuration for full hand

        # assume that default is zeros, but could change
        """
        finger_data = torch.chunk(partial, chunks=len(self.__finger_list), dim=-1)
        full = []
        ctr = 0
        for finger_name in self.__all_finger_list:
            if finger_name not in self.__finger_list:
                full.append(torch.zeros_like(finger_data[0]))
            else:
                full.append(finger_data[ctr])
                ctr += 1
        full = torch.cat(full, dim=-1)
        return full

def regularized_ik(n_tgts):
    """
    Generate an initial guess for the IK solver that keeps the arm configuration close to previously used config
    """
    init_dof = torch.zeros((n_tgts, 7), device=device)
    # The arm config from ICRA 25 in degrees. To avoid strange local minima
    init_dof += torch.tensor([[50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78]], device=device)
    init_dof = init_dof / 180 * np.pi

    init_dof += torch.randn_like(init_dof) * 0.01

    return init_dof

if __name__ == "__main__":
    """
    Calculate the arm config that aligns the hand with the object in hardware, matching alignment in simulation
    """
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_csvto_only.yaml').read_text())
    default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                dim=1)
    sim_env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                    use_cartesian_controller=False,
                                    viewer=config['visualize'],
                                    steps_per_action=60,
                                    friction_coefficient=config['friction_coefficient'] * 2.5,
                                    # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                    device=config['sim_device'],
                                    video_save_path=img_save_dir,
                                    joint_stiffness=config['kp'],
                                    fingers=config['fingers'],
                                    gradual_control=False,
                                    gravity=True, # For data generation only
                                    randomize_obj_start=config.get('randomize_obj_start', False),
                                    )
    obj_reader = ObjectPoseReader(obj='screwdriver', mode='relative')
    # # rospy.spin()
    import time
    time.sleep(.1)
    device = 'cuda:0'
    chain = pk.build_serial_chain_from_urdf(open(urdf_path, mode='rb').read(), 'allegro_hand_base_link', root_link_name='victor_right_arm_link_1')
    chain = chain.to(device=device)
    lim = torch.tensor(chain.get_joint_limits(serial=True), device=device)
    ik = pk.PseudoInverseIK(chain, max_iterations=100, num_retries=20,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="any",
                            debug=False,
                            config_sampling_method=regularized_ik,
                            # init_for_non_serial_chain=
                            lr=0.2)
    while True:
        root_coor, root_ori = obj_reader.get_state()
        # root_ori = [0, 0, 0]

        # num goals x num retries x DOF tensor of joint angles; if not converged, best solution found so far
        # print(sol.solutions)
        # num goals x num retries can check for the convergence of each run
        # print(sol.converged)
        # num goals x num retries can look at errors directly
        print(root_ori)
        # if np.abs(root_ori[0]) < .02 and np.abs(root_ori[1]) < .02:

        #     # Get converged solutions
        #     tgt_ik_pose = obj_reader.get_target_IK_pose()
        #     sol = ik.solve(tgt_ik_pose.to(chain.device))
        #     converged_sol = sol.solutions[sol.converged]
            
        #     if converged_sol.shape[0] > 0:
        #         converged_sol = converged_sol[0]
       
        #         print(converged_sol / np.pi * 180)
        #         print(sol.err_pos[sol.converged][0])
        #         print(sol.err_rot[sol.converged][0])

        cur_pose = sim_env.get_state()['q'].reshape(-1)
        
        cur_pose[-4:-1] = torch.tensor(root_ori, dtype=cur_pose.dtype)
        
        sim_env.set_pose(cur_pose.reshape(1,-1))
        
        # print(sim_env.get_state()['q'])
        # print('index', sim_env.get_state()['index_q'])
        # print('middle', sim_env.get_state()['middle_q'])
        # # print(sim_env.get_state()['ring_q'])
        # print('thumb', sim_env.get_state()['thumb_q'])
        
        # print(sim_env._q)
        # print()

