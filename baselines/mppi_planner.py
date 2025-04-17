class DummyProblem:
    def __init__(self, dx, T):
        self.dx = dx
        self.du = dx
        self.T = T
        self.data = {}
class MPPIPlanner:
    def __init__(self, ctrl, dx, T):
        self.ctrl = ctrl
        self.problem = DummyProblem(dx, T)

        self.warmed_up = False
        self.path = []
    
    def step(self, state):
        """
        Perform one step of the MPPI planner.
        
        Args:
            state: The current state of the system.
        
        Returns:
            action: The computed action based on the current state.
        """
        prime_dof_state = self.ctrl.F.env.dof_states.clone()[0].to(device=self.ctrl.d)
        prime_dof_state = prime_dof_state.unsqueeze(0).repeat(self.ctrl.K, 1, 1)
        
        orig_external_wrench_perturb = self.ctrl.F.env.external_wrench_perturb
        
        self.ctrl.F.env.set_external_wrench_perturb(False)

        if not self.warmed_up:
            for _ in range(4):
                action = self.ctrl.command(prime_dof_state[0].reshape(-1), shift_nominal_trajectory=False)
            self.warmed_up = True

        action = self.ctrl.command(prime_dof_state[0].reshape(-1), shift_nominal_trajectory=True)

        self.ctrl.F.env.set_pose(prime_dof_state, semantic_order=False, zero_velocity=False, ignore_img=True)
        self.ctrl.F.env.set_external_wrench_perturb(orig_external_wrench_perturb)
        action = action.repeat(self.ctrl.K, 1)

        return action, action.unsqueeze(1)
    
    def reset(self, *args, **kwargs):
        pass
