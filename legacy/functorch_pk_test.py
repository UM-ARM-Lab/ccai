import allegro_optimized_wrapper as pk
import torch
import functorch

asset = '/home/tpower/dev/isaac_test/IsaacVictorEnvs/isaac_victor_envs/assets/victor/victor.urdf'
ee_name = 'l_palm'

chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)

f = functorch.vmap(chain.forward_kinematics)
df = functorch.vmap(functorch.jacrev(chain.forward_kinematics))
ddf = functorch.vmap(functorch.hessian(chain.forward_kinematics))
theta = torch.randn(5, 7)
print(theta.shape)
transform = f(theta)
print(transform)

df(theta)
h = ddf(theta)
print(h[0])


