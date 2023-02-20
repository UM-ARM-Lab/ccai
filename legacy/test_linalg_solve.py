import torch
import time


n = 64
d = 800

device = 'cpu'
device = 'cuda:0'
# try a banded diagonal
A1 = torch.eye(d).reshape(1, d, d).repeat(n, 1, 1).to(device=device)
A2 = torch.diag_embed(torch.ones(n, d), offset=1)[:, :-1, :-1].to(device=device)
A3 = torch.diag_embed(torch.ones(n, d), offset=-1)[:, :-1, :-1].to(device=device)
b = torch.randn(n, d).to(device=device)
A = A1 + A2 + A3
A = A @ A.permute(0, 2, 1) + A1
print(A.shape)
print('LINALG solve')
torch.cuda.synchronize()
s = time.time()
for _ in range(100):
    X1 = torch.linalg.solve(A, b)
torch.cuda.synchronize()
print((time.time() - s) / 100)

print('LINALG inv')
torch.cuda.synchronize()
s = time.time()
for _ in range(100):
    _ = torch.linalg.pinv(A)
torch.cuda.synchronize()
print((time.time() - s) / 100)

print('CHOLESKY VERSION')
b = b.unsqueeze(-1)
torch.cuda.synchronize()
s = time.time()
for _ in range(100):

    u = torch.linalg.cholesky(A)
    X2 = torch.cholesky_solve(b, u)
torch.cuda.synchronize()
print((time.time() - s) / 100)

print('Conjugate gradient version')
from torch_cg import cg_batch
torch.cuda.synchronize()
s = time.time()
b.requires_grad = True
A_bmm = lambda x: A @ x

for _ in range(100):
    X3, _ = cg_batch(A_bmm, b, verbose=False)
torch.cuda.synchronize()
print((time.time() - s) / 100)
print(X1.shape, X2.shape, X3.shape)
print((X1 - X2.squeeze(-1)).abs().max())
print((X1 - X3.squeeze(-1)).abs().max())

#assert torch.allclose(X1, X2.squeeze(-1))

#device='cpu'
# test if for l0oop slower than combined big sparse
N, M = 400, 100
b = torch.randn(N).to(device=device)
A = torch.eye(N).to(device=device)
torch.cuda.synchronize()

s = time.time()
for _ in range(M):
    torch.linalg.solve(A, b)
    #_ = torch.cholesky_solve(b.unsqueeze(-1), A)
torch.cuda.synchronize()

e = time.time()

print(e - s)
b = torch.randn(N*M).to(device=device)
A = torch.eye(N*M).to(device=device)
torch.cuda.synchronize()

s = time.time()
#_ = torch.solve(A, b)
_ = torch.cholesky_solve(b.unsqueeze(-1), A)
torch.cuda.synchronize()

e = time.time()
print(e - s)

b = torch.randn(M, N).to(device=device)
A = torch.eye(N).to(device=device).reshape(1, N, N).repeat(M, 1, 1)
torch.cuda.synchronize()
s = time.time()
#_ = torch.solve(A, b)
_ = torch.cholesky_solve(b.unsqueeze(-1), A)
torch.cuda.synchronize()

e = time.time()
print(e - s)
torch.cuda.synchronize()
