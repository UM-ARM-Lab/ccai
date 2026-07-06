import math

import torch

from ccai.models.trajectory_samplers_sac import TrajectorySampler


class _FakeSampler:
    T = 6
    use_mixed_precision = False
    convert_yaw_to_sine_cosine = TrajectorySampler.convert_yaw_to_sine_cosine
    check_id_batch = TrajectorySampler.check_id_batch
    check_id = TrajectorySampler.check_id

    def __init__(self):
        self.sample_calls = []

    def sample(self, N, H, start, constraints):
        self.sample_calls.append(
            {
                "N": N,
                "H": H,
                "start": start.detach().clone(),
                "constraints": constraints.detach().clone(),
            }
        )
        likelihood = torch.arange(N, device=start.device, dtype=start.dtype)
        samples = torch.empty((N, H, 1), device=start.device, dtype=start.dtype)
        return samples, None, likelihood


def test_check_id_batch_averages_likelihood_samples_per_state():
    sampler = _FakeSampler()
    states = torch.zeros((2, 15), dtype=torch.float32)
    states[0, 14] = math.pi / 2.0
    states[1, 14] = math.pi

    likelihood = sampler.check_id_batch(states, 3, likelihood_only=True)

    torch.testing.assert_close(likelihood, torch.tensor([1.0, 4.0]))
    assert len(sampler.sample_calls) == 1
    call = sampler.sample_calls[0]
    assert call["N"] == 6
    assert call["H"] == sampler.T
    assert call["start"].shape == (6, 16)
    assert call["constraints"].shape == (6, 3)
    torch.testing.assert_close(call["start"][0:3], call["start"][0].repeat(3, 1))
    torch.testing.assert_close(call["start"][3:6], call["start"][3].repeat(3, 1))
    torch.testing.assert_close(call["start"][0, 14:16], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(call["start"][3, 14:16], torch.tensor([-1.0, 0.0]), atol=1e-6, rtol=1e-6)


def test_scalar_check_id_likelihood_only_matches_one_state_batch():
    sampler = _FakeSampler()
    state = torch.zeros(15, dtype=torch.float32)

    scalar_likelihood = sampler.check_id(state, 3, likelihood_only=True)
    batch_likelihood = sampler.check_id_batch(state.unsqueeze(0), 3, likelihood_only=True)

    assert scalar_likelihood == batch_likelihood.item()
