import torch

from causal_meta.runners.metrics.graph import structural_interventional_distance


def test_sid_zero_for_identical_graphs() -> None:
    # 0 -> 1
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    samples = torch.tensor([[[[0, 1], [0, 0]]]]).float()  # (S=1, B=1, N=2, N=2)

    sid = structural_interventional_distance(target, samples)
    assert sid.shape == (1,)
    assert sid[0].item() == 0.0


def test_sid_counts_missed_and_false_effects() -> None:
    # True: 0 -> 1
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    # Pred: 1 -> 0 (reversed)
    samples = torch.tensor([[[[0, 0], [1, 0]]]]).float()

    # Pairs are (0,1) and (1,0). Both are wrong => SID=2.
    sid = structural_interventional_distance(target, samples)
    assert sid[0].item() == 2.0


def test_sid_detects_backdoor_adjustment_failure() -> None:
    # True: 1 -> 0, 1 -> 2, 0 -> 2 (confounding + causal effect 0->2)
    target = torch.tensor(
        [
            [
                [0, 0, 1],
                [1, 0, 1],
                [0, 0, 0],
            ]
        ]
    ).float()
    # Pred: only 0 -> 2 (misses confounding and 1's effects)
    samples = torch.tensor(
        [
            [
                [
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        ]
    ).float()

    # Expected errors:
    # - (0,2): effect exists, but Pa_est(0)=∅ doesn't block backdoor 0<-1->2 => error
    # - (1,0): missed effect => error
    # - (1,2): missed effect => error
    sid = structural_interventional_distance(target, samples)
    assert sid[0].item() == 3.0

