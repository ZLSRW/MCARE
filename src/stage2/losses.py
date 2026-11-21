"""Objective loss functions for structured need-contrast and specialist pi_E training."""
import torch
import torch.nn.functional as F


def dpo(policy, reference, beta=.1):
    return -F.logsigmoid(beta * ((policy[0] - policy[1]) - (reference[0] - reference[1])))


def reversal(logodds_l, logodds_a, margin=1.):
    return F.relu(margin - (logodds_l - logodds_a))


def distill(student, teacher, temperature=1.):
    q = F.softmax(teacher.detach() / temperature, dim=0)
    return F.kl_div(F.log_softmax(student / temperature, dim=0), q, reduction='sum') * temperature**2
