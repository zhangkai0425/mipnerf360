import torch
import torch.nn as nn

def distillation(student,teacher):
    """implementation of distillation network"""
    