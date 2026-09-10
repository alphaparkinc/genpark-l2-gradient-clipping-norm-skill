import math

class GradientClipper:
    """
    Global L2 Norm Gradient Clipper.
    Scales down gradients proportionally if global Euclidean norm exceeds threshold.
    """
    def __init__(self, max_norm=1.0, eps=1e-6):
        self.max_norm = max_norm
        self.eps = eps

    def clip_gradients(self, gradients):
        total_norm = math.sqrt(sum(g ** 2 for g in gradients))
        if total_norm > self.max_norm:
            scale = self.max_norm / (total_norm + self.eps)
            return [g * scale for g in gradients], total_norm, True
        return list(gradients), total_norm, False
