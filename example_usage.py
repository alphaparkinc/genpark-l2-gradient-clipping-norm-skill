import sys
from client import GradientClipper

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def run():
    print(">>> Demonstrating L2 Gradient Clipping...")
    clipper = GradientClipper(max_norm=5.0)

    # Gradients with norm = sqrt(3^2 + 4^2) = 5.0 (within limit)
    g1, norm1, clipped1 = clipper.clip_gradients([3.0, 4.0])
    assert clipped1 is False
    assert norm1 == 5.0

    # Gradients with norm = sqrt(6^2 + 8^2) = 10.0 (exceeds 5.0)
    g2, norm2, clipped2 = clipper.clip_gradients([6.0, 8.0])
    assert clipped2 is True
    print(f"Original norm: {norm2:.2f}, Clipped gradients: {g2}")
    clipped_norm = (g2[0]**2 + g2[1]**2) ** 0.5
    assert abs(clipped_norm - 5.0) < 1e-4
    print("[PASS] L2 Gradient Clipping verified.")

if __name__ == "__main__":
    run()
