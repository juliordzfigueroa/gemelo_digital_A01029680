"""CUDA availability check for the training machine.

Confirms that PyTorch sees the GPU and that it can actually run kernels on it.
The matrix multiplication at the end is not decorative: torch.cuda.is_available()
can return True while kernel launches still fail, so the only reliable check is
running a real operation.

Worth keeping because PyTorch and CUDA installations on Windows break quietly:
after an NVIDIA driver update or a reinstall inside the venv, training silently
falls back to CPU and simply takes far longer, with no error to point at.

Usage:
    python check_gpu.py

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Added the standard file header.
"""

import torch

print("PyTorch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Version CUDA (build):", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
    print("Numero de GPUs:", torch.cuda.device_count())
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM total: {props.total_memory / 1024**3:.1f} GB")

    # Prueba real: operacion de tensores en la GPU
    x = torch.rand(5000, 5000, device="cuda")
    y = torch.rand(5000, 5000, device="cuda")
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print("Multiplicacion de matrices en GPU: OK")
    print("Tensor resultante esta en:", z.device)
else:
    print("CUDA NO disponible - algo salio mal, revisar instalacion")