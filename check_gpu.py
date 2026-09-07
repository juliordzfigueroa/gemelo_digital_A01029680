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