import torch

def test_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
        
        # Crée un tensor et fait une opération simple sur GPU
        x = torch.tensor([1.0, 2.0, 3.0]).to(device)
        y = x * 2
        print(f"Tenseur x sur GPU : {x}")
        print(f"Tenseur y = x * 2 sur GPU : {y}")
    else:
        print("GPU non disponible, utilisation CPU")

if __name__ == "__main__":
    test_gpu()
