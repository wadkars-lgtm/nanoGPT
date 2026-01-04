import sys
import torch

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_ckpt.py <path_to_checkpoint>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("Checkpoint:", ckpt_path)
    print("iter_num:", ckpt.get("iter_num"))
    print("best_val_loss:", ckpt.get("best_val_loss"))

if __name__ == "__main__":
    main()
