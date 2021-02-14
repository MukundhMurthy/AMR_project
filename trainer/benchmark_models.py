import torch

if __name__ == '__main__':
    model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()