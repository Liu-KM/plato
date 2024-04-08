import torch

def convert_to_low_rank(r,weights):


    U,S,V = torch.svd(weights)

    U_truncated = U[:,:r]
    S_truncated = S[:r]
    V_truncated = V[:,:r]
    print("U_truncated:",U_truncated.shape)
    print("S_truncated:",S_truncated.shape)
    print("V_truncated:",V_truncated.shape)

    lora_A = U_truncated.t()
    lora_B = V_truncated@torch.diag(S_truncated).t()

    # # A = U ; B = V@S   
    # lora_A = (U_truncated).t()
    # lora_B = torch.diag(S_truncated)@V_truncated
    print(lora_A.shape, lora_B.shape)
    result = lora_B@lora_A
    print(result.shape)


matrix = torch.rand(1024,1024)
convert_to_low_rank(8,matrix)