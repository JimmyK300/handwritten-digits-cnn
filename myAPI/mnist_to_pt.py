import torch

def mnist_pt(path):
    with open(path, "rb") as f:
        data = f.read()
        
        dim_count = data[3]
        
        dims = []
        for i in range(1, dim_count + 1):
            dim_val = int.from_bytes(data[4*i : 4*(i+1)], byteorder="big")
            dims.append(dim_val)

        header_size = 4 + (4 * dim_count)
        img_data = bytearray(data[header_size:])

        output = torch.frombuffer(img_data, dtype=torch.uint8)
        
        return output.reshape(*dims)