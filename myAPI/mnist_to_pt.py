import torch
# from myAPI.model import MNISTDataset

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
    
def pt_mnist(data ,path):
    data = data.to(torch.uint8)

    dims = data.shape
    magic_num = 8*2**8 + len(dims)

    flat = torch.flatten(data)
    
    magic_num_b = magic_num.to_bytes(4, "big")
    dims_b = b''.join(dim.to_bytes(4, "big") for dim in dims)
    data_b = bytes(flat.tolist())

    out = magic_num_b + dims_b + data_b

    with open(path, "wb") as f:
        f.write(out)

        
