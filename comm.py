import os
import torch
import torch.distributed as dist

def init_process_group(backend="nccl", master_addr=None, master_port=None, rank=None, world_size=None):
    # env override
    if master_addr is None:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if master_port is None:
        master_port = os.environ.get("MASTER_PORT", "29500")
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

def rank():
    return dist.get_rank()

def world_size():
    return dist.get_world_size()

def barrier():
    return dist.barrier()

def cleanup():
    if dist.is_intialized():
        dist.destroy_process_group()

def all_gather_tensor_on_gpu(tensor):
    # TODO: research this more (getting correct tensors?)
    ws = dist.get_world_size()
    # "create list of tensors with same shape as local for gathering"
    gather_list = [torch.empty_like(tensor) for _ in range(ws)] # TODO: empty_like
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=-1) # concat along head-output dim

# TODO: above all_gather has potential failure points, must research more
#   below is suggested by copilot:

# def all_gather_tensor_on_gpu(tensor):
#     # corrected and safer all-gather + concat on last dim
#     ws = dist.get_world_size()
#     if ws <= 0:
#         # not part of a group or single process -> return clone/identity
#         return tensor.clone()

#     # create list of correctly-sized tensors for gathering
#     gather_list = [torch.empty_like(tensor) for _ in range(ws)]

#     # handle complex dtypes if needed (optional)
#     if tensor.is_complex():
#         # gather real-view tensors and reconstruct
#         real_tensor = torch.view_as_real(tensor)
#         real_gather_list = [torch.empty_like(real_tensor) for _ in range(ws)]
#         dist.all_gather(real_gather_list, real_tensor)
#         gathered = torch.cat(real_gather_list, dim=-1)
#         return torch.view_as_complex(gathered)

#     # standard path
#     dist.all_gather(gather_list, tensor)
#     return torch.cat(gather_list, dim=-1)



def broadcast_object(obj, src=0):
    # "helper for python object boradcast via pickling"
    if dist.get_world_size() == 1:
        return obj
    obj_list =[obj if dist.get_rank() == src else None]
    # "simple scheme: use torch.distributed.broadcast_object_list"
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]