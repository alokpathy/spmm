# Standalone 3D distributed SpMM implementation
# Largely borrowed from CAGNET

import argparse
import math
import torch
import os

import torch.distributed as dist
import torch.multiprocessing as mp

from sparse_coo_tensor_cpp import spmm_gpu

def split3dspmm_sparse(adj_matrix, inputs, rank, row, col, rank_c, size, acc_per_rank, 
                            row_groups, col_groups, c_groups, 
                            height, middim, width):

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    proc_c = proc_c_size(size)

    # Compute the height, middim, and width for the local spmm
    height_per_proc = height // proc_row
    width_per_proc  = width // proc_col

    middim_per_proc = middim // (proc_col * proc_c)
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    # Handle boundary conditions if this rank is in the last process row or column
    if row == proc_row - 1:
        height_per_proc = height - height_per_proc * (proc_row - 1)

    if col == proc_col - 1:
        width_per_proc = width - width_per_proc * (proc_col - 1)

    # Initialize output matrix for local spmm
    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=device).fill_(0)

    # Determine column size to split output matrix after local spmm's
    chunk_sizes_col = []
    chunk_len = inputs.size(1) // proc_c
    for i in range(proc_c):
        if i == proc_c - 1:
            chunk_sizes_col.append(inputs.size(1) - chunk_len * (proc_c - 1))
        else:
            chunk_sizes_col.append(chunk_len)

    for k in range(proc_col):

        row_src_rank = row * (proc_col * proc_c) + rank_c + k * proc_c # src rank for row bcast
        col_src_rank = col * proc_row + rank_c + k * proc_c * proc_row # src rank for col bcast

        # Determine middle dimension of matrices for local spmm
        middim_per_col = middim // proc_col
        if k == proc_col - 1:
            middim_per_col = middim - middim_per_col * (proc_col - 1)
        
        middim_per_proc = middim_per_col // proc_c
        if rank_c == proc_c - 1:
            middim_per_proc = middim_per_col - middim_per_proc * (proc_c - 1)

        if row_src_rank == rank:
            acol_indices_len = torch.cuda.LongTensor(
                                            [adj_matrix.indices().contiguous()[0].size(0)], 
                                            device=device)
            acol_values_len = torch.cuda.LongTensor([adj_matrix.values().contiguous().size(0)],
                                                    device=device)
        else:
            acol_indices_len = torch.cuda.LongTensor([0], device=device)
            acol_values_len = torch.cuda.LongTensor([0], device=device)
        
        # Broadcast nnz across rows (necessary for row bcast)
        dist.broadcast(acol_indices_len, row_src_rank, row_groups[row][rank_c])

        acol_indices_len = acol_indices_len.item() # nnz
        acol_values_len = acol_indices_len

        # Initialize new empty matrix for row bcast if this rank is not the src rank 
        if row_src_rank == rank:
            acol_indices = adj_matrix.indices().contiguous().long()
            acol_values = adj_matrix.values().contiguous().float()
        else:
            acol_indices = torch.cuda.LongTensor(2, acol_indices_len, device=device).fill_(0)
            acol_values = torch.cuda.FloatTensor(acol_values_len, device=device).fill_(0)

        acol = torch.cat((acol_indices.float(), acol_values.unsqueeze(0)), dim=0)

        # Row bcast 
        dist.broadcast(acol.contiguous(), row_src_rank, row_groups[row][rank_c])

        acol_indices = acol[:2].long()
        acol_values = acol[2].squeeze(0)

        if row_src_rank == rank:
            acol = adj_matrix
        else:
            acol = sparse_coo_tensor_gpu(acol_indices, acol_values, 
                                            torch.Size([height_per_proc, middim_per_proc]))

        # Initialize new empty matrix for col bcast if this rank is not the src rank 
        if col_src_rank == rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=device)

        # Col bcast
        brow = brow.contiguous()
        dist.broadcast(brow, col_src_rank, col_groups[col][rank_c])

        # Local spmm
        spmm_gpu(acol_indices[0].int(), acol_indices[1].int(), acol_values, 
                        height_per_proc, middim_per_proc, brow, z_loc)

    z_loc = z_loc.contiguous()

    # All-Reduce across third process grid dimension
    dist.all_reduce(z_loc, group=c_groups[int(rank // proc_c)])

    # Split the output of the all-reduce across third process grid dimension
    # Each rank only keeps its submatrix
    z_loc = torch.split(z_loc, chunk_sizes_col, dim=1)
    z_loc = z_loc[rank_c].contiguous()

    return z_loc

def rank_to_devid(rank, acc_per_rank):
    return rank % acc_per_rank

def proc_row_size(size):
    cube_root = int(size ** (1./ 3.))
    if cube_root ** 3 == size:
        return cube_root
    elif (cube_root + 1) ** 3 == size:
        return cube_root + 1
    else:
        print(f"CUBE ROOT ERROR")

def proc_col_size(size):
    cube_root = int(size ** (1./ 3.))
    if cube_root ** 3 == size:
        return cube_root
    elif (cube_root + 1) ** 3 == size:
        return cube_root + 1
    else:
        print(f"CUBE ROOT ERROR")

def proc_c_size(size):
    cube_root = int(size ** (1./ 3.))
    if cube_root ** 3 == size:
        return cube_root
    elif (cube_root + 1) ** 3 == size:
        return cube_root + 1
    else:
        print(f"CUBE ROOT ERROR")

def get_proc_groups(rank, size, group):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    proc_c = proc_c_size(size)

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid
    
    row_groups = []
    col_groups = []
    c_groups = []

    row_procs = []
    col_procs = []
    c_procs = []

    for i in range(proc_row):
        row_groups_c = []
        row_procs_c = []
        for j in range(proc_c):
            proc_start = i * proc_col * proc_c + j
            proc_end = (i + 1) * proc_col * proc_c + j
            row_groups_c.append(dist.new_group(list(range(proc_start, proc_end, proc_c))))
            row_procs_c.append(list(range(proc_start, proc_end, proc_c)))
        row_groups.append(row_groups_c)
        row_procs.append(row_procs_c)

    for i in range(proc_col):
        col_groups_c =[]
        col_procs_c =[]
        for j in range(proc_c):
            proc_start = i * proc_c + j
            proc_end = proc_row * proc_col * proc_c + i * proc_c + j
            col_groups_c.append(dist.new_group(list(range(proc_start, proc_end, proc_c * proc_col))))
            col_procs_c.append(list(range(proc_start, proc_end, proc_c * proc_col)))
        col_groups.append(col_groups_c)
        col_procs.append(col_procs_c)

    for i in range(0, size, proc_c):
        c_groups.append(dist.new_group(list(range(i, i + proc_c))))
        c_procs.append(list(range(i, i + proc_c)))

    return row_groups, col_groups, c_groups

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim, size):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    proc_c = proc_c_size(size)

    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices = vtx_indices[:proc_c]
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def twod_partition(rank, size, inputs, adj_matrix, device):
    node_count = inputs.size(0)
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    proc_c = proc_c_size(size)

    # n_per_proc = math.ceil(float(node_count) / proc_row)
    n_per_proc = node_count // proc_row

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    
    am_partitions = None
    am_pbyp = None

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1, size)

        proc_node_count = vtx_indices[rank_col + 1] - vtx_indices[rank_col]
        am_pbyp, _ = split_coo(am_partitions[rank_col], node_count, n_per_proc, 0, size)
        for i in range(len(am_pbyp)):
            if i == proc_row - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

        # input_rowparts = torch.split(inputs, math.ceil(float(inputs.size(0)) / proc_row), dim=0)
        inputs_per_row = inputs.size(0) // proc_row
        inputs_per_col = inputs.size(1) // proc_col
        chunks_per_row = []
        chunks_per_col = []
        for i in range(proc_row):
            if i == proc_row - 1:
                chunks_per_row.append(inputs.size(0) - inputs_per_row * (proc_row - 1))
            else:
                chunks_per_row.append(inputs_per_row)
        for i in range(proc_col):
            if i == proc_col - 1:
                chunks_per_col.append(inputs.size(1) - inputs_per_col * (proc_col - 1))
            else:
                chunks_per_col.append(inputs_per_col)

        # input_rowparts = torch.split(inputs, math.ceil(float(inputs.size(0)) / proc_row), dim=0)
        input_rowparts = torch.split(inputs, chunks_per_row, dim=0)
        input_partitions = []
        for i in input_rowparts:
            # input_partitions.append(torch.split(i, math.ceil(float(inputs.size(1)) / proc_col), 
            #                            dim=1))
            input_partitions.append(torch.split(i, chunks_per_col, dim=1))

        adj_matrix_loc = am_pbyp[rank_row]
        inputs_loc = input_partitions[rank_row][rank_col]

    return inputs_loc, adj_matrix_loc, am_pbyp

def threed_partition_loc(rank, size, inputs, adj_matrix, height, width,  device):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    proc_c = proc_c_size(size)

    n_per_proc = width // proc_c

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid
    
    am_partitions, vtx_indices = split_coo(adj_matrix, width, n_per_proc, 1, size)

    for i in range(len(am_partitions)):
        if i == proc_c - 1:
            last_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], torch.ones(am_partitions[i].size(1)), 
                                                            size=(height, last_node_count),
                                                            requires_grad=False)

        else:
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], torch.ones(am_partitions[i].size(1)), 
                                                            size=(height, n_per_proc),
                                                            requires_grad=False)

    inputs_per_row = inputs.size(0) // proc_c
    chunks_per_row = []
    for i in range(proc_c):
        if i == proc_row - 1:
            chunks_per_row.append(inputs.size(0) - inputs_per_row * (proc_c - 1))
        else:
            chunks_per_row.append(inputs_per_row)

    input_rowparts = torch.split(inputs, chunks_per_row, dim=0)

    adj_matrix_loc = am_partitions[rank_c]
    inputs_loc = input_rowparts[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)

    return inputs_loc, adj_matrix_loc 


def main(mata_indices_path, matb_path, acc_per_rank):
    # Load matrices as pytorch tensors
    mata_indices = torch.load(mata_indices_path)
    matb = torch.load(matb_path)

    # Initialize distributed environment
    mp.set_start_method('spawn', force=True)
    if "OMPI_COMM_WORLD_RANK" in os.environ.keys():
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    size = dist.get_world_size()
    print("Processes: " + str(size))

    devid = rank_to_devid(rank, acc_per_rank)
    device = torch.device('cuda:{}'.format(devid))
    torch.cuda.set_device(device)
    curr_devid = torch.cuda.current_device()
    print(f"curr_devid: {curr_devid}", flush=True)
    devcount = torch.cuda.device_count()

    # Initialize all process groups
    group = dist.new_group(list(range(size)))
    row_groups, col_groups, c_groups = get_proc_groups(rank, size, group)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    proc_c = proc_c_size(size)

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid

    if rank_row >= proc_row or rank_col >= proc_col or rank_c >= proc_c:
        return

    # Partition both input matrices across process grid and get local mata and matb copies
    print(f"Before partitioning...", flush=True)
    matb_loc, mata_loc, _ = twod_partition(rank, size, matb, mata_indices, device)
    mata_loc = mata_loc.coalesce()

    matb_loc, mata_loc = threed_partition_loc(rank, size, matb_loc, mata_loc.indices(), \
                                                                mata_loc.size(0), mata_loc.size(1), device)
    print(f"After partitioning...", flush=True)

    mata_loc = mata_loc.coalesce()

    matb_loc = matb_loc.to(device)
    mata_loc = mata_loc.to(device)

    dist.barrier(group)

    # Call 3D distributed SpMM algorithm
    z = split3dspmm_sparse(mata_loc, matb_loc, rank, rank_row, rank_col, rank_c, size, acc_per_rank,  \
                                                row_groups, col_groups, c_groups, \
                                                matb.size(0), matb.size(0), matb.size(1))

    print(z)
    print(torch.sum(z))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mata-indices", type=str)
    parser.add_argument("--matb", type=str)
    parser.add_argument("--accperrank", type=int)

    args = parser.parse_args()
    print(args)

    mata_indices_path = args.mata_indices
    matb_path = args.matb
    acc_per_rank = args.accperrank

    main(mata_indices_path, matb_path, acc_per_rank)
