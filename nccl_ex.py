# Standalone 1.5D distributed SpMM implementation
# Largely borrowed from CAGNET

import argparse
import math
import os
import time
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

from sparse_coo_tensor_cpp import spmm_gpu

comp_time = dict()
comm_time = dict()
bcast_comm_time = dict()
reduce_comm_time = dict()
timing = False

def start_time(group, rank):
    if not timing:
        return 0.0
    if group is not None:
        torch.cuda.synchronize(device=device)
    tstart = 0.0
    if rank == 0:
        tstart = time.time()
    return tstart

def stop_time(group, rank, tstart):
    if not timing:
        return 0.0
    if group is not None:
        torch.cuda.synchronize(device=device)
    tstop = 0.0
    if rank == 0:
        tstop = time.time()
    return tstop - tstart

def dspmm(node_count, am_partitions, inputs, rank, size, replication, row_groups, col_groups, group, device):
    global comm_time
    global comp_time
    global bcast_comm_time
    global reduce_comm_time

    n_per_proc = math.ceil(float(node_count) / (size / replication))

    z_loc = torch.cuda.FloatTensor(am_partitions[0].size(0), inputs.size(1), device=device).fill_(0)

    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=device).fill_(0)

    rank_c = rank // replication # effectively row-rank 
    rank_col = rank % replication

    stages = size // (replication ** 2)
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages

    for i in range(stages):
        # Compute src rank in bcast 
        q = (rank_col * (size // (replication ** 2)) + i) * replication + rank_col

        q_c = q // replication

        am_partid = rank_col * (size // replication ** 2) + i

        # If this rank is the src rank for bcast, set inputs_recv to the local matrix
        # Else, instantiate a new empty matrix
        if q == rank:
            inputs_recv = inputs.clone()
        elif q_c == size // replication - 1:
            inputs_recv = torch.cuda.FloatTensor(am_partitions[am_partid].size(1), inputs.size(1), device=device).fill_(0)

        inputs_recv = inputs_recv.contiguous()
        tstart_comm = start_time(col_groups[rank_col], rank)
        dist.broadcast(inputs_recv, src=q, group=col_groups[rank_col])
        dur = stop_time(col_groups[rank_col], rank, tstart_comm)

        comm_time[rank] += dur
        bcast_comm_time[rank] += dur

        tstart_comp = start_time(col_groups[rank_col], rank)

        spmm_gpu(am_partitions[am_partid].indices()[0].int(), am_partitions[am_partid].indices()[1].int(), 
                        am_partitions[am_partid].values(), am_partitions[am_partid].size(0), 
                        am_partitions[am_partid].size(1), inputs_recv, z_loc)

        dur = stop_time(col_groups[rank_col], rank, tstart_comp)
        comp_time[rank] += dur

    z_loc = z_loc.contiguous()

    tstart_comm = start_time(row_groups[rank_c], rank)
    dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=row_groups[rank_c])
    dur = stop_time(row_groups[rank_c], rank, tstart_comm)

    comm_time[rank] += dur
    reduce_comm_time[rank] += dur

    return z_loc

def rank_to_devid(rank, acc_per_rank):
    return rank % acc_per_rank

def get_proc_groups(rank, size, replication):
    rank_c = rank // replication
     
    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

def oned_partition(rank, size, inputs, adj_matrix, replication, device):
    node_count = inputs.size(0)
    # n_per_proc = math.ceil(float(node_count) / size)
    n_per_proc = math.ceil(float(node_count) / (size / replication))

    am_partitions = None
    am_pbyp = None

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)

        print(f"rank: {rank} replication: {replication} rank_c: {rank_c} len_vtxind: {len(vtx_indices)}")
        proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        am_pbyp, _ = split_coo(am_partitions[rank_c], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            if i == size // replication - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)

        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def main(mata_indices_path, k, acc_per_rank, replication):
    # Load matrices as pytorch tensors
    mata_indices = torch.load(mata_indices_path)
    if not isinstance(mata_indices, torch.Tensor): # if Reddit/Cora
        mata_indices = mata_indices[0].edge_index
    print(mata_indices)

    node_count = torch.max(mata_indices[0]) + 1
    matb = torch.rand(node_count, k)  

    # Initialize process groups
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

    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size, replication) 

    rank_c = rank // replication
    if rank_c >= (size // replication):
        return

    # Partition both input matrices across process grid and get local mata and matb copies
    matb_loc, mata_loc, mata_pbyp = oned_partition(rank, size, matb, mata_indices, replication, device)

    mata_loc = mata_loc.to(device)
    matb_loc = matb_loc.to(device)
    for i in range(len(mata_pbyp)):
        mata_pbyp[i] = mata_pbyp[i].t().coalesce().to(device)
    mata_loc = mata_loc.coalesce()

    comm_time[rank] = 0.0
    comp_time[rank] = 0.0
    bcast_comm_time[rank] = 0.0
    reduce_comm_time[rank] = 0.0

    dist.barrier(group)

    if rank == 0:
        summa_start_time = time.time()

    # Call 1.5D distributed SpMM algorithm
    z = dspmm(mata_loc.size(0), mata_pbyp, matb_loc, rank, size, replication, \
                            row_groups, col_groups, group, device)

    dist.barrier(group)
    if rank == 0:
        print(f"summa_time: {time.time() - summa_start_time}")
        print(f"rank: {rank} comm_time: {comm_time[rank]}")
        print(f"rank: {rank} comp_time: {comp_time[rank]}")
        print(f"rank: {rank} bcast_comm_time: {bcast_comm_time[rank]}")
        print(f"rank: {rank} reduce_comm_time: {reduce_comm_time[rank]}")
        print(f"rank: {rank} {outputs}")

    print(z)
    print(torch.sum(z))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mata-indices", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--accperrank", type=int)
    parser.add_argument("--replication", type=int)
    parser.add_argument("--timing", type=str)

    args = parser.parse_args()
    print(args)

    mata_indices_path = args.mata_indices
    acc_per_rank = args.accperrank
    replication = args.replication
    timing = args.timing == "True"

    main(mata_indices_path, args.k, acc_per_rank, replication)
