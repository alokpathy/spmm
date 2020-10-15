#include <cmath>
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <cstring>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDA_AWARE

int main(int argc, char** argv) {

  if(argc < 4) {
      std::cout << "Usage: ./mpi_ex_15d <datasize> <gpus/resource_set> <proccount>" << std::endl;
      return 0;
  }

  int n, nprocs, ngpus;
  n = atoi(argv[1]);
  ngpus = atoi(argv[2]);
  nprocs = atoi(argv[3]);
  n *= 1000;

  int rank, size;

  // Initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  printf("rank: %d, Hostname: %s\n", rank, hostname);

#ifdef CUDA_AWARE
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  int local_gpuid = rank % ngpus;
  printf("rank: %d local_gpuid: %d\n", rank, local_gpuid); fflush(stdout);
  CUDACHECK(cudaSetDevice(local_gpuid));
#endif

  // Obtain the group of processes in the world communicator
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // Remove all unnecessary ranks
  MPI_Group new_group;
  int ranges[1][3];
  ranges[0][0] = nprocs;
  ranges[0][1] = size - 1;
  ranges[0][2] = 1;
  int range_count = nprocs < size;
  
  MPI_Group_range_excl(world_group, range_count, ranges, &new_group);

  int procdim = (int)std::sqrt((float)nprocs);
  int proc_col_id = rank % procdim;
  int proc_row_id = rank / procdim;


  // Create a new communicator
  MPI_Comm mpi_new_world;
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &mpi_new_world);

  double *h_data = new double[n]();
  for (int i = 0; i < n; i++) {
    h_data[i] = (double) (5.0);
  }

  // Initialize send/receive buffers, streams, and timers
  double *sendbuff, *recvbuff;

#ifdef CUDA_AWARE
  CUDACHECK(cudaMalloc(&sendbuff, n * sizeof(double)));
  CUDACHECK(cudaMemcpy(sendbuff, h_data, n * sizeof(double), cudaMemcpyHostToDevice));

  CUDACHECK(cudaMalloc(&recvbuff, n * procdim * sizeof(double)));
#else
  sendbuff = h_data;
#endif

  // 1D bcast on MPI_COMM_WORLD
  double bcast1d_start = MPI_Wtime();
  MPICHECK(MPI_Bcast((void *)sendbuff, n, MPI_DOUBLE, 0, MPI_COMM_WORLD));
  double bcast1d_end = MPI_Wtime();
  double bcast1d_total = bcast1d_end - bcast1d_start;
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " bcast1d_time: " << bcast1d_total << " bw: " << ((n * sizeof(double)) / bcast1d_total) << std::endl;
  
  if (mpi_new_world == MPI_COMM_NULL) {
    MPI_Finalize();
    exit(0);
  }

  // Create 2D process grid in MPI to bcast NCCL unique id's
  MPI_Comm mpi_row_comm, mpi_col_comm;

  MPICHECK(MPI_Comm_split( mpi_new_world, proc_row_id, rank, &mpi_row_comm ));  // processes with the same color are in the same new communicator 
  MPICHECK(MPI_Comm_split( mpi_new_world, proc_col_id, rank, &mpi_col_comm ));

  MPI_Barrier(mpi_new_world);

  // 1.5D MPI-FAUN
  double row_time = 0.0;
  double col_time = 0.0;
  
  double start_time = MPI_Wtime(); 

  double col_time_start = MPI_Wtime();
  MPICHECK(MPI_Allgather((void *)sendbuff, n, MPI_DOUBLE, (void *)recvbuff, n, MPI_DOUBLE, mpi_col_comm));
  double col_time_stop = MPI_Wtime();
  col_time = col_time_stop - col_time_start;

  double row_time_start = MPI_Wtime();
  MPICHECK(MPI_Reduce_scatter_block((void *)recvbuff, (void *)sendbuff, n, MPI_DOUBLE, MPI_SUM, mpi_row_comm));
  double row_time_stop = MPI_Wtime();
  row_time = row_time_stop - row_time_start;

  double end_time = MPI_Wtime(); 
  double total_time = end_time - start_time;


  // Collect timings and verify broadcast worked.
  double *h_recvbuff = new double[n]();
#ifdef CUDA_AWARE
  cudaMemcpy(h_recvbuff, sendbuff, n * sizeof(double), cudaMemcpyDeviceToHost);
#else
  memcpy(h_recvbuff, sendbuff, n * sizeof(double));
#endif
  for (int i = 0; i < n; i++) {
    if (h_recvbuff[i] != (double)(5.0 * procdim)) {
      std::cout << "mpi-faun error " << h_recvbuff[i] << " " << (procdim - 1) << std::endl;
      exit(0);
    }
  }

#ifdef CUDA_AWARE
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " gpu_time: " << total_time << " bw: " << ((n * sizeof(double) + procdim * n * sizeof(double)) / total_time) << std::endl;
#else
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " cpu_time: " << total_time << " bw: " << ((n * sizeof(double) + procdim * n * sizeof(double)) / total_time) << std::endl;
#endif
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " row_time: " << row_time << " bw: " << ((n * sizeof(double)) / row_time) << std::endl;
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " col_time: " << col_time << " bw: " << ((procdim * n * sizeof(double)) / col_time) << std::endl;

#ifdef CUDA_AWARE
  // Freeing device memory
  CUDACHECK(cudaFree(sendbuff));
#else
  delete[] sendbuff;
#endif

  // Finalizing MPI
  MPICHECK(MPI_Finalize());

  // printf("[MPI Rank %d] Success \n", rank);
  return 0;
}
