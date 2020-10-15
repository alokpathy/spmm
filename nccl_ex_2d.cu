#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>

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


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char** argv) {

  if(argc < 3) {
      std::cout << "Usage: ./nccl_ex_2d <datasize> <proccount>" << std::endl;
      return 0;
  }

  int n;
  int nprocs;
  n = atoi(argv[1]);
  nprocs = atoi(argv[2]);
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

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  int local_gpuid = rank % 3;
  // int local_gpuid = rank % 1;
  CUDACHECK(cudaSetDevice(local_gpuid));

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

  if (mpi_new_world == MPI_COMM_NULL) {
    MPI_Finalize();
    exit(0);
  }

  // CUDACHECK(cudaSetDevice(0));

  // float** sendbuff = (float**)malloc(ngpus * sizeof(float*));
  // float** recvbuff = (float**)malloc(ngpus * sizeof(float*));
  // cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*ngpus);
  // cudaEvent_t start[ngpus];
  // cudaEvent_t stop[ngpus];

  // float *h_data = new float[n]();
  double *h_data = new double[n]();
  for (int i = 0; i < n; i++) {
    h_data[i] = (double) (5.0);
  }

  // Initialize send/receive buffers, streams, and timers
  // float *sendbuff;
  // float *recvbuff;
  double *sendbuff;
  double *recvbuff;
  cudaEvent_t start[3]; // row, col, overall
  cudaEvent_t stop[3]; // row, col, overall

  CUDACHECK(cudaMalloc(&sendbuff, n * sizeof(double)));
  CUDACHECK(cudaMemcpy(sendbuff, h_data, n * sizeof(double), cudaMemcpyHostToDevice));

  CUDACHECK(cudaMalloc(&recvbuff, n * sizeof(double)));
  CUDACHECK(cudaMemset(recvbuff, 0, n * sizeof(double)));

  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*2);
  CUDACHECK(cudaStreamCreate(&s[0]));
  CUDACHECK(cudaStreamCreate(&s[1]));

  CUDACHECK(cudaEventCreate(&start[0]));
  CUDACHECK(cudaEventCreate(&start[1]));
  CUDACHECK(cudaEventCreate(&start[2]));
  CUDACHECK(cudaEventCreate(&stop[0]));
  CUDACHECK(cudaEventCreate(&stop[1]));
  CUDACHECK(cudaEventCreate(&stop[2]));

  // Create 2D process grid in MPI to bcast NCCL unique id's
  MPI_Comm mpi_row_comm, mpi_col_comm;

  MPICHECK(MPI_Comm_split( mpi_new_world, proc_row_id, rank, &mpi_row_comm ));  // processes with the same color are in the same new communicator 
  MPICHECK(MPI_Comm_split( mpi_new_world, proc_col_id, rank, &mpi_col_comm ));

  // Create NCCL unique id's (one per subcommunicator) and MPI_bcast them to every proc in subcommunicator.
  ncclComm_t comms[2]; // one comm for row, one comm for col

  // Create NCCL unique id's for process rows
  ncclUniqueId id_row;

  // Generating NCCL unique ID at one process and broadcasting it to all
  if (proc_col_id == 0) {
    ncclGetUniqueId(&id_row);
  }

  MPICHECK(MPI_Bcast((void *)&id_row, sizeof(id_row), MPI_BYTE, 0, mpi_row_comm));

  MPI_Barrier(mpi_new_world);

  // Initialize NCCL communicator.
  NCCLCHECK(ncclCommInitRank(&comms[0], procdim, id_row, proc_col_id));

  MPI_Barrier(mpi_new_world);

  // Create NCCL unique id's for process cols
  ncclUniqueId id_col;

  // Generating NCCL unique ID at one process and broadcasting it to all
  if (proc_row_id == 0) {
    ncclGetUniqueId(&id_col);
  }

  MPICHECK(MPI_Bcast((void *)&id_col, sizeof(id_col), MPI_BYTE, 0, mpi_col_comm));

  MPI_Barrier(mpi_new_world);

  // Initialize NCCL communicator.
  NCCLCHECK(ncclCommInitRank(&comms[1], procdim, id_col, proc_row_id));

  MPI_Barrier(mpi_new_world);

  CUDACHECK(cudaEventRecord(start[2], 0));
  
  // 2D SUMMA
  int trials = 20;
  float row_time = 0.0;
  float col_time = 0.0;

  for (int i = 0; i < procdim; i++) {

    CUDACHECK(cudaEventRecord(start[0], 0));

    // Call ncclBroadcast (ncclGroup* calls make this function as one ncclBroadcast call).
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < trials; j++) {
      NCCLCHECK(ncclBroadcast((const void*)sendbuff, (void*)recvbuff, n, ncclDouble, i, comms[0], 0));
    }
    NCCLCHECK(ncclGroupEnd());
    cudaDeviceSynchronize();
    CUDACHECK(cudaEventRecord(stop[0], 0));

    float row_bcast_time;
    CUDACHECK(cudaEventElapsedTime(&row_bcast_time, start[0], stop[0]));
    CUDACHECK(cudaEventSynchronize(stop[0]));
    row_time += row_bcast_time / 1000;

    CUDACHECK(cudaEventRecord(start[1], 0));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < trials; j++) {
      NCCLCHECK(ncclBroadcast((const void*)sendbuff, (void*)recvbuff, n, ncclDouble, i, comms[1], 0));
    }
    NCCLCHECK(ncclGroupEnd());
    cudaDeviceSynchronize();
    CUDACHECK(cudaEventRecord(stop[1], 0));

    float col_bcast_time;
    CUDACHECK(cudaEventElapsedTime(&col_bcast_time, start[1], stop[1]));
    CUDACHECK(cudaEventSynchronize(stop[1]));
    col_time += col_bcast_time / 1000;
  }

  CUDACHECK(cudaEventRecord(stop[2], 0));
  CUDACHECK(cudaEventSynchronize(stop[0]));
  CUDACHECK(cudaEventSynchronize(stop[1]));
  CUDACHECK(cudaEventSynchronize(stop[2]));

  // Collect timings and verify broadcast worked.
  // float *h_recvbuff = new float[n]();
  double *h_recvbuff = new double[n]();
  float time_row, time_col, gpu_time;
  CUDACHECK(cudaEventElapsedTime(&time_row, start[0], stop[0]));
  CUDACHECK(cudaEventElapsedTime(&time_col, start[1], stop[1]));
  CUDACHECK(cudaEventElapsedTime(&gpu_time, start[2], stop[2]));
  time_row /= 1000; // seconds
  time_col /= 1000; // seconds
  gpu_time /= 1000; // seconds

  cudaMemcpy(h_recvbuff, recvbuff, n * sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    if (h_recvbuff[i] != (double)(5.0)) {
      std::cout << "bcast error " << h_recvbuff[i] << " " << (procdim - 1) << std::endl;
      exit(0);
    }
  }

  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " gpu_time: " << gpu_time << " bw: " << ((2 * procdim * n * sizeof(double) * trials) / gpu_time) << std::endl;
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " row_time: " << row_time << " bw: " << ((procdim * n * sizeof(double) * trials) / row_time) << std::endl;
  std::cout << "rank: " << rank << " size: " << (n * sizeof(double)) << " col_time: " << col_time << " bw: " << ((procdim * n * sizeof(double) * trials) / col_time) << std::endl;

  // Freeing device memory
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // Finalizing NCCL
  ncclCommDestroy(comms[0]);
  ncclCommDestroy(comms[1]);

  // Finalizing MPI
  MPICHECK(MPI_Finalize());

  // printf("[MPI Rank %d] Success \n", rank);
  return 0;
}
