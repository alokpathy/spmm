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


static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  // int size = 32*1024*1024;
  // int size = 32*1024*32768; // 2^15
  // int size = 32*1024*8192; // 2^13
  if(argc < 3) {
      std::cout << "Please specify the number of vertices (data size) in thousands and number of gpus per node";
      return 0;
  }

  int size;
  int nDev;
  size = atoi(argv[1]);
  nDev = atoi(argv[2]);
  size *= 1000;

  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  cudaEvent_t start[nDev];
  cudaEvent_t stop[nDev];

  float *h_sequence = new float[size]();
  for (int i = 0; i < size; i++) {
    h_sequence[i] = 1.0f;
  }
  //picking GPUs based on localRank
  for (int i = 0; i < nDev; ++i) {
    // CUDACHECK(cudaSetDevice(localRank*nDev + i));
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    // CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff[i], h_sequence, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));

    CUDACHECK(cudaEventCreate(&start[i]));
    CUDACHECK(cudaEventCreate(&stop[i]));
  }


  ncclUniqueId id;
  ncclComm_t comms[nDev];

  //generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  MPI_Barrier(MPI_COMM_WORLD);

  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++) {
     // CUDACHECK(cudaSetDevice(localRank*nDev + i));
     CUDACHECK(cudaSetDevice(i));
     NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
     CUDACHECK(cudaEventRecord(start[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  clock_t begin = clock();
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++)
     // NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
     //       comms[i], s[i]));
     NCCLCHECK(ncclBroadcast((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, 0,
           comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());


  //synchronizing on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaStreamSynchronize(s[i]));
  }
  // MPI_Barrier(MPI_COMM_WORLD);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  // std::cout << "rank: " << myRank << " time: " << time_spent << " bw: " << (size / time_spent) << std::endl;

  for (int i = 0; i < nDev; i++) {
      // CUDACHECK(cudaSetDevice(localRank*nDev + i));
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaEventRecord(stop[i], s[i]));
  }

  float *h_recvbuff = new float[size]();
  for (int i = 0; i < nDev; i++) {
    // CUDACHECK(cudaSetDevice(localRank*nDev + i));
    CUDACHECK(cudaSetDevice(i));

    float time;
    CUDACHECK(cudaEventElapsedTime(&time, start[i], stop[i]));
    time /= 1000; // seconds

    cudaMemcpy(h_recvbuff, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "rank: " << myRank << " gpu: " << i << " nums: " << h_recvbuff[0] << " " << h_recvbuff[1] << " " << h_recvbuff[2] << " size: " << size << " time: " << time << " bw: " << (size / time) << std::endl;
  } 



  //freeing device memory
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for (int i=0; i<nDev; i++) {
     ncclCommDestroy(comms[i]);
  }


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
