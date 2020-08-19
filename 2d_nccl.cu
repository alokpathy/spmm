#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

#include "mpi.h"
#include <nccl.h>

#define DEVICES 6

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    printf("Test CUDA failure %s:%d '%s'\n",    \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Test NCCL failure %s:%d '%s'\n",    \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
  }                                                 \
} while(0)

using namespace std;

void testBcast(ncclUniqueId id, ncclComm_t *comms, int n, int ngpus) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *h_data = new double[n];
    double **d_data = new double*[ngpus];
    for (int i = 0; i < ngpus; i++) {
        // CUDACHECK(cudaSetDevice(i));
        // CUDACHECK(cudaMalloc(&d_data[i], n * sizeof(double)));
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], n * sizeof(double));
    }

    cudaSetDevice(0);
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            h_data[i] = (double) i;
        }
        random_shuffle(h_data, h_data + n);

        cudaMemcpy(d_data[0], h_data, n * sizeof(double), cudaMemcpyHostToDevice);
    }

    // only works for ngpus=1
    cudaEvent_t start[DEVICES];
    cudaEvent_t stop[DEVICES];

    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
        cudaDeviceSynchronize();
    }
    cudaStreamSynchronize(cudaStreamDefault);


    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        cudaEventRecord(start[i]);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        NCCLCHECK(ncclBroadcast(d_data[0], d_data[i], n * sizeof(double), ncclDouble, 0, comms[i], cudaStreamDefault));
    }
    NCCLCHECK(ncclGroupEnd());

    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    cudaStreamSynchronize(cudaStreamDefault);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        cudaEventRecord(stop[i]);
        cudaEventSynchronize(stop[i]);
    }

    for (int i = 0; i < ngpus; i++) {
        float time;
        cudaSetDevice(i);
        cudaEventElapsedTime(&time, start[i], stop[i]);
        time /= 1000; // seconds
        double size = n * sizeof(double);
        double bandwidth = ((double)(n) * sizeof(double)) / time;
        cout << "rank: " << rank << " gpu: " << i << " size: " << size << " time: " << time << " bw: " << bandwidth << endl;
    }
}

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        cout << "Please specify the number of vertices (data size) in thousands and number of gpus per node";
        return 0;
    }
    int n;
    int ngpus;
    n = atoi(argv[1]);
    ngpus = atoi(argv[2]);
    n *= 1000;

    // MPI_Comm squarerowcomm, squarecolcomm;
    // MPI_Comm tallrowcomm, tallcolcomm;
    // MPI_Comm widerowcomm, widecolcomm;

    int rank, nprocs;
    MPI_Init( 0, 0 );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs); 
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    int devcount = 0;
    cudaGetDeviceCount(&devcount);
    std::cout << "rank: " << rank << " devcount: " << devcount << std::endl;

    ncclUniqueId ncclId;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * ngpus);
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < ngpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(&comms[i], nprocs * ngpus, ncclId, rank * ngpus + i));
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclGroupStart());
    testBcast(ncclId, comms, n, ngpus);
    NCCLCHECK(ncclGroupEnd());
    
    // // First do square grid 
    // int grcols = (int)std::sqrt((float)nprocs); 
    // int grrows = grcols; 
    // 
    // int myproccol = rank % grcols; 
    // int myprocrow = rank / grcols; 
    // MPI_Comm_split( MPI_COMM_WORLD, myprocrow, rank, &squarerowcomm );  
    // processes with the same color are in the same new communicator 
    // MPI_Comm_split( MPI_COMM_WORLD, myproccol, rank, &squarecolcomm );
    // if(rank == 0) cout << "*** Processor row ***" << endl;
    // DoA2A(squarerowcomm, 32*n, squarecolcomm);
    // if(rank == 0) cout << "*** Processor column ***" << endl;
    // DoA2A(squarecolcomm, 32*n, squarerowcomm);
    // if(rank == 0) cout << "*** Processor row ***" << endl;
    // DoAG(squarerowcomm, n, squarecolcomm);
    // if(rank == 0) cout << "*** Processor column ***" << endl;
    // DoAG(squarecolcomm, n, squarerowcomm);
    //     

    // if(rank == 0)
    //     cout << "### TALL GRID ###" << endl;
    // // Now do tall grid
    // int tallgrcols = grcols / 2;
    // int tallgrrows = grrows * 2; 
    //     myproccol = rank % tallgrcols;
    // myprocrow = rank / tallgrcols;
    //     MPI_Comm_split( MPI_COMM_WORLD, myprocrow, rank, &tallrowcomm );
    //     MPI_Comm_split( MPI_COMM_WORLD, myproccol, rank, &tallcolcomm );
    // DoA2A(tallrowcomm, 32*n, tallcolcomm);
    // DoA2A(tallcolcomm, 32*n, tallrowcomm);
    // DoAG(tallrowcomm, n, tallcolcomm);
    // DoAG(tallcolcomm, n, tallrowcomm);

    // if(rank == 0)
    //     cout << "### WIDE GRID ###" << endl;
    // // Now do wide grid
    // int widegrcols = grcols * 2;
    // int widegrrows = grrows / 2; 
    //     myproccol = rank % widegrcols;
    // myprocrow = rank / widegrcols;
    //     MPI_Comm_split( MPI_COMM_WORLD, myprocrow, rank, &widerowcomm );
    //     MPI_Comm_split( MPI_COMM_WORLD, myproccol, rank, &widecolcomm );
    // DoA2A(widerowcomm, 32*n, widecolcomm);
    // DoA2A(widecolcomm, 32*n, widerowcomm);
    // DoAG(widerowcomm, n, widecolcomm);
    // DoAG(widecolcomm, n, widerowcomm);

    // for (int i = 0; i < ngpus; i++) {
    //     NCCLCHECK(ncclCommDestroy(comms[i]));
    // }
    // free(comms);
    MPI_Finalize( );
    
    return 0;
}

