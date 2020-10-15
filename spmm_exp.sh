proccount=$1
procdim=$2
library=$3

procdim=$((2 * procdim))
if [ "$library" = "mpi" ]; then
    nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm mpi_ex_2d.cpp -o mpi_ex_2d
elif [ "$library" = "mpi_cpu" ]; then
    nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm mpi_ex_2d.cpp -o mpi_ex_2d
elif [ "$library" = "nccl" ]; then
    nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl nccl_ex_2d.cu -o nccl_ex_2d
fi

echo -n "" > $library\_2d_bcast_results/proc$proccount\.txt
for i in {1..13}
    do
        echo $i
        echo $i >> $library\_2d_bcast_results/proc$proccount\.txt

        if [ "$library" = "mpi" ]; then
            jsrun -n$procdim -a3 -g3 --smpiargs="-gpu" ./mpi_ex_2d $(echo 2^$i | bc) 3 $proccount >> mpi_2d_bcast_results/proc$proccount\.txt
        elif [ "$library" = "mpi_cpu" ]; then
            jsrun -n$procdim -a1 -cALL_CPUS ./mpi_ex_2d $(echo 2^$i | bc) 1 $proccount >> mpi_cpu_2d_bcast_results/proc$proccount\.txt
        elif [ "$library" = "nccl" ]; then
            jsrun -n$procdim -a3 -g3 ./nccl_ex_2d $(echo 2^$i | bc) 3 $proccount >> nccl_2d_bcast_results/proc$proccount\.txt
        fi

    done
