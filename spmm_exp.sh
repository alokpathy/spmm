proccount=$1
procdim=$2
library=$3
exp=$4

procdim=$((2 * procdim)) # one resource set per socket, 2 sockets per node

if [ "$library" = "mpi" ]; then
    nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm mpi_ex_$exp\.cpp -o mpi_ex_$exp
elif [ "$library" = "mpi_cpu" ]; then
    nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm mpi_ex_$exp\.cpp -o mpi_ex_$exp
elif [ "$library" = "nccl" ]; then
    nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl nccl_ex_$exp\.cu -o nccl_ex_$exp
fi

echo -n "" > $library\_$exp\_results/proc$proccount\.txt
for i in {1..13}
    do
        echo $i
        echo $i >> $library\_$exp\_results/proc$proccount\.txt

        if [ "$library" = "mpi" ]; then
            jsrun -n$procdim -a3 -g3 --smpiargs="-gpu" ./mpi_ex_$exp $(echo 2^$i | bc) 3 $proccount >> mpi_$exp\_results/proc$proccount\.txt
        elif [ "$library" = "mpi_cpu" ]; then
            jsrun -n$procdim -a3 -c21 ./mpi_ex_$exp $(echo 2^$i | bc) 3 $proccount >> mpi_cpu_$exp\_results/proc$proccount\.txt
        elif [ "$library" = "nccl" ]; then
            jsrun -n$procdim -a3 -g3 ./nccl_ex_$exp $(echo 2^$i | bc) 3 $proccount >> nccl_$exp\_results/proc$proccount\.txt
        fi

    done
