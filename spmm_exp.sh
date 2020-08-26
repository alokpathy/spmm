# nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl 2d_nccl.cu -o 2d_nccl 
nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl nccl_ex.cu -o nccl_ex

echo "" > ring1_results/ring1_results_ex.txt
echo "" > ring3_results/ring3_results_ex.txt
echo "" > ring6_results/ring6_results_ex.txt

nodecount=$1

for gpu in 1 3 6
    do
        echo "gpu: $gpu"
        for i in {5..15}
            do
                echo $i
                echo $i >> ring$gpu\_results/ring$gpu\_results_ex.txt
                # jsrun -n$nodecount -g$gpu ./2d_nccl $(echo 2^$i | bc)  $gpu >> ring$gpu\_results/ring$gpu\_results.txt
                jsrun -n$nodecount -g$gpu ./nccl_ex $(echo 2^$i | bc) $gpu >> ring$gpu\_results/ring$gpu\_results_ex.txt
            done
    done

