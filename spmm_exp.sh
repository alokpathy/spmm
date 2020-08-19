nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl 2d_nccl.cu -o 2d_nccl 

echo "" > ring3_results/ring3_results.txt
echo "" > ring6_results/ring6_results.txt

nodecount=$1

for gpu in 3 6
    do
        echo "gpu: $gpu"
        for i in {10..18}
            do
                echo $i
                echo $i >> ring$gpu\_results/ring$gpu\_results.txt
                jsrun -n$nodecount -g$gpu ./2d_nccl $(echo 2^$i | bc)  $gpu >> ring$gpu\_results/ring$gpu\_results.txt
            done
    done

