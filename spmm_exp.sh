nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl nccl_ex_2d.cu -o nccl_ex_2d

nodecount=$1

echo -n "" > results/node$nodecount\_gpu1_results_ex.txt

# # nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl 2d_nccl.cu -o 2d_nccl 
# nvcc -I$MPI_ROOT/include -L$MPI_ROOT/lib -lmpi_ibm -lnccl nccl_ex.cu -o nccl_ex
# 
# nodecount=$1
# 
# echo -n "" > results/node$nodecount\_gpu1_results_ex.txt
# echo -n "" > results/node$nodecount\_gpu3_results_ex.txt
# echo -n "" > results/node$nodecount\_gpu6_results_ex.txt
# 
# for gpu in 1 3 6
#     do
#         echo "gpu: $gpu"
#         # for i in {10..20}
#         for i in {5..20}
#             do
#                 echo $i
#                 echo $i >> results/node$nodecount\_gpu$gpu\_results_ex.txt
#                 # jsrun -n$nodecount -g$gpu ./2d_nccl $(echo 2^$i | bc)  $gpu >> ring$gpu\_results/ring$gpu\_results.txt
#                 jsrun -n$nodecount -g$gpu ./nccl_ex $(echo 2^$i | bc) $gpu >> results/node$nodecount\_gpu$gpu\_results_ex.txt
#             done
#     done
# 
