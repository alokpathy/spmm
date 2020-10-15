import statistics
import sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <filename> <bwtype>")
        return

    filename = sys.argv[1]
    bwtype = sys.argv[2]

    print(f"data,min,max,avg,median")
    with open(filename) as f:
        datasize = -1
        bandwidths = []
        for line in f:
            if "Success" in line or "Hostname" in line or "Warning" in line or "local_gpuid" in line or "bcast1d_time" in line:
                continue

            tokens = line.strip().split()
            if len(tokens) == 1:
                if datasize != -1:
                    min_bw = min(bandwidths)
                    max_bw = max(bandwidths)
                    avg_bw = sum(bandwidths) / len(bandwidths)
                    med_bw = statistics.median(bandwidths)
                    print(f"{2**datasize * 1000},{min_bw},{max_bw},{avg_bw},{med_bw}")
                    bandwidths = []

                datasize = int(tokens[0])
            elif "{}_time:".format(bwtype) in line:
                bandwidth = float(tokens[-1])
                bandwidths.append(bandwidth)
    
if __name__ == "__main__":
    main()
