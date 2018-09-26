set -e
nvcc nnd_cuda.cu -o nnd_cuda.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
