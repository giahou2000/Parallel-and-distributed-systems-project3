%%cu

/*
 * CUDA version:3(multiple threads per block, shared memory usage, multiple moments computation per thread)
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// ***Modulo for negative numbers*** (useful only for the device - GPU. It is used in Ising function)
__device__ int mod(int a, int b){
	int r = a % b;
    return r < 0 ? r + b : r;
}

// the mathematical formula that will be run on the GPU
__shared__ int* lattice;
__global__ void Ising(int* older, int* newer, int n, int w, int t, int m){
	
	//***inputs***
	// 'older' is the 2D square lattice that contains the initial info for the dipoles' moments
	// 'newer' is the 2D square lattice that will be used for the model's procedure(a simple exchange)
	// 'n' is the dimension of the square lattices
	//***output***
	// 'none': There is no output. The function does some calculations only.

	

  //***___save the right amount of columns from global to shared memory___***
  if((threadIdx.x == 0) && (threadIdx.y == 0)){
      for(int i = 0 ; i < (t*w) ; i++){
					lattice[i + 1] = older[(mod(blockIdx.x - 1, m) * blockDim.x * w + (t-1) * w + w-1) * n + blockIdx.y * blockDim.y * w + i];
					lattice[(t * w + 1) * (t * w + 2) + i + 1] = older[mod(blockIdx.x + 1, m) * blockDim.x * w * n + blockIdx.y * blockDim.y * w + i];					
			}

			for(int i = 0 ; i < t*w ; i++){
				for(int j = 0 ; j < t*w ; j++){
					lattice[(i + 1)*(t * w + 2) + j + 1] = older[(blockIdx.x * blockDim.x * w + i) * n + blockIdx.y * blockDim.y * w + j];
				}
			}

			for(int i = 0 ; i < t*w ; i++){
					lattice[(i + 1)*(t * w + 2)] = older[(blockIdx.x * blockDim.x * w + i) * n + mod(blockIdx.y - 1, m) * blockDim.y * w + t * w - 1];
					lattice[(i + 1)*(t * w + 2) + t * w + 1] = older[(blockIdx.x * blockDim.x * w + i) * n + mod(blockIdx.y + 1, m) * blockDim.y * w];
			}
  }
	__syncthreads();
  
  // the standard procedure
	int lat_dim = t * w + 2;
  for(int i = 0 ; i < w ; i++){
      for(int j = 0 ; j < w ; j++){
          int col = threadIdx.x * w + 1 + i;
          int row = threadIdx.y * w + 1 + j;
					int sum = lattice[col * lat_dim + row] + lattice[(col - 1) * lat_dim + row] + lattice[(col + 1) * lat_dim + row] + lattice[col * lat_dim + (row - 1)] + lattice[col * lat_dim + (row + 1)];
          if(sum > 0){
						newer[(blockIdx.x * blockDim.x * w + threadIdx.x * w + i) * n + blockIdx.y * blockDim.y * w + threadIdx.y * w + j] = 1;
					} else {
						newer[(blockIdx.x * blockDim.x * w + threadIdx.x * w + i) * n + blockIdx.y * blockDim.y * w + threadIdx.y * w + j] = -1;
					}
      }
  }

	__syncthreads();
}



// swap implementation
void swap(int **x, int **y){
	int *temp = *x;
	*x = *y;
	*y = temp;
}



int main(void){
	
	printf("\n");
	printf("Let's begin");
	printf("\n");
	
	//***___CPU variables___***
	int k = 10; // k iterations of the formula
	int n = 8; // the dimensions of the square 2D lattice
	int *G1, *G2; // the 2D square lattices stored in a form of an array
	
	// 1st 2D square lattice initialization
	G1 = (int *)malloc(n * n * sizeof(int));
	// filling the 2D square lattice(array) with -1 or 1
	for(int i = 0; i < n * n; i++){
			int random = (rand() % 2); // random numbers between 0 and 1
			if(random == 1){ // if random is 1 the G(i,j) is 1
				G1[i] = 1;
			} else { // if random is 0 the G(i,j) is -1
				G1[i] = -1;
			}
	}
	
	// 2nd 2D square lattice initialization
	G2 = (int *)malloc(n * n * sizeof(int));
	
	//***___GPU variables___***
	int *CUDAG1, *CUDAG2; // the 2D square lattices(arrays) that will exist in the GPU
	
  cudaMalloc((void**)&CUDAG1, n * n * sizeof(int));

  cudaMalloc((void**)&CUDAG2, n * n * sizeof(int));	
	



	printf("\n");
	printf("Let's start the procedure!!!");
	printf("\n");

	//***___CUDA parameters___***
	int m = 2; // for gpu blocks -> m*m blocks
	dim3 dimGrid(m, m); // grid size / number of blocks
	printf("m = : %d block_dimension \n", m);
	int t = 2; // for gpu threads -> t*t threads per block
 	dim3 dimBlock(t, t); // Block size / number of threads
 	printf("t = : %d threads dimension\n", t);
	int w = 2; // for work per thread -> w*w spin computations per thread
	printf("w = : %d work per thread\n", t);
	int sh_size = (t * w + 2)*(t * w + 2)*sizeof(int); // shared memory size
	printf("sh_size = : %d \n", sh_size);
	

	// initial state
	printf("The initial state of the ferromagnetic substance is: \n");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf(" %d ",G1[i * n + j]);
		}
		printf("\n");
	}

	printf("\n");
	printf("\n");
	printf("\n");
	printf("\n");

	//***___start stopwatch___***
	clock_t begin = clock();
	
	//***___k iterations___***
	for(int i = 0 ; i < k ; i++){
		printf("*****____ iteration: %d ____***** \n", i);
		cudaMemcpy(CUDAG1, G1, n*n*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(CUDAG2, G2, n*n*sizeof(int), cudaMemcpyHostToDevice);
		Ising<<<dimGrid, dimBlock, sh_size>>>(CUDAG1, CUDAG2, n, w, t, m);
		cudaMemcpy(G1, CUDAG1, n*n*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(G2, CUDAG2, n*n*sizeof(int), cudaMemcpyDeviceToHost);
		swap(&G1, &G2);
		for(int l = 0; l < n; l++){
			for(int j = 0; j < n; j++){
				printf(" %d ",G1[l * n + j]);
			}
			printf("\n");
		}
	}
	
	// stop stopwatch and print time
	clock_t end = clock();
	
	// The execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("\n");
	printf("End of the procedure!!!");
	printf("\n");
	
	// print the execution time
	printf("The time spent for execution was: %f \n", time_spent);

	
	// print the finished state of the moments(the G1 array, because it holds the results after the last swap)
	printf("\n");
	printf("\n");
	printf("\n");
	printf("\n");

	printf("The final state of the ferromagnetic substance is: \n");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf(" %d ",G1[i * n + j]);
		}
		printf("\n");
	}

	// free the memory, do not need it anymore
	cudaFree(CUDAG1);
	cudaFree(CUDAG2);
	free(G1);
	free(G2);
	
	return 0;
}
