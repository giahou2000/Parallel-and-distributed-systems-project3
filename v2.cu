%%cu

/*
 * CUDA version:2(multiple threads per block, multiple work per thread)
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
__global__ void Ising(int* older, int* newer, int n, int w){
	
	//***inputs***
	// 'older' is the 2D square lattice that contains the initial info for the dipoles' moments
	// 'newer' is the 2D square lattice that will be used for the model's procedure(a simple exchange)
	// 'n' is the dimension of the square lattices
	//***output***
	// 'none': There is no output. The function does some calculations only.

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i * n + j;
	int index = ind * w;
	int sum;
	// build the new phase of the material(or just create the formula that describes our model)
 	if ( i < n && j < n ){
		for(int y = 0 ; y < w ; y++){
			for(int x = 0 ; x < w ; x++){
				int r = i * w + y;
				int c = j * w + x;
				sum = older[r + n * c] + older[n * c + mod(r - 1, n)] + older[n * c + (r + 1) % n] + older[mod(c - 1, n) * n + r] + older[((c + 1) % n) * n + r];
				if(sum > 0){
					newer[r + n * c] = 1;
				} else {
					newer[r + n * c] = -1;
				}
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
	int t = 2; // t*tpb(t*t threads per block)
 	dim3 dimBlock(t, t); // Block size / number of threads
 	printf("t = : %d threads dimension\n", t);
	int w = 2; // w*w work per thread - how much blocks from the 2d square lattice are assigned to the thread
	printf("w = : %d work per thread\n", t);
	

	// initial state
	printf("The initial state of the ferromagnetic substance is: \n");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf(" %d ",G1[i * n + j]);
		}
		printf("\n");
	}

	//***___start stopwatch___***
	clock_t begin = clock();
	
	//***___k iterations___***
	for(int i = 0 ; i < k ; i++){
		printf("*****____ iteration: %d ____***** \n", i);
		cudaMemcpy(CUDAG1, G1, n*n*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(CUDAG2, G2, n*n*sizeof(int), cudaMemcpyHostToDevice);
		Ising<<<dimGrid, dimBlock>>>(CUDAG1, CUDAG2, n, w);
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
 
	/*
	printf("The final state of the ferromagnetic substance is: \n");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf(" %d ",G1[i * n + j]);
		}
		printf("\n");
	}
	*/
	// free the memory, do not need it anymore
	cudaFree(CUDAG1);
	cudaFree(CUDAG2);
	free(G1);
	free(G2);
	
	return 0;
}
