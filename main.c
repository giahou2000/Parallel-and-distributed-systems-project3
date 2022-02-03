// Ising model and CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "seq_Ising.h"

int main(int argc, char** argv){
	printf("\n");
	printf("Let's begin");
	printf("\n");
	
	int k = 1000; // k iterations of the formula
	int n = 5; // the dimensions of the square 2D lattice
	int **G1, **G2; // the 2D square lattices
	
	// print some info
	printf("\n");
	printf("We will do %d iterations of the formula", k);
	printf("\n");
	printf("The 2D square lattice's dimensions are %d x %d", n, n);
	printf("\n");
	
	// 1st 2D square lattice initialization
	G1 = (int **)malloc(n * sizeof(int *));
	for(int i = 0; i < n; i++){
		G1[i] = (int *)malloc(n * sizeof(int));
	}
	// filling the 2D square lattice with -1 or 1
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int random = (rand() % 2); // random numbers between 0 and 1
			if(random == 1){ // if random is 1 the G(i,j) is 1
				G1[i][j] = 1;
			} else { // if random is 0 the G(i,j) is -1
				G1[i][j] = -1;
			}
		}
	}
	
	// print the lattice for checking(uncomment to use!!!)
	/* printf("\n");
	printf("The beginning lattice");
	printf("\n");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf(" %d ",G1[i][j]);
		}
		printf("\n");
	} */
	
	// 2nd 2D square lattice initialization
	G2 = (int **)malloc(n * sizeof(int *));
	for(int i = 0; i < n; i++){
		G2[i] = (int *)malloc(n * sizeof(int));
	}
	
	printf("\n");
	printf("Let's start the procedure!!!");
	printf("\n");
	
	// Stopwatch start:
	clock_t begin = clock();	
	
	//***_____Call the sequential algorithm that does not use CUDA_____***
	seq_Ising(G1, G2, k, n);
	
	// Stopwatch finish
	clock_t end = clock();
	
	// Stopwatch results
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	
	//***_____End of procedure_____***
	
	//The time was:
	printf("The time_spent was: %f", time_spent);
	
	// print the lattice for checking(uncomment to use!!!)
	printf("\n");
	printf("The ending lattice");
	printf("\n");
	if((k%2) == 0){ // if k(the iterations) is an even number the result is at the G2 lattice
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				printf(" %d ",G2[i][j]);
			}
			printf("\n");
		}
	} else { // else if k(the iterations) is an odd number the result is at the G1 lattice
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				printf(" %d ",G1[i][j]);
			}
			printf("\n");
		}
	}
	
	
	// just before finish
	free(G1);
	free(G2);
	
	return 0;
	
}
