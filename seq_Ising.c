// Sequential algorithm
#include <stdlib.h>
#include "seq_Ising.h"

// swap implementation
void swap(int ***x, int ***y){
int **temp;
temp = *x;
*x = *y;
*y = temp;
}

// modulo implementation for negative numbers
int mod(int a, int b){ // b must be always positive
    int r = a % b;
    return r < 0 ? r + b : r;
}

// function implementation
void seq_Ising(int** older, int** newer, int k, int n){
	int sum;
	for(int l = 0 ; l < k ; l++){ // the k iterations
		// build the new phase of the material(or just create the formula that describes our model)
		for(int i = 0 ; i < n ; i++){
			for(int j = 0 ; j < n ; j++){
				sum = older[i][j] + older[i][mod(j-1, n)] + older[i][(j+1)%n] + older[(mod(i-1, n))][j] + older[(i+1)%n][j];
				if(sum > 0){
					newer[i][j] = 1;
				} else {
					newer[i][j] = -1;
				}
			}
		}
		
		//exchange the pointers so that the new becomes old again(so that the iterations do not go back and forward)
		swap(&older, &newer);
	}
}