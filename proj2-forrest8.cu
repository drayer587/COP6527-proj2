/* ==================================================================
	// histogram privatization + tiling on shared mem
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include "timing.cuh"


#define BOX_SIZE	23000 /* size of the data box on one dimension            */
#define BUCKET_TYPE unsigned long long // data type of a bucket

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	float x_pos;
	float y_pos;
	float z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


BUCKET_TYPE * histogram;	/* list of all buckets in the histogram   */
unsigned PDH_acnt;		/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
float   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
float p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	float dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos]++;
		} 
	}
	return 0;
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
float report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (float)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(BUCKET_TYPE * _histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		int tmp = (int)_histogram[i];
		printf("%15lld ", (long long)tmp);
		total_cnt += _histogram[i];
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/* histogram differences */
BUCKET_TYPE * hist_diff(BUCKET_TYPE * hist1, BUCKET_TYPE * hist2){
	BUCKET_TYPE * diff = (BUCKET_TYPE*)malloc(num_buckets*sizeof(BUCKET_TYPE));
	for (int i=0; i<num_buckets; i++){
		diff[i] = hist1[i] - hist2[i];
	}
	return diff;
}

__device__ double dis_func(float x1,float y1,float z1,float x2,float y2,float z2){
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* ---------------------------------------------------- GPU CODE -------------------------------------------------- */

__global__ void kernel(float* d_x, float* d_y, float* d_z, int n_atoms, BUCKET_TYPE * d_hist, float PDH_res, int M, int buckets){
        int t = threadIdx.x;
        int b = blockIdx.x;
        int B = blockDim.x;
        extern __shared__ atom mydata[];
	atom *R = (atom *)mydata;
	unsigned int *shmout = (unsigned int *)&mydata[B];
       	register atom reg;
       	
	float dist;
        int j, h_pos;
	//bounds check
        int i = t+B*b;
        if(i < n_atoms) {
        	//initialize shared memory to zero
        	for(j = t; j < buckets; j += B) shmout[j] = 0; 
        	//the t-th datum of b-th input data block
		reg.x_pos = d_x[i];
		reg.y_pos = d_y[i];
		reg.z_pos = d_z[i];
                for(j = b+1; j < M; j++) {
                	//R <- the i-th input data block loaded to cache
			if(t + j*B < n_atoms){
			R[t].x_pos = d_x[t + j*B];
			R[t].y_pos = d_y[t + j*B];
			R[t].z_pos = d_z[t + j*B];
			}
        	        int k;
        	        __syncthreads();
        	        for(k = 0; k < B; k++){
        	        	//d <- DisFunction(r eg, R[ j])
        	        	dist = dis_func(reg.x_pos,reg.y_pos,reg.z_pos,
        	        			R[k].x_pos,R[k].y_pos,R[k].z_pos);
        	        	h_pos = (int) (dist / PDH_res);
	                        atomicAdd((unsigned int *) &shmout[h_pos],1);
        	        }
		}
		//L ← the b-th input data block loaded to cache
		R[t] = reg;
		__syncthreads();
                for(j = t+1; j < B; j++) {
                	//d <- DisFunction(reg, L[i])
        	        dist = dis_func(reg.x_pos,reg.y_pos,reg.z_pos,
        	        		R[j].x_pos,R[j].y_pos,R[j].z_pos);
        	        h_pos = (int) (dist / PDH_res);
	        	atomicAdd((unsigned int *) &shmout[h_pos],1);
                }
		__syncthreads();
        	for(j = t; j < buckets; j += B) 
        		atomicAdd((BUCKET_TYPE *) &d_hist[j], shmout[j]);
	}
}

/*
I have shared with you the code to call a kernel function by passing a shared memory size in the SDH2.cu. Note that there is a 3rd runtime parameter for calling the kernel, that is the shared memory size. In addition, within the kernel, you should define the corresponding shared data structure as follows:

extern __shared__ int mydata[];

instead of 

__shared__ int my data[256]; 

as you did before. 
*/


/* ---------------------------------------------------- MAIN -------------------------------------------------- */
int main(int argc, char **argv)
{
	int i;
	int block_size;

	if(argc < 2){
		printf("Missing particle number and distance input\n");
		exit(0);
	}
	if(argc < 3){
		printf("Missing particle number or distance input\n");
		exit(0);
	}

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);

	if (argc > 3) block_size = atoi(argv[3]);
	else{printf("No block size given\n"); exit(0);}

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	int num_blocks = ceil(PDH_acnt/block_size);
	histogram = (BUCKET_TYPE *)malloc(sizeof(BUCKET_TYPE)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	float* h_x = (float*)malloc(PDH_acnt*sizeof(float));
	float* h_y = (float*)malloc(PDH_acnt*sizeof(float));
	float* h_z = (float*)malloc(PDH_acnt*sizeof(float));

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		h_x[i] = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		h_y[i] = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		h_z[i] = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].x_pos = h_x[i];
		atom_list[i].y_pos = h_y[i];
		atom_list[i].z_pos = h_z[i];
	}

	// /* call CPU single thread version to compute the histogram */
	// /* start counting time */
	TIMING_START();
	
	// PDH_baseline();
	PDH_baseline();
	
	/* check the total running time */ 
	TIMING_STOP();
	
	/* print out the histogram */
	printf("\nCPU results: \n");
	output_histogram(histogram);
	TIMING_PRINT();

	/* ---------------------------- GPU version ---------------------------- */
	// copy data to GPU 
	float *d_x, *d_y, *d_z;
	cudaMalloc((void**)&d_x, PDH_acnt*sizeof(float));
	cudaMalloc((void**)&d_y, PDH_acnt*sizeof(float));
	cudaMalloc((void**)&d_z, PDH_acnt*sizeof(float));
	cudaMemcpy(d_x, h_x, PDH_acnt*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, PDH_acnt*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, h_z, PDH_acnt*sizeof(float), cudaMemcpyHostToDevice);

	// create output space
	BUCKET_TYPE *d_hist;
	cudaMalloc((void**)&d_hist, num_buckets*sizeof(BUCKET_TYPE));
	cudaMemset(d_hist, 0, num_buckets*sizeof(BUCKET_TYPE));

	// dynamic shared mem
	//size_t sharedMemSize = num_buckets*sizeof(unsigned);
	size_t sharedMemSize = sizeof(atom)*block_size + num_buckets*sizeof(int);
	
	TIMING_START();
	kernel <<< num_blocks, block_size, sharedMemSize >>> (d_x, d_y, d_z, PDH_acnt, d_hist, PDH_res, num_blocks, num_buckets);
	cudaDeviceSynchronize();
	TIMING_STOP();

	// copy data back to host
	BUCKET_TYPE * histogram_GPU = (BUCKET_TYPE*)malloc(num_buckets*sizeof(BUCKET_TYPE));
	cudaMemcpy(histogram_GPU, d_hist, num_buckets*sizeof(BUCKET_TYPE), cudaMemcpyDeviceToHost);
	// print out
	printf("\nGPU results: \n");
	output_histogram(histogram_GPU);
	TIMING_PRINT();

	// check diff
	BUCKET_TYPE * diff = hist_diff(histogram_GPU, histogram);
	printf("\nDIFFERENCES: \n");
	output_histogram(diff);
	
	return 0;
}

