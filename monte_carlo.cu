/*
    By: Carrick McClain
    Sources:
        http://csweb.cs.wfu.edu
        https://developer.download.nvidia.com
        https://stackoverflow.com
        http://www.cplusplus.com
        https://devtalk.nvidia.com
        https://docs.nvidia.com/cuda/cuda-c-programming-guide
*/

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define NUM_RANDS 50000000  // Number of Monte-Carlo simulations
#define NUM_BLOCKS 40
#define NUM_BLOCKS_MC 256
#define THREADS_PER_BLOCK 256
#define TRIALS_PER_THREAD (256*16)

inline void gpu_handle_error( cudaError_t err, const char* file, int line, int abort = 1 )
{
	if (err != cudaSuccess)
	{
		fprintf (stderr, "gpu error: %s, %s, %d\n", cudaGetErrorString (err), file, line);
		if (abort)
			exit (EXIT_FAILURE);
	}
}
#define gpu_err_chk(e) {gpu_handle_error( e, __FILE__, __LINE__ );}

/*
Integral Functions
You can replace any invoked math function with another.
    To test this, you can replace the function calls in the
    trapezoidal functions (host & device) with any of the others below.
I tried to implement these functions with functors, but they didn't work
as expected with device code.    */
float func_1a( float input )
{
    return 1/(1+input*input);
}
__device__ float func_1b( float input )
{
    return 1/(1+input*input);
}

// function 2 (host & gpu versions)
float func_2a( float input )
{
    return ((1.0*input*input) + (3.0*input*input) + 5.0);
}
__device__ float func_2b( float input )
{
    return ((1.0*input*input) + (3.0*input*input) + 5.0);
}

//function 3 (host & gpu versions)
float func_3a( float input )
{
    return ((2.0*input*input*input) / (5.0*input*input));
}
__device__ float func_3b( float input )
{
    return ((2.0*input*input*input) / (5.0*input*input));
}



// Serial trapezoidal rule function.
// Change around the commented lines to run it with other math functions.
float trapezoidal( float a, float b, float n )
{
    float delta = (b-a)/n;
    float s = func_1a(a) + func_1a(b);
    // float s = func_2a(a) + func_2a(b);
    // float s = func_3a(a) + func_3a(b);

    for( int i = 1; i < n; i++ )
    {
        s += 2.0*func_1a(a+i*delta);
        // s += 2.0*func_2a(a+i*delta);
        // s += 2.0*func_3a(a+i*delta);
    }
    return (delta/2)*s;
}

// Parallelized trapezoidal rule function.
// Change around the commented lines to run it with other math functions.
__global__ void trapezoidal_kernel( float a, float b, float n, float* output )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float delta = (b-a)/n;
    float s = a + (float)tid * delta;

    if( tid < n )
    {
        output[tid] = func_1b(s) + func_1b(s + delta);
        // output[tid] = func_2b(s) + func_2b(s + delta);
        // output[tid] = func_3b(s) + func_3b(s + delta);
    }
}

// Assignment 8 device code
// Parallelized Monte Carlo Method
// Should work for any function on any interval.
__global__ void monte_carlo_kernel( float a, float b, float* estimates, curandState *states )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int points_under_curve = 0;
    float x, y;

    curand_init(1234, tid, 0, &states[tid]);
    for( int i=0; i < TRIALS_PER_THREAD; i++ )
    {
        x = curand_uniform (&states[tid]);
        y = curand_uniform (&states[tid]);
        points_under_curve += func_1b(x) <= 1.0f;
    }
    estimates[tid] = points_under_curve / (float)TRIALS_PER_THREAD;
}

int main()
{
    // starts CUDA context, absorbs cost of startup
    // while starting, the program may seem to hang for a few seconds!
    // don't worry, it will work eventually.
    cudaFree(0);

    // initializations
    cudaError_t err;
    float a = 0.0f;     // interval start
    float b = 1.0f;     // interval end
    int n = 10000;      // number of trapezoids
    float delta = (b-a)/n;
    float parallel_result = 0.0f;
    float* h_kernel_output = (float*)malloc(n * sizeof(float));
    float* d_kernel_output;

    // print out host function result
    cout << "Serial: Value of integral is " << trapezoidal(a, b, n) << endl;
    
/*  Now the parallel part.
    The cudaMalloc was taking tons of time when I tested, not sure why.
    That's why I made the cudaFree(0) at the beginning.
    It absorbs the time cost of setting up the CUDA context,
        so the cudaMalloc() then takes much less time to execute.  */
    err = cudaMalloc( (void**) &d_kernel_output, n * sizeof(float) );
    gpu_err_chk(err);
    err = cudaMemcpy( d_kernel_output, h_kernel_output, n * sizeof(float), cudaMemcpyHostToDevice );
    gpu_err_chk(err);
    
    // call kernel function
    trapezoidal_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>( a, b, n, d_kernel_output);
    err = cudaGetLastError();
    gpu_err_chk(err);
    
    // copy data back from device
    err = cudaMemcpy( h_kernel_output, d_kernel_output, n * sizeof(float), cudaMemcpyDeviceToHost );
    gpu_err_chk(err);

    // get correct sum of trapezoid array
    for( int i=0; i<n; i++ )
    {
        parallel_result += h_kernel_output[i];
    }
    parallel_result *= delta/2.0;

    // print out device function result
    cout << "Parallel: Value of integral is " << parallel_result << endl;

    // free up memory
    free(h_kernel_output);
    cudaFree(d_kernel_output);



// Assignment 8 host code
// Monte Carlo attempt
    float estimate;
    float* h_est = (float*)malloc(NUM_BLOCKS_MC * THREADS_PER_BLOCK * sizeof(float));
    float* d_est;
    curandState* h_states;
    curandState* d_states;

    err = cudaMalloc( (void**)&d_est, NUM_BLOCKS_MC * THREADS_PER_BLOCK * sizeof(float) );
    gpu_err_chk(err);
    err = cudaMalloc( (void**)&d_states, NUM_BLOCKS_MC * THREADS_PER_BLOCK * sizeof(float) );
    gpu_err_chk(err);

    monte_carlo_kernel<<<NUM_BLOCKS_MC, THREADS_PER_BLOCK>>>( a, b, d_est, d_states );
    err = cudaGetLastError();
    gpu_err_chk(err);

    cudaMemcpy( h_est, d_est, NUM_BLOCKS_MC * THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(d_est);

    for( int i=0; i < NUM_BLOCKS_MC * THREADS_PER_BLOCK; i++ )
    {
        estimate += h_est[i];
    }
    estimate /= (NUM_BLOCKS_MC * THREADS_PER_BLOCK);    // Get the average value from threads

    cout << "Estimate: " << estimate << endl;
    
    return 0;
}