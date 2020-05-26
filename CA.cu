/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  Test three linear solvers, including Cholesky, LU and QR.
 *  The user has to prepare a sparse matrix of "matrix market format" (with extension .mtx).
 *  For example, the user can download matrices in Florida Sparse Matrix Collection.
 *  (http://www.cise.ufl.edu/research/sparse/matrices/)
 *
 *  The user needs to choose a solver by switch -R<solver> and
 *  to provide the path of the matrix by switch -F<file>, then
 *  the program solves
 *          A*x = b  where b = ones(m,1)
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|)
 *
 *  The elapsed time is also reported so the user can compare efficiency of different solvers.
 *
 *  How to use
 *      ./cuSolverDn_LinearSolver                     // Default: cholesky
 *     ./cuSolverDn_LinearSolver -R=chol -filefile>   // cholesky factorization
 *     ./cuSolverDn_LinearSolver -R=lu -file<file>     // LU with partial pivoting
 *     ./cuSolverDn_LinearSolver -R=qr -file<file>     // QR factorization
 *
 *  Remark: the absolute error on solution x is meaningless without knowing condition number of A.
 *     The relative error on residual should be close to machine zero, i.e. 1.e-15.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include "cusolverDn.h"
#include <helper_cuda.h>
#include "helper_cusolver.h"
#include <time.h>
#include <device_launch_parameters.h>
#include "Utilities.cuh"
//Added by Renchang Dai 4/30/2020 
#define BLOCK_SIZE 32

template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);


void UsageDN(void)
{
    printf( "<options>\n");
    printf( "-h          : display this help\n");
    printf( "-R=<name>    : choose a linear solver\n");
    printf( "              chol (cholesky factorization), this is default\n");
    printf( "              qr   (QR factorization)\n");
    printf( "              lu   (LU factorization)\n");
    printf( "-lda=<int> : leading dimension of A , m by default\n");
    printf( "-file=<filename>: filename containing a matrix in MM format\n");
    printf( "-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit( 0 );
}

/*
 *  solve A*x = b by Cholesky factorization
 *
 */
int linearSolverCHOL(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));


    // prepare a copy of A because potrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: Cholesky factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: cholesky = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info)); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }

    return 0;
}


/*
 *  solve A*x = b by LU with partial pivoting
 *
 */

__global__ void global_scale(double *a, int size, int index){
        int i;
        int start=(index*size+index);
        int end=(index*size+size);

        for(i=start+1;i<end;i++){
             if (a[i]!=0){
                a[i]=(a[i]/a[start]);
            }
        }

}

__global__ void global_reduce(double *a, int size, int index){
        int i;
       // int tid=threadIdx.x;
        int tid=blockIdx.x;
        int start= ((index+tid+1)*size+index);
        int end= ((index+tid+1)*size+size);

        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
                 if (a[i]!=0){
                     a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
                 }
        }

}

__global__ void scale(double *a, int size, int index, int nCtg){
        //printf("in scale fun blockidx.x = %d, blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
        //int iCtg = 0;
        //printf("nCtg=%d\n", nCtg);
        //for (iCtg = 0; iCtg < nCtg; iCtg ++){
            //double *a = 
            //int beginIndex = size*size * iCtg;
            int i;
            int start=(index*size+index);
            int end=(index*size+size);
            //printf("start=%d, end = %d\n", start, end);
            for(i=start+1;i<end;i++){
                //printf("a[i]_1=%f\n", a[i]);
                if (a[i] != 0){
                    a[i]=(a[i]/a[start]);
                }
                //printf("a[i]_2=%f\n", a[i]);
            }
        //}

}

__global__ void reduce_solo(double *a, int size, int index){
        int i;
        int tid=threadIdx.x;
        int start= ((index+tid+1)*size+index);
        int end= ((index+tid+1)*size+size);

        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
                 a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
        }

}

__global__ void reduce(double *a, int size, int index, int b_size, int nCtg){
        //printf("=================================================in reduce fun blockidx.x = %d, blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
    //int iCtg = 0;
    //for (iCtg = 0; iCtg < nCtg; iCtg ++){
            //double *a =
        //int beginIndex = size*size * iCtg;
        extern __shared__ double pivot[];
        int i;

        int tid=threadIdx.x;
        int bid=blockIdx.x;
        int block_size=b_size;

        int pivot_start=(index*size+index);
        int pivot_end=(index*size+size);
        
        int start;
        int end;
        int pivot_row;
        int my_row;
        int matrix_size = size*size;
        int num_matrix = index/matrix_size;
        int remainder = index % matrix_size;
        //int resize = (iCtg+1)*size*size;
        //int row_size = (iCtg+1)*size
        //int presize = (iCtg)*size*size;
        int reindex = index - num_matrix * matrix_size;
        //printf("index = %d, reindex = %d, size=%d, tid = %d\n", index, reindex, size, tid);
        //int N = 128;

        if(tid==0){
             for(i=reindex;i<size;i++){ 
                 //printf("i=%d\n", i);
                 pivot[i]=a[(index*size)+i];
                 //printf("pivot[i] = %f\n", pivot[i]);
             }
        }

        __syncthreads();

        pivot_row=(reindex*size);
        
        //pivot_row = ((index-iCtg*size*size)*size);
        my_row=(((block_size*bid) + tid)*size);
        start=my_row+index;
        //end=my_row+size;
        end = my_row + (num_matrix+1)*size;

        if(my_row >pivot_row){
        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
                // a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
                if (a[i]!=0){
                   a[i]=a[i]-(a[start]*pivot[(i-my_row-num_matrix*matrix_size)]);
                }

             }
        }
        
    //}

}

__global__ void  lud_base_cuda(double *m, int rowsA, int dimx)
{
  __shared__ double shadow[BLOCK_SIZE][BLOCK_SIZE];
  int i,j;
  printf("threadIdx.x=%d, threadIdx.y = %d, blockIdx.x=%d, blockIdx.y = %d, blockIdx.z=%d, blockDim.x = %d, blockDim.y = %d, blockDim.z = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z);
  //for (int block = 0; block < dimx; block ++){
  int block = blockIdx.x;
  for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) {
        //shadow[blockIdx.y*blockDim.y + threadIdx.y][[blockIdx.x*blockDim.x + threadIdx.x] = m[]
        shadow[tix][threadIdx.y] =  m[block * block + tix*BLOCK_SIZE+threadIdx.y];
        printf("tix = %d, threadIdx.y = %d, threadIdx.x = %d, BLOCK_SIZE=%d, items = %f, tix*BLOCK_SIZE+threadIdx.y= %d, item in m=%f\n", tix, threadIdx.y, threadIdx.x, BLOCK_SIZE, shadow[tix][threadIdx.y], tix*BLOCK_SIZE+threadIdx.y, m[tix*BLOCK_SIZE+threadIdx.y]);
  }
  //}
  __syncthreads();
  
  for(i=0; i < BLOCK_SIZE-1; i++)
   {
      //printf("i = %d\n", i);
      if ( threadIdx.y>i && threadIdx.x==0 )//计算下侧的L部分
      {
        for(j=0; j < i; j++)
          shadow[threadIdx.y][i] -= shadow[threadIdx.y][j]*shadow[j][i];
        
      }
      else if(threadIdx.y>=i   && threadIdx.x==1 ) //计算上侧的U部分
      {
        for(j=0; j < i; j++)
          shadow[i][threadIdx.y] -= shadow[i][j]*shadow[j][threadIdx.y];
      }
      __syncthreads();
      if(threadIdx.y>i && threadIdx.x==0)//对L进行后续处理
      {
        shadow[threadIdx.y][i] /= shadow[i][i];
      }
      __syncthreads();
  }
  //for (int i = 0; i  < BLOCK_SIZE; i++){
  //     for (int j = 0; j < BLOCK_SIZE; j ++){
  //          printf("i = %d, j = %d, item=%d\n", i, j, shadow[i][j]);
  //     }
  //}
  printf("blockIdx.x = %d, blockIdx.y=%d\n", blockIdx.x, blockIdx.y);
  block = blockIdx.x;
  for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) {
        //printf("tix=%d\n", tix);
        //printf("threadIdx.y=%d\n", threadIdx.y);
        //printf("tix*BLOCK_SIZE+threadIdx.y=%d\n", tix*BLOCK_SIZE+threadIdx.y);
        //m[tix*BLOCK_SIZE+threadIdx.y]=shadow[tix][threadIdx.y];
        m[block* block + tix*BLOCK_SIZE+threadIdx.y]=shadow[tix][threadIdx.y];
        //printf("m[tix*BLOCK_SIZE+threadIdx.y]=%f\n", m[tix*BLOCK_SIZE+threadIdx.y]);
  }
}


// __global__ void  lud_right_looking(double *d_A_Ctg, int nCtg, int lda, int colsA)
 __global__ void  lud_right_looking(double *m, int nCtg, int lda, int colsA)
{
    //int deep, row;
    //double *m;
    //for(deep =0; deep < nCtg; deep++){
    //m = (double *)((char*)d_A_Ctg + lda * colsA);
    __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];
        for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) 
      shadow[tix][threadIdx.y] =  m[tix*BLOCK_SIZE+threadIdx.y];
    __syncthreads();
  
    for (int k=0; k< BLOCK_SIZE-1; k++)
    {
      //这行语句不适合放进下面的for循环中，多执行一次就多浪费一次资源。
      if (threadIdx.y>k && threadIdx.x==0)
        shadow[threadIdx.y][k]=shadow[threadIdx.y][k]/shadow[k][k];
      __syncthreads();
      for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) 
      {
        if(tix>k && threadIdx.y>k)
            shadow[tix][threadIdx.y] -= shadow[tix][k]*shadow[k][threadIdx.y];
        __syncthreads();
      }
    }
    
    for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) 
        m[tix*BLOCK_SIZE+threadIdx.y]=shadow[tix][threadIdx.y];
    //}
}

int  linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    double start, stop;
    double time_solve;

    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));
    //checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy[i], lda, &bufferSize));//Revised by Renchang Dai 4/30/2020.

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    //start = second();
    //start = second();
    start = clock();
    //checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    //dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 DimGrid(iDivUp(N, BLOCK_SIZE), iDivUp(N, BLOCK_SIZE));

    //cusolverDnDgetrf<<<DimGrid, DimBlock>>>(handle, n, n, A, lda, buffer, ipiv, info);
    cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info);
    //stop = second();
    stop = clock();
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));
    if ( 0 != h_info ){
        fprintf(stderr, "Error in LU: LU factorization failed\n");
    }
    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());
    //stop = second();

    //time_solve = stop - start;
    printf("timing: LU=%f\n",(double)(stop-start)/CLOCKS_PER_SEC);
    //time_solve
    //fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
}


/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    double *tau = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    const double one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnDormqr_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        NULL,
        x,
        n,
        &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);
    
    bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ; 

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(double)*n));

// prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

// compute QR factorization
    checkCudaErrors(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        tau,
        x,
        n,
        buffer,
        bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(
         cublasHandle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         1,
         &one,
         A,
         lda,
         x,
         n));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: QR = %10.6f sec\n", time_solve);

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (tau   ) { checkCudaErrors(cudaFree(tau)); }

    return 0;
}


void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
    {
        UsageDN();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "R"))
    {
        char *solverType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

        if (solverType)
        {
            if ((STRCASECMP(solverType, "chol") != 0) && (STRCASECMP(solverType, "lu") != 0) && (STRCASECMP(solverType, "qr") != 0))
            {
                printf("\nIncorrect argument passed to -R option\n");
                UsageDN();
            }
            else
            {
                opts.testFunc = solverType;
            }
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageDN();
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "lda"))
    {
        opts.lda = getCmdLineArgumentInt(argc, (const char **)argv, "lda");
    }
}

//int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

int main (int argc, char *argv[])
{
    int blocks;
    double startTiming, stopTiming;
    struct testOpts opts;
    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;
	
	const int N = 128; // Grid size is N x N. Added by Renchang Dai 4/30/2020
	int nCtg = 100; // Number of contingencies. Added by Renchang Dai 4/30/2020

    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format
    int lda   = 0; // leading dimension in dense matrix

    // CSR(A) from I/O
    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    double *h_csrValA = NULL;

    double *h_A = NULL; // dense matrix from CSR(A)
    double *h_A_Ctg = NULL; // dense matrix from CSR(A) for contingencies
    double *h_x = NULL; // a copy of d_x
    double *h_b = NULL; // b = ones(m,1)
    double *h_r = NULL; // r = b - A*x, a copy of d_r

    double *d_A = NULL; // a copy of h_A
    double *d_res = NULL;
    double *d_A_Ctg = NULL; // a copy of h_A_ctg
    double *d_x = NULL; // x = A \ b
    double *d_b = NULL; // a copy of h_b
    double *d_r = NULL; // r = b - A*x
    double *d_tmp = NULL;

    // the constants are used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;

    double x_inf = 0.0;
    double r_inf = 0.0;
    double A_inf = 0.0;
    int errors = 0;

    parseCommandLineArguments(argc, argv, opts);

    if (NULL == opts.testFunc)
    {
        opts.testFunc = "chol"; // By default running Cholesky as NO solver selected with -R option.
    }

    findCudaDevice(argc, (const char **)argv);

    printf("step 1: read matrix market format\n");

    if (opts.sparse_mat_filename == NULL)
    {
        opts.sparse_mat_filename =  sdkFindFilePath("gr_900_900_crg.mtx", argv[0]);
        if (opts.sparse_mat_filename != NULL)
            printf("Using default input file [%s]\n", opts.sparse_mat_filename);
        else
            printf("Could not find gr_900_900_crg.mtx\n");
    }
    else
    {
        printf("Using input file [%s]\n", opts.sparse_mat_filename);
    }

    if (opts.sparse_mat_filename == NULL)
    {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return EXIT_FAILURE;
    }
    printf("Begin to open file...");
    printf(opts.sparse_mat_filename);
    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true , &rowsA, &colsA,
               &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true))
    {
        exit(EXIT_FAILURE);
    }
    printf("sucess...");
    baseA = h_csrRowPtrA[0]; // baseA = {0,1}

    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    if ( rowsA != colsA )
    {
        fprintf(stderr, "Error: only support square matrix\n");
        exit(EXIT_FAILURE);
    }

	// Added Begin by Renchang Dai 4/30/2020
	// Contingency analysis
	// Lower triangular matrix
	double key;
	int iCtg = 0;
        lda = opts.lda ? opts.lda : rowsA;
        printf("lda: %d\n", lda);
        printf("colsA: %d\n", colsA);
        printf("nCtg: %d\n", nCtg);
	h_A_Ctg = (double*)malloc(sizeof(double)*nCtg*lda*colsA);
        //double **result;
        //double **result_b;
        //h_A_Ctg = (double **)malloc(sizeof(double)*nCtg);
        //result=(double **)malloc(sizeof(double *)*rowsA);
        //for(int i=0;i<nCtg;i++){
        //   h_A_Ctg[i]=(double *)malloc(sizeof(double)*lda*colsA);
           //result_b[i]=(double *)malloc(sizeof(double)*rowsA);
           //b[i]=(float *)malloc(sizeof(float)*N);
        //}

        //for (iCtg = 0; iCtg < nCtg; iCtg++){
	for(int i = 0; i<rowsA; ++i) 
	{
		for (int j = h_csrRowPtrA[i]; j<h_csrRowPtrA[i+1]-1; ++j)
		{
			//iCtg ++;
			if (iCtg>=nCtg) break;
			// Take one element out
			if (h_csrColIndA[j] != i)
			{
				key = h_csrValA[j];
				h_csrValA[h_csrRowPtrA[i+1]] += key;
				h_csrValA[j] = 1e-4;
				
				//step 2: convert CSR(A) to dense matrix
				//printf("step 2: convert CSR(A) to dense matrix\n");
				if (lda < rowsA)
				{
					fprintf(stderr, "Error: lda must be greater or equal to dimension of A\n");
					exit(EXIT_FAILURE);
				}

				assert(NULL != h_A_Ctg);

				for(int row = 0 ; row < rowsA ; row++)
				{
					const int start = h_csrRowPtrA[row  ] - baseA;
					const int end   = h_csrRowPtrA[row+1] - baseA;
					for(int colidx = start ; colidx < end ; colidx++)
					{
						const int col = h_csrColIndA[colidx] - baseA;
						const double Areg = h_csrValA[colidx];
						//h_A_Ctg[iCtg][row + col*lda] = Areg;
                                                //printf("index = %d", iCtg*(row + col*lda));
                                                h_A_Ctg[iCtg*(rowsA*rowsA)+ (row + col*lda)] = Areg;

					}
				}
                                iCtg ++;
				// Restore CSR(A)
				h_csrValA[j] = key;
				h_csrValA[h_csrRowPtrA[i+1]] -= key;
			}
		}
	}
        //}
        
        
	// Added End by Renchang Dai 4/30/2020
	printf("step 2-2: convert CSR(A) to dense matrix\n");
        
	lda = opts.lda ? opts.lda : rowsA;
	if (lda < rowsA)
	{
		fprintf(stderr, "Error: lda must be greater or equal to dimension of A\n");
		exit(EXIT_FAILURE);
	}
	h_A = (double*)malloc(sizeof(double)*lda*colsA);
        d_res = (double*)malloc(sizeof(double)*nCtg*lda*colsA);
	h_x = (double*)malloc(sizeof(double)*colsA);
	h_b = (double*)malloc(sizeof(double)*rowsA);
	h_r = (double*)malloc(sizeof(double)*rowsA);
	assert(NULL != h_A);
	assert(NULL != h_x);
	assert(NULL != h_b);
	assert(NULL != h_r);
	memset(h_A, 0, sizeof(double)*lda*colsA);
        memset(d_res, 0, sizeof(double)*lda*colsA);
	for(int row = 0 ; row < rowsA ; row++)
	{
		const int start = h_csrRowPtrA[row  ] - baseA;
		const int end   = h_csrRowPtrA[row+1] - baseA;
		for(int colidx = start ; colidx < end ; colidx++)
		{
			const int col = h_csrColIndA[colidx] - baseA;
                        //printf(" h_csrValA[colidx]: %f",  h_csrValA[colidx]);
			const double Areg = h_csrValA[colidx];
			h_A[row + col*lda] = Areg;
		}
	}

        

    printf("step 3: set right hand side vector (b) to 1\n");
    char *b_file = NULL;
    char *sourceChar = opts.sparse_mat_filename;
    char *pDelimiter = "_";
    b_file = strtok(sourceChar, pDelimiter);
    char combine_b_file[10];
    strcpy(combine_b_file, b_file);
    strcat(combine_b_file, "B.txt");
    printf("0\n");
    FILE *fp;
    int id;
    double d;
    fp = fopen(combine_b_file, "r");
    int row = 0;
    while (fscanf(fp, "%d\t%lf", &id, &h_b[row]) > 0){
        //printf("id: %d; h_b: %lf\n", id, h_b[row]);
        row ++;
        
    }
    fclose(fp);
    //for(int row = 0 ; row < rowsA ; row++)
    //{
        //h_b[row] = 1.0;
    //    fscanf(fpRead, "%f", h_b[row]);
    //    printf("%lf \n", h_b[row]);
    //}

    // verify if A is symmetric or not.
    if ( 0 == strcmp(opts.testFunc, "chol") )
    {
        int issym = 1;
        for(int j = 0 ; j < colsA ; j++)
        {
            for(int i = j ; i < rowsA ; i++)
            {
                double Aij = h_A[i + j*lda];
                double Aji = h_A[j + i*lda];
                if ( Aij != Aji )
                {
                    issym = 0;
                    break;
                }
            }
        }
        if (!issym)
        {
            printf("Error: A has no symmetric pattern, please use LU or QR \n");
            exit(EXIT_FAILURE);
        }
    }

    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*lda*colsA));
    //checkCudaErrors(cudaMalloc((void **)&d_A_Ctg, sizeof(double)*nCtg*lda*colsA)); //Added by Renchang Dai 4/30/2020
    checkCudaErrors(cudaMalloc((void **)&d_A_Ctg, sizeof(double)*nCtg*lda*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double)*rowsA));
    //cudaMalloc((void **)&d_tmp, sizeof(double)*lda*colsA);
    //for(int i=0;i<nCtg;i++){
           //d_A_Ctg[i]=(double *)malloc(sizeof(double)*lda*colsA);
    //       printf("d_A_Ctg: %f\n", d_A_Ctg[i]);
    //       printf("i = %d\n", i);
           //d_A_Ctg[i] = d_tmp + i*lda*colsA;
    //       d_A_Ctg[i]=(double *)malloc(sizeof(double)*lda*colsA);
           //checkCudaErrors(cudaMalloc((double **)&d_A_Ctg[i], sizeof(double)*lda*colsA));
    //       printf("Done");
    //    }

    printf("step 4: prepare data on device\n");
    startTiming = clock();
    
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_Ctg, h_A_Ctg, sizeof(double)*nCtg*lda*colsA, cudaMemcpyHostToDevice)); //Revised by Renchang Dai 4/30/2020
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
    stopTiming = clock();
    printf("timing: cudaMemcpyHostToDevice=%f\n",(double)(stopTiming-startTiming)/CLOCKS_PER_SEC);
    
    printf("step 5: solve A*x = b \n");
    //printf("Matrix h_A_Ctg is :\n");
    //    for(int i=0; i<(nCtg*rowsA*rowsA); i++){
    //       if(i%rowsA==0)
    //      printf("\n %f ", h_A_Ctg[i]);
    //       else printf("%lf ",h_A_Ctg[i]);
    //    }

    //printf("\nMatrix h_A is :\n");
    //    for(int i=0; i<(rowsA*rowsA); i++){
    //       if(i%rowsA==0)
    //       printf("\n %f ", h_A[i]);
    //       else printf("%lf ",h_A[i]);
    //     }

    //double **result;
    //double **result_b;
    //result=(double **)malloc(sizeof(double *)*rowsA);
    //result_b=(double **)malloc(sizeof(double *)*rowsA);
    //for(int i=0;i<rowsA;i++){
    //       result[i]=(double *)malloc(sizeof(double)*rowsA);
    //       result_b[i]=(double *)malloc(sizeof(double)*rowsA);
           //b[i]=(float *)malloc(sizeof(float)*N);
    // }
    // d_A and d_b are read-only
    if ( 0 == strcmp(opts.testFunc, "chol") )
    {
        linearSolverCHOL(handle, rowsA, d_A, lda, d_b, d_x);
    }
    else if ( 0 == strcmp(opts.testFunc, "lu") )
    {
		//Added by Renchang Dai 4/30/2020
		dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 DimGrid(iDivUp(rowsA, DimBlock.x), iDivUp(rowsA, DimBlock.y), 1);	

        startTiming = clock();
        printf("\nnCtg=%d, lda=%d, colsA=%d\n", nCtg, lda, colsA);
       
        //lud_right_looking<<<DimGrid, DimBlock>>>(d_A, nCtg, lda, colsA);
        //const int rowsN;
        //rowsN = rowsA
        //printf("DimGridx, y, y=%d, %d, %d, DimBlockx, y, z=%d, %d, %d", DimGrid.x, DimGrid.y, DimGrid.z, DimBlock.x, DimBlock.y, DimBlock.z);
        //lud_base_cuda<<<DimGrid, DimBlock>>>(d_A, rowsA, DimGrid.x);
        //linearSolverLU(handle, rowsA, d_A, lda, d_b, d_x);
        //linearSolverLU<<<DimGrid, DimBlock>>>(handle, rowsA, d_A_Ctg, lda, d_b, d_x);
        //lud_base_cuda<<<DimGrid, DimBlock>>>(d_A);
        //for(int i =0; i< 10; i++){
            
            //lud_right_looking<<<DimGrid, DimBlock>>>(d_A);
        //    printf("i = %d\n", i);
        //    lud_right_looking<<<DimGrid, DimBlock>>>(d_A_Ctg[i]);
        //    printf("after lud_right\n");
        //}
        //// start shared memorg
        int totalRowsA = nCtg*rowsA;
        for(int i=0;i<totalRowsA;i++){
            //printf("nCtg*rowsA = %d\n", totalRowsA);
            scale<<<1, 1>>>(d_A_Ctg,rowsA,i, nCtg);
            blocks=((totalRowsA/512 + 1));
            //printf("Number of blocks rxd : %d \n",blocks);
            reduce<<<blocks,512,rowsA*sizeof(double)>>>(d_A_Ctg,rowsA,i,512, nCtg);
       }
        // start global memorg
        //for(int i=0;i<totalRowsA;i++){
        //global_scale<<<1, 1>>>(d_A_Ctg,rowsA,i);
        //global_reduce<<<(rowsA-i-1),1>>>(d_A_Ctg,rowsA,i);
        // }
       stopTiming = clock();
       printf("\ntiming: map and reduce solver=%f\n",(double)(stopTiming-startTiming)/CLOCKS_PER_SEC);
       cudaMemcpy( d_res, d_A_Ctg, nCtg*rowsA*rowsA*sizeof(double),cudaMemcpyDeviceToHost );
       //printf("\nres A is \n");
       //for ( int i = 0; i < (nCtg*rowsA*rowsA); i++) {
       //        if(i%rowsA==0)
       //        printf( "\n%f  ", d_res[i]);
       //        else printf("%lf ",d_res[i]);
       // }
       //printf("\n");

/*
        for(int i=0;i<rowsA;i++){
             for(int j=0;j<rowsA;j++){
                result[i][j]=d_res[i*rowsA+j];
                }
        }
        printf("\nThe result matrix\n");
        for(int i=0;i<rowsA;i++){
                for(int j=0;j<rowsA;j++){
                printf("%lf ",result[i][j]);
                }
          printf("\n");
        }
*/
        /*
        int N = rowsA;
        double l1;
        double u1;

        for(int i=0;i<N;i++){
           for(int j=0;j<N;j++){
                result_b[i][j]=0;
              for(int k=0;k<N;k++){
                 if(i>=k){
                     l1=result[i][k];
                 }else l1=0;

                  if(k==j)u1=1;
                  else if(k<j)u1=result[k][j];//figured it out
                  else u1=0.0;

               result_b[i][j]=result_b[i][j]+(l1*u1);

             }
           }
         }
          */


        //printf("==================================================");
        /*printf("\nThe b matrix\n");

         for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                printf("%lf ",result_b[i][j]);
                }
          printf("\n");
          }

        */
        //linearSolverLU(handle, rowsA, d_A, lda, d_b, d_x); //Revised by Renchang Dai 4/30/2020
    }
    else if ( 0 == strcmp(opts.testFunc, "qr") )
    {
        linearSolverQR(handle, rowsA, d_A, lda, d_b, d_x);
    }
    else
    {
        fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
        exit(EXIT_FAILURE);
    }
    printf("step 6: evaluate residual\n");
    // r = b - A*x
    checkCudaErrors(cublasDgemm_v2(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        rowsA,
        1,
        colsA,
        &minus_one,
        d_A,
        lda,
        d_x,
        rowsA,
        &one,
        d_r,
        rowsA));

    startTiming = clock();
    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double)*rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));
    stopTiming = clock();
    printf("timing: cudaMemcpyDeviceToHost=%f\n",(double)(stopTiming-startTiming)/CLOCKS_PER_SEC);

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = mat_norminf(rowsA, colsA, h_A, lda);

    printf("|b - A*x| = %E \n", r_inf);
    printf("|A| = %E \n", A_inf);
    printf("|x| = %E \n", x_inf);
    printf("|b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

    if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (h_csrValA   ) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }

    if (h_A) { free(h_A); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }
    if (h_r) { free(h_r); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }

    return 0;
}

