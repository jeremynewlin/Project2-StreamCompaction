#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "streamCompaction.h"

using namespace std;
#include <thrust/copy.h>

// ...
// struct is_even
// {
//   __host__ __device__
//   bool operator()(const int x)
//   {
//     return (x % 2) == 0;
//   }
// };
// ...
// int N = 6;
// int data[N]    = { 0, 1,  2, 3, 4, 5};
// int stencil[N] = {-2, 0, -1, 0, 1, 2};
// int result[4];
// thrust::copy_if(data, data + N, stencil, result, is_even());
// // data remains    = { 0, 1,  2, 3, 4, 5};
// // stencil remains = {-2, 0, -1, 0, 1, 2};
// // result is now     { 0, 1,  3, 5}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
  }
} 

__global__ void sum(int* in, int* out, int n, int d1){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (k<n){
    int ink = in[k];
    if (k>=d1){
      out[k] = in[k-d1] + ink;
    }
    else{
      out[k] = ink;
    }
  }
}

__global__ void shift(int* in, int* out, int n){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  out[0] = 0;
  if (k<n && k>0){
    out[k] = in[k-1];
  }
}

__global__ void naiveSumGlobal(int* in, int* out, int n){

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  int logn = ceil(log(float(n))/log(2.0f));
  for (int d=1; d<=logn; d++){
    
    int offset = powf(2.0f, d-1);
    
    if (index >= offset){
      out[index] = in[index-offset] + in[index];
    }
    else{
      out[index] = in[index]; 
    }
    __syncthreads();

    int* temp = in;
    in = out;
    out = temp;
  }
}

__global__ void naiveSumSharedSingleBlock(int* in, int* out, int n){

  int index = threadIdx.x;

  if (index >= n) return;

  extern __shared__ int shared[];
  int *tempIn = &shared[0];
  int *tempOut = &shared[n];

  tempOut[index] = (index > 0) ? in[index-1] : 0;  

  __syncthreads();

  for (int offset = 1; offset <= n; offset *= 2){
    int* temp = tempIn;
    tempIn = tempOut;
    tempOut = temp;

    if (index >= offset){
      tempOut[index] = tempIn[index-offset] + tempIn[index];
    }
    else{
      tempOut[index] = tempIn[index]; 
    }
    __syncthreads();
  }
  out[index] = tempOut[index];
}

__global__ void naiveSumSharedArbitrary(int* in, int* out, int n, int* sums=0){

  int localIndex = threadIdx.x;
  int globalIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

  // if (globalIndex >= n) return;

  // out[k] = index; return;

  extern __shared__ int shared[];
  int *tempIn = &shared[0];
  int *tempOut = &shared[n];

  tempOut[localIndex] = in[globalIndex];  
  
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2){
    int* temp = tempIn;
    tempIn = tempOut;
    tempOut = temp;

    if (localIndex >= offset){
      tempOut[localIndex] = tempIn[localIndex-offset] + tempIn[localIndex];
    }
    else{
      tempOut[localIndex] = tempIn[localIndex]; 
    }
    __syncthreads();
  }

  if (sums) sums[blockIdx.x] = tempOut[n-1];
  out[globalIndex] = tempOut[localIndex];
}

__global__ void workEfficientSumSingleBlock(int* in, int* out, int n){

  extern __shared__ float temp[];
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = 1;

  if (2*index+1<=n){
    temp[2*index] = in[2*index];
    temp[2*index+1] = in[2*index+1];

    for (int d = n>>1; d>0; d >>= 1){
      __syncthreads();
      if (index < d){
        int ai = offset * (2*index+1) - 1;
        int bi = offset * (2*index+2) - 1;

        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (index == 0) temp[n - 1] = 0;

    for (int d = 1; d<n; d*=2){
      offset >>= 1;
      __syncthreads();
      if (index < d){

        int ai = offset * (2*index+1) - 1;
        int bi = offset * (2*index+2) - 1;

        if (ai < n && bi < n){
          float t = temp[ai];
          temp[ai] = temp[bi];
          temp[bi] += t;
        }
      }
    }
    __syncthreads();

    out[2*index] = temp[2*index];
    out[2*index+1] = temp[2*index+1];
  }

}

__global__ void workEfficientArbitrary(int* in, int* out, int n, int* sums=0){

  extern __shared__ float temp[];

  int realIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  int offset = 1;
  int index = threadIdx.x;

  temp[2*index] = in[2*realIndex];
  temp[2*index+1] = in[2*realIndex+1];

  for (int d = n>>1; d>0; d >>= 1){
    __syncthreads();
    if (index < d){
      int ai = offset * (2*index+1) - 1;
      int bi = offset * (2*index+2) - 1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  
  if (index == 0){
    if (sums) sums[blockIdx.x] = temp[n-1];
    temp[n - 1] = 0;
  }

  for (int d = 1; d<n; d*=2){
    offset >>= 1;
    __syncthreads();
    if (index < d){

      int ai = offset * (2*index+1) - 1;
      int bi = offset * (2*index+2) - 1;

      if (ai < n && bi < n){
        float t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
  }
  __syncthreads();

  out[2*realIndex] = temp[2*index];
  out[2*realIndex+1] = temp[2*index+1];

}

__global__ void addIncs(int* cudaAuxIncs, int* cudaIndicesB, int n){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;


  // if (index < n){
    // cudaIndicesB[index] = blockIdx.x; //cudaAuxIncs[blockIdx.x];
    cudaIndicesB[index] += cudaAuxIncs[blockIdx.x];
  // }
}

__global__ void streamCompaction(dataPacket* inRays, int* indices, dataPacket* outRays, int numElements){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k<numElements){
    dataPacket inRay = inRays[k];
    if (inRay.alive){
      outRays[indices[k]] = inRay;
    }
  }
}

struct isAlive
{
  __host__ __device__
  bool operator()(const dataPacket& dp)
  {
    return dp.alive;
  }
};

struct isEven
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x%2 == 0);
  }
};

struct isOne
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x == 1);
  }
};

__global__ void killStream(int index, dataPacket* inRays, int* indices, int numElements){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k<numElements){
    if (k == index){
      inRays[k].alive = false;
      indices[k] = 0;
    }
  }
}

__global__ void resetStreams(dataPacket* inRays, int* indices, int numElements){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k<numElements){
    inRays[k].alive = true;
    indices[k] = 1;
  }
}

void testStreamCompaction(){
  //Testing stream compaction
  int numElements = 10;
  dataPacket* arrayOfElements = new dataPacket[numElements];
  for (int i=0; i<numElements; i+=1){
    dataPacket rb(i);
    arrayOfElements[i] = rb;
  }

  arrayOfElements[1].alive=false;
  arrayOfElements[4].alive=false;
  arrayOfElements[5].alive=false;
  arrayOfElements[7].alive=false;
  arrayOfElements[8].alive=false;


  dataPacket* cudaArrayA;
  dataPacket* cudaArrayB;

  cudaMalloc((void**)&cudaArrayA, numElements*sizeof(dataPacket));
  cudaMalloc((void**)&cudaArrayB, numElements*sizeof(dataPacket));

  int* testin;
  int* testout;
  int* cputest = new int[numElements];

  for (int i=0; i<numElements; i++){
    if (arrayOfElements[i].alive){
      cputest[i]=1;
    }
    else{
      cputest[i]=0;
    }
  }

  cudaMalloc((void**)&testin, numElements*sizeof(int));
  cudaMalloc((void**)&testout, numElements*sizeof(int));

  cudaMemcpy(cudaArrayA, arrayOfElements, numElements*sizeof(dataPacket), cudaMemcpyHostToDevice);  
  cudaMemcpy(cudaArrayB, arrayOfElements, numElements*sizeof(dataPacket), cudaMemcpyHostToDevice);  
  cudaMemcpy(testin, cputest, numElements*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(testout, cputest, numElements*sizeof(int), cudaMemcpyHostToDevice);

  for (int i=0; i<numElements; i++){
    std::cout<<arrayOfElements[i].index<<", "<<cputest[i]<<std::endl;
  }

  dim3 threadsPerBlock(64);
  dim3 fullBlocksPerGrid(int(ceil(float(numElements)/64.0f)));

  //scan
  for (int d=1; d<=ceil(log(numElements)/log(2))+1; d++){
    sum<<<fullBlocksPerGrid, threadsPerBlock>>>(testin, testout, numElements, int(pow(2.0f,d-1)));
    cudaThreadSynchronize();
    cudaMemcpy(cputest, testout, numElements*sizeof(int), cudaMemcpyDeviceToHost);


    int* temp = testin;
    testin=testout;
    testout=temp;
  }
  //Compact
  streamCompaction<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaArrayA, testin, cudaArrayB, numElements);
  cudaArrayA = cudaArrayB;
  cudaThreadSynchronize();

  cudaMemcpy(&numElements, &testin[numElements-1], 1*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout<<"number of rays left: "<<numElements<<std::endl;

  // for (int i=0; i<numElements; i++){
  //   std::cout<<cputest[i]<<std::endl;
  // }    

  cudaMemcpy(cputest, testin, numElements*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(arrayOfElements, cudaArrayA, numElements*sizeof(dataPacket), cudaMemcpyDeviceToHost);


  for (int i=0; i<numElements; i++){
    std::cout<<arrayOfElements[i].index<<std::endl;

  }
  std::cout<<"___________________________________"<<std::endl;


  delete [] cputest;
  cudaFree(testin);
  cudaFree(testout);

  delete [] arrayOfElements;
  cudaFree(cudaArrayA);
  cudaFree(cudaArrayB);
}

DataStream::DataStream(int numElements, dataPacket * data){
  m_data = data;

  if (numElements % (THREADS_PER_BLOCK*2)) numElements+=1;

  m_numElementsAlive = numElements;

  if (numElements % (THREADS_PER_BLOCK*2) != 0){
    int counter = 1;
    while (THREADS_PER_BLOCK*2*counter < numElements){
      counter += 1;
    }
    numElements = THREADS_PER_BLOCK*2*counter;
  }

  m_numElements = numElements;

  m_indices = new int[numElements];
  for (int i=0; i<numElements; i+=1){
    if (i < m_numElementsAlive){
      m_indices[i] = 1;
    }
    else{
      m_indices[i] = 0;
    }
  }

  m_auxSums = new int[numElements/(THREADS_PER_BLOCK*2)];
  for (int i=0; i<numElements/(THREADS_PER_BLOCK*2); i+=1){
    m_auxSums[i] = 0;
  }

  //cudaInit (cudaDataA, cudaDataB, cudaIndicesA, cudaIndicesB);
  cudaMalloc ((void**)&cudaDataA, numElements*sizeof (dataPacket));
  cudaMalloc ((void**)&cudaDataB, numElements*sizeof (dataPacket));
  cudaMalloc ((void**)&cudaIndicesA, numElements*sizeof (int));
  cudaMalloc ((void**)&cudaIndicesB, numElements*sizeof (int));
  cudaMalloc ((void**)&cudaAuxSums, numElements/(THREADS_PER_BLOCK*2)*sizeof (int));
  cudaMalloc ((void**)&cudaAuxIncs, numElements/(THREADS_PER_BLOCK*2)*sizeof (int));

  cudaMemcpy(cudaDataA, m_data, numElements*sizeof(dataPacket), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDataB, m_data, numElements*sizeof(dataPacket), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaIndicesA, m_indices, numElements*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaIndicesB, m_indices, numElements*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaAuxSums, m_auxSums, numElements/(THREADS_PER_BLOCK*2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaAuxIncs, m_auxSums, numElements/(THREADS_PER_BLOCK*2)*sizeof(int), cudaMemcpyHostToDevice);
}

DataStream::~DataStream(){
  cudaFree (cudaDataA);
  cudaFree (cudaDataB);
  cudaFree (cudaIndicesA);
  cudaFree (cudaIndicesB);
  cudaFree (cudaAuxSums);
  cudaFree (cudaAuxIncs);

  delete [] m_data;
  delete [] m_indices;
  delete [] m_auxSums;
}

void DataStream::serialScan(){
  m_indices[0] = 0;
  for (int i=1; i<m_numElementsAlive; i+=1){
    m_indices[i] = m_indices[i] + m_indices[i-1];
  }
}

void DataStream::globalSum(int* in, int* out, int n){
  int threadsPerBlock = THREADS_PER_BLOCK;

  dim3 threadsPerBlockL(threadsPerBlock);
  dim3 fullBlocksPerGridL(m_numElements/threadsPerBlock);

  for (int d=1; d<=ceil(log(m_numElementsAlive)/log(2)); d++){
    sum<<<fullBlocksPerGridL, threadsPerBlockL>>>(in, out, m_numElementsAlive, powf(2.0f, d-1));
    cudaThreadSynchronize();
    int* temp = in;
    in = out;
    out = temp;
  }
  shift<<<fullBlocksPerGridL, threadsPerBlockL>>>(in, out, m_numElementsAlive);
}

void DataStream::thrustStreamCompact(){
  thrust::copy_if (m_data, m_data+m_numElements, m_indices, m_data, isOne());
}

void DataStream::compactWorkEfficientArbitrary(){

  int numElements = m_numElements;
  int threadsPerBlock = THREADS_PER_BLOCK; // 8
  int procsPefBlock = threadsPerBlock*2;   // 16

  dim3 initialScanThreadsPerBlock(procsPefBlock/2);        //8
  dim3 initialScanBlocksPerGrid(numElements/procsPefBlock);//

  int sumSize = numElements/(THREADS_PER_BLOCK*2);

  dim3 initialScanThreadsPerBlock2(sumSize/2);        //16
  dim3 initialScanBlocksPerGrid2(sumSize/(sumSize/2)+1);//1024/16

  dim3 initialScanThreadsPerBlock3(procsPefBlock);        //8
  dim3 initialScanBlocksPerGrid3(numElements/procsPefBlock);//3

  dim3 threadsPerBlockL(threadsPerBlock);
  dim3 fullBlocksPerGridL(int(ceil(float(m_numElementsAlive)/float(threadsPerBlock))));

  workEfficientArbitrary<<<initialScanBlocksPerGrid, initialScanThreadsPerBlock, procsPefBlock*sizeof(int)>>>(cudaIndicesA, cudaIndicesB, procsPefBlock, cudaAuxSums);
  checkCUDAError("kernel failed1!");

  // cudaMemcpy(m_indices, cudaIndicesA, m_numElements*sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i=0; i<numAlive(); i+=1){
  //   cout<<m_indices[i];
  //   if (i<numAlive()-1) cout<<",";
  // }
  // cout<<endl;

  // cudaMemcpy(m_indices, cudaIndicesB, m_numElements*sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i=0; i<numAlive(); i+=1){
  //   cout<<m_indices[i];
  //   if (i<numAlive()-1) cout<<",";
  // }
  // cout<<endl;

  for (int d=1; d<=ceil(log(sumSize)/log(2)); d++){
    sum<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaAuxSums, cudaAuxIncs, sumSize, powf(2.0f, d-1));
    cudaThreadSynchronize();
    int* temp = cudaAuxSums;
    cudaAuxSums = cudaAuxIncs;
    cudaAuxIncs = temp;
  }
  shift<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaAuxSums, cudaAuxIncs, m_numElementsAlive);

  addIncs<<<initialScanBlocksPerGrid3, initialScanThreadsPerBlock3>>>(cudaAuxIncs, cudaIndicesB, m_numElements);
  checkCUDAError("kernel failed2!");

  // cudaMemcpy(m_indices, cudaIndicesA, m_numElements*sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i=0; i<numAlive(); i+=1){
  //   cout<<m_indices[i];
  //   if (i<numAlive()-1) cout<<",";
  // }
  // cout<<endl;

  // cudaMemcpy(m_indices, cudaIndicesB, m_numElements*sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i=0; i<numAlive()+1; i+=1){
  //   cout<<m_indices[i];
  //   if (i<numAlive()-1) cout<<",";
  // }
  // cout<<endl;

  // cudaMemcpy(m_auxSums, cudaAuxIncs, m_numElements/(THREADS_PER_BLOCK*2)*sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i=0; i<m_numElements/(THREADS_PER_BLOCK*2); i+=1){
  //   cout<<m_auxSums[i];
  //   if (i<numAlive()-1) cout<<",";
  // }
  // cout<<endl;

  //Stream compation from A into B, then save back into A
  streamCompaction<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaDataA, cudaIndicesB, cudaDataB, m_numElementsAlive);
  dataPacket * temp = cudaDataA;
  cudaDataA = cudaDataB;
  cudaDataB = temp;

  // update numrays
  cudaMemcpy(&m_numElementsAlive, &cudaIndicesB[m_numElementsAlive], sizeof(int), cudaMemcpyDeviceToHost);

  resetStreams<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaDataA, cudaIndicesA, m_numElementsAlive);
}

void DataStream::compactNaiveSumGlobal(){

  int threadsPerBlock = THREADS_PER_BLOCK;

  dim3 threadsPerBlockL(threadsPerBlock);
  dim3 fullBlocksPerGridL(m_numElements/threadsPerBlock);

  for (int d=1; d<=ceil(log(m_numElementsAlive)/log(2)); d++){
    sum<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaIndicesA, cudaIndicesB, m_numElementsAlive, powf(2.0f, d-1));
    cudaThreadSynchronize();
    int* temp = cudaIndicesA;
    cudaIndicesA = cudaIndicesB;
    cudaIndicesB = temp;
  }
  shift<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaIndicesA, cudaIndicesB, m_numElementsAlive);

  // Stream compation from A into B, then save back into A
  streamCompaction<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaDataA, cudaIndicesB, cudaDataB, m_numElementsAlive);
  dataPacket * temp = cudaDataA;
  cudaDataA = cudaDataB;
  cudaDataB = temp;

  // update numrays
  cudaMemcpy(&m_numElementsAlive, &cudaIndicesA[m_numElementsAlive-1], sizeof(int), cudaMemcpyDeviceToHost);
  cout<<m_numElementsAlive<<endl;

  resetStreams<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaDataA, cudaIndicesA, m_numElementsAlive);
}

void DataStream::compactNaiveSumSharedSingleBlock(){

  int threadsPerBlock = THREADS_PER_BLOCK;

  dim3 threadsPerBlockL(threadsPerBlock);
  dim3 fullBlocksPerGridL(int(ceil(float(m_numElementsAlive)/float(threadsPerBlock))));

  naiveSumSharedSingleBlock<<<fullBlocksPerGridL, threadsPerBlockL, 2*m_numElements*sizeof(int)>>>(cudaIndicesA, cudaIndicesB, m_numElements);
  checkCUDAError("kernel failed!");

  cudaMemcpy(m_indices, cudaIndicesB, m_numElements*sizeof(int), cudaMemcpyDeviceToHost);
}

void DataStream::compactNaiveSumSharedArbitrary(){

  ////////////////////////////////////////////////////////////////////////////////////////
  int threadsPerBlock = THREADS_PER_BLOCK;

  dim3 threadsPerBlockL(threadsPerBlock*2);
  dim3 fullBlocksPerGridL(m_numElements/(threadsPerBlock*2));

  naiveSumSharedArbitrary<<<fullBlocksPerGridL, threadsPerBlockL, 2*m_numElements/(m_numElements/(threadsPerBlock*2))*sizeof(int)>>>(cudaIndicesA, cudaIndicesB, threadsPerBlock*2, cudaAuxSums);
  checkCUDAError("kernel failed 1 !");
  ////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////////////
  int sumSize = m_numElements/(THREADS_PER_BLOCK*2);
  dim3 initialScanThreadsPerBlock2(threadsPerBlock);        
  dim3 initialScanBlocksPerGrid2(sumSize/threadsPerBlock+1);

  dim3 threadsPerBlockOld(threadsPerBlock);
  dim3 fullBlocksPerGridOld(int(ceil(float(sumSize)/float(threadsPerBlock))));
  
  cudaMemcpy(cudaAuxIncs, cudaAuxSums, m_numElements/(THREADS_PER_BLOCK*2)*sizeof(int), cudaMemcpyDeviceToDevice);

  for (int d=1; d<=ceil(log(sumSize)/log(2)); d++){
    sum<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaAuxSums, cudaAuxIncs, sumSize, powf(2.0f, d-1));
    cudaThreadSynchronize();
    int* temp = cudaAuxSums;
    cudaAuxSums = cudaAuxIncs;
    cudaAuxIncs = temp;
  }
  shift<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaAuxSums, cudaAuxIncs, m_numElementsAlive);

  addIncs<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaAuxIncs, cudaIndicesB, m_numElements);

  shift<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaIndicesB, cudaIndicesA, m_numElementsAlive);
  int * temp = cudaIndicesA;
  cudaIndicesA = cudaIndicesB;
  cudaIndicesB = temp;
  ////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////////////
  dim3 threadsPerBlockLL(threadsPerBlock);
  dim3 fullBlocksPerGridLL(m_numElements/threadsPerBlock);

  //Stream compation from A into B, then save back into A
  streamCompaction<<<fullBlocksPerGridLL, threadsPerBlockLL>>>(cudaDataA, cudaIndicesB, cudaDataB, m_numElementsAlive);
  dataPacket * tempDP = cudaDataA;
  cudaDataA = cudaDataB;
  cudaDataB = tempDP;

  // // update numrays
  ////////////////////////////////////////////////////////////////////////////////////////
  cudaMemcpy(&m_numElementsAlive, &cudaIndicesA[m_numElementsAlive-1], sizeof(int), cudaMemcpyDeviceToHost);
  cout<<m_numElementsAlive<<endl;
  //////////////////////////////////////////////////////////////////////////////////////

  resetStreams<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaDataA, cudaIndicesA, m_numElementsAlive);
}

bool DataStream::getData(int index, dataPacket& data){

  if (index > m_numElements) return false;

  data = m_data[index];
  return true;
}

int DataStream::numAlive(){
  return m_numElementsAlive;
}

void DataStream::fetchDataFromGPU(){
  cudaMemcpy(m_data, cudaDataA, m_numElementsAlive*sizeof(dataPacket), cudaMemcpyDeviceToHost);
}

void DataStream::kill(int index){
  if (index > m_numElementsAlive) return;

  dim3 threadsPerBlockL(64);
  dim3 fullBlocksPerGridL(int(ceil(float(m_numElementsAlive)/64.0f)));

  killStream<<<fullBlocksPerGridL, threadsPerBlockL>>>(index, cudaDataA, cudaIndicesA, m_numElementsAlive);

  cudaMemcpy(m_indices, cudaIndicesA, m_numElementsAlive*sizeof(int), cudaMemcpyDeviceToHost);
}