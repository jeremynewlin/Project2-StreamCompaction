#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "streamCompaction.h"

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

__global__ void test(int* in, int* out, int n){

  extern __shared__ float temp[];
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = 1;

  if (2*index+1<=n){
    temp[2*index] = in[2*index];
    temp[2*index+1] = in[2*index+1];

    for (int d = n>>1; d>0; d >>= 1){
    //for (int d=0; d<floor(log(float(n-1)/log(2.0f))); d+=1){
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

__global__ void streamCompaction(dataPacket* inRays, int* indices, dataPacket* outRays, int numElements){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k<numElements){
    dataPacket inRay = inRays[k];
    if (inRay.alive){
      outRays[indices[k]-1] = inRay;
    }
  }
}

__global__ void killStream(int index, dataPacket* inRays, int* indices, int numElements){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k<numElements){
    inRays[k].alive = true;
    indices[k] = 1;
    if (k == index){
      inRays[k].alive = false;
      indices[k] = 0;
    }
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
  m_numElementsAlive = numElements;
  m_numElements = numElements;

  m_indices = new int[numElements];
  for (int i=0; i<numElements; i+=1){
    m_indices[i] = 1;
  }

  //cudaInit (cudaDataA, cudaDataB, cudaIndicesA, cudaIndicesB);
  cudaMalloc ((void**)&cudaDataA, numElements*sizeof (dataPacket));
  cudaMalloc ((void**)&cudaDataB, numElements*sizeof (dataPacket));
  cudaMalloc ((void**)&cudaIndicesA, numElements*sizeof (int));
  cudaMalloc ((void**)&cudaIndicesB, numElements*sizeof (int));

  cudaMemcpy(cudaDataA, m_data, numElements*sizeof(dataPacket), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDataB, m_data, numElements*sizeof(dataPacket), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaIndicesA, m_indices, numElements*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaIndicesB, m_indices, numElements*sizeof(int), cudaMemcpyHostToDevice);
}

DataStream::~DataStream(){
  cudaFree (cudaDataA);
  cudaFree (cudaDataB);
  cudaFree (cudaIndicesA);
  cudaFree (cudaIndicesB);

  delete [] m_data;
  delete [] m_indices;
}

void DataStream::compact(){

  int numElements = m_numElementsAlive;
  int threadsPerBlock = 64;
  int procsPefBlock = threadsPerBlock*2;

  dim3 initialScanThreadsPerBlock(threadsPerBlock/2);
  dim3 initialScanBlocksPerGrid(numElements/threadsPerBlock);

  dim3 threadsPerBlockL(threadsPerBlock);
  dim3 fullBlocksPerGridL(int(ceil(float(m_numElementsAlive)/float(threadsPerBlock))));

  // // scan algorithm
  //   for (int d=1; d<=ceil(log(m_numElementsAlive)/log(2)); d++){
  //     sum<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaIndicesA, cudaIndicesB, m_numElementsAlive, powf(2.0f, d-1));
  //     int* temp = cudaIndicesA;
  //     cudaIndicesA = cudaIndicesB;
  //     cudaIndicesB = temp;
  //   }

    test<<<fullBlocksPerGridL, threadsPerBlockL, m_numElementsAlive*sizeof(int)>>>(cudaIndicesA, cudaIndicesB, m_numElementsAlive);
    checkCUDAError("kernel failed!");
    cudaMemcpy(m_indices, cudaIndicesB, m_numElements*sizeof(int), cudaMemcpyDeviceToHost);
    // //Stream compation from A into B, then save back into A
    // streamCompaction<<<fullBlocksPerGridL, threadsPerBlockL>>>(cudaDataA, cudaIndicesA, cudaDataB, m_numElementsAlive);
    // dataPacket * temp = cudaDataA;
    // cudaDataA = cudaDataB;
    // cudaDataB = temp;

    // // update numrays
    // cudaMemcpy(&m_numElementsAlive, &cudaIndicesA[m_numElementsAlive-1], sizeof(int), cudaMemcpyDeviceToHost);
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
}