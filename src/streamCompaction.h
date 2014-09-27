// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef STREAM_COMPACTION_H
#define STREAM_COMPACTION_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <time.h>
#include <map>

#define THREADS_PER_BLOCK 64

struct dataPacket{
	int index;
	bool alive;
	dataPacket(){
		index = -1;
		alive = true;
	}
	dataPacket(int i){
		index = i;
		alive = true;
	}
};

class DataStream{
private:

	dataPacket * m_data;

	int m_numElementsAlive, m_numElements;

	dataPacket * cudaDataA;
	dataPacket * cudaDataB;

	int * cudaIndicesA;
	int * cudaIndicesB;
	
	int * cudaAuxSums;
	int * cudaAuxIncs;

public:
	int * m_indices;
	int * m_auxSums;

	DataStream(int numElements, dataPacket * data);
	~DataStream();

	void compactWorkEfficientArbitrary();
	void compactNaiveSumGlobal();
	void compactNaiveSumSharedSingleBlock();
	void compactNaiveSumSharedArbitrary();
	bool getData(int index, dataPacket& data);
	int numAlive();
	void kill(int index);
	void fetchDataFromGPU();

};

void cudaVectorSum(int * indicesA, int * indicesB, int numElements, float k);

void cudaInit (dataPacket * a, dataPacket * b, int * ia, int * ib);

void testStreamCompaction();

#endif
