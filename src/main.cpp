#include <iostream>
#include <time.h>

#include "streamCompaction.h"

using namespace std;

void naive(){
	int numElements = 25;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0){
		int toKill = rand() % ds.numAlive();
		ds.kill(toKill);
		ds.compactWorkEfficientArbitrary ();

		cout<<"killing "<<toKill<<", "<<ds.numAlive()<<" streams remain"<<endl;

		ds.fetchDataFromGPU();

		for (int i=0; i<ds.numAlive(); i+=1){
			dataPacket cur;
			ds.getData(i, cur);
			cout<<cur.index;
			if (i<ds.numAlive()-1) cout<<",";
		}
		cout<<endl<<endl;
		bound+=1;
	}
}

void naiveSumGlobal(){
	int numElements = 256;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	ds.compactNaiveSumGlobal();

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;
}

void naiveSumSharedSingleBlock(){
	int numElements = 16;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	ds.compactNaiveSumSharedSingleBlock();

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;
}

void naiveSumSharedArbitrary(){
	int numElements = 33;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	ds.compactNaiveSumSharedArbitrary();

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;

	for (int i=0; i<numElements/(THREADS_PER_BLOCK*2); i+=1){
		cout<<ds.m_auxSums[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;
}

void workEfficientArbitrary(){
	int numElements = 512;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	// for (int i=0; i<ds.numAlive(); i+=1){
	// 	cout<<ds.m_indices[i];
	// 	if (i<ds.numAlive()-1) cout<<",";
	// }
	// cout<<endl;

	ds.compactWorkEfficientArbitrary();

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;

	for (int i=0; i<numElements/(THREADS_PER_BLOCK*2); i+=1){
		cout<<ds.m_auxSums[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;
}

int main(){
	//testStreamCompaction();
	srand (time(NULL));
	workEfficientArbitrary ();
	return 0;
}