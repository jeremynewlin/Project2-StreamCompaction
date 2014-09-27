#include <iostream>
#include <time.h>

#include "streamCompaction.h"

using namespace std;

void serialSum(){
	int numElements = 256;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	ds.serialScan();

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;
}

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
	int numElements = 40;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<ds.m_numElements<<" streams"<<endl;

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;

	ds.compactNaiveSumGlobal();

	for (int i=0; i<ds.numAlive(); i+=1){
		cout<<ds.m_indices[i];
		if (i<ds.numAlive()-1) cout<<",";
	}
	cout<<endl;
}

void naiveCompactGlobal(){
	int numElements = 33;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0 && bound < 1){
		int toKill = rand() % ds.numAlive();
		toKill = 10;
		ds.kill(toKill);
		ds.compactNaiveSumGlobal ();

		dataPacket cur;
		ds.getData(toKill, cur);
		cout<<"killing "<<cur.index<<", "<<ds.numAlive()<<" streams remain"<<endl;

		ds.fetchDataFromGPU();

		for (int i=0; i<ds.numAlive(); i+=1){
			ds.getData(i, cur);
			cout<<cur.index;
			if (i<ds.numAlive()-1) cout<<",";
		}
		cout<<endl<<endl;
		bound+=1;
	}
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
	int numElements = 34;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int toKill = 32;
	ds.kill(toKill);

	dataPacket cur;
	ds.getData(toKill, cur);
	cout<<"killing "<<cur.index<<", "<<ds.numAlive()<<" streams remain"<<endl;

	// for (int i=0; i<ds.numAlive(); i+=1){
		// cout<<ds.m_indices[i];
		// if (i<ds.numAlive()-1) cout<<",";
	// }
	// cout<<endl;


	ds.compactNaiveSumSharedArbitrary();
	//ds.compactNaiveSumGlobal();

	// for (int i=0; i<ds.numAlive(); i+=1){
	// 	cout<<ds.m_indices[i];
	// 	if (i<ds.numAlive()-1) cout<<",";
	// }
	// cout<<endl;

	// for (int i=0; i<numElements/(THREADS_PER_BLOCK*2); i+=1){
	// 	cout<<ds.m_auxSums[i];
	// 	if (i<ds.numAlive()-1) cout<<",";
	// }
	// cout<<endl;

}

void naiveCompactSharedArbitrary(){
	int numElements = 33;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0 && bound < 10){
		int toKill = rand() % ds.numAlive();
		// toKill = 10;
		ds.kill(toKill);
		ds.compactNaiveSumSharedArbitrary ();

		dataPacket cur;
		ds.getData(toKill, cur);
		cout<<"killing "<<cur.index<<", "<<ds.numAlive()<<" streams remain"<<endl;

		ds.fetchDataFromGPU();

		for (int i=0; i<ds.numAlive(); i+=1){
			ds.getData(i, cur);
			cout<<cur.index;
			if (i<ds.numAlive()-1) cout<<",";
		}
		cout<<endl<<endl;
		bound+=1;
	}
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
	// naiveCompactGlobal ();
	naiveCompactSharedArbitrary ();
	return 0;
}