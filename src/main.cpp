#include <iostream>

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

	// for (int i=0; i<ds.numAlive(); i+=1){
	// 	cout<<ds.m_indices[i];
	// 	if (i<ds.numAlive()-1) cout<<",";
	// }
	// cout<<endl;
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

	int ne[] = {100, 1000, 10000, 100000, 1000000, 10000000};
	for (int  i=0; i<6; i+=1){
		int numElements = ne[i];

		dataPacket * ints = new dataPacket[numElements];
		for (int i=0; i<numElements; i+=1){
			ints[i] = dataPacket(i);
		}

		DataStream ds(numElements, ints);

		// cout<<"starting with "<<ds.m_numElements<<" streams"<<endl;

		// for (int i=0; i<ds.numAlive(); i+=1){
		// 	cout<<ds.m_indices[i];
		// 	if (i<ds.numAlive()-1) cout<<",";
		// }
		// cout<<endl;

		
		ds.compactNaiveSumGlobal();

		// for (int i=0; i<ds.numAlive(); i+=1){
		// 	cout<<ds.m_indices[i];
		// 	if (i<ds.numAlive()-1) cout<<",";
		// }
		// cout<<endl;
	}
}

void naiveCompactGlobal(){
	int numElements = 100000;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0){
		for (int i=0; i<numElements/25; i+=1){
			int toKill = rand() % ds.numAlive();
			ds.kill(toKill);
		}
		//
		ds.compactNaiveSumGlobal ();
		//

		cout<<"killing ~"<<numElements/25<<" streams, "<<ds.numAlive()<<" streams remain"<<endl;

		ds.fetchDataFromGPU();

		// for (int i=0; i<ds.numAlive(); i+=1){
		// 	dataPacket cur;
		// 	ds.getData(i, cur);
		// 	cout<<cur.index;
		// 	if (i<ds.numAlive()-1) cout<<",";
		// }
		cout<<endl;
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

void compactNaiveSumSharedArbitrary(){
	int ne[] = {100, 1000, 10000, 100000, 1000000, 10000000};
	for (int  i=0; i<6; i+=1){
		int numElements = ne[i];

		dataPacket * ints = new dataPacket[numElements];
		for (int i=0; i<numElements; i+=1){
			ints[i] = dataPacket(i);
		}

		DataStream ds(numElements, ints);

		ds.compactNaiveSumGlobal();
	}
}

void naiveCompactSharedArbitrary(){
	int numElements = 32;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0 && bound < 10){

		int toKill = rand() % ds.numAlive();
		ds.kill(toKill);
		dataPacket cur;
		ds.getData(toKill, cur);
		cout<<"killed "<<cur.index<<endl;

		toKill = rand() % ds.numAlive();
		ds.kill(toKill);
		ds.getData(toKill, cur);
		cout<<"killed "<<cur.index<<endl;

		ds.compactNaiveSumSharedArbitrary ();

		cout<<ds.numAlive()<<" streams remain"<<endl;

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
	// int numElements = 40;

	// dataPacket * ints = new dataPacket[numElements];
	// for (int i=0; i<numElements; i+=1){
	// 	ints[i] = dataPacket(i);
	// }

	// DataStream ds(numElements, ints);

	// cout<<"starting with "<<numElements<<" streams"<<endl;

	// ds.compactWorkEfficientArbitrary();

	// // for (int i=0; i<ds.numAlive(); i+=1){
	// // 	cout<<ds.m_indices[i];
	// // 	if (i<ds.numAlive()-1) cout<<",";
	// // }
	// // cout<<endl;

	// // for (int i=0; i<numElements/(THREADS_PER_BLOCK*2); i+=1){
	// // 	cout<<ds.m_auxSums[i];
	// // 	if (i<ds.numAlive()-1) cout<<",";
	// // }
	// // cout<<endl;

	int numElements = 100000;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0){
		for (int i=0; i<numElements/25; i+=1){
			int toKill = rand() % ds.numAlive();
			ds.kill(toKill);
		}
		ds.compactWorkEfficientArbitrary ();

		cout<<"killing ~"<<numElements/25<<" streams, "<<ds.numAlive()<<" streams remain"<<endl;

		ds.fetchDataFromGPU();

		// for (int i=0; i<ds.numAlive(); i+=1){
		// 	dataPacket cur;
		// 	ds.getData(i, cur);
		// 	cout<<cur.index;
		// 	if (i<ds.numAlive()-1) cout<<",";
		// }
		cout<<endl;
	}
}

void workEfficientCompactSharedArbitrary(){
	int numElements = 33;

	dataPacket * ints = new dataPacket[numElements];
	for (int i=0; i<numElements; i+=1){
		ints[i] = dataPacket(i);
	}

	DataStream ds(numElements, ints);

	cout<<"starting with "<<numElements<<" streams"<<endl;

	int bound = 0;
	while(ds.numAlive () > 0 && bound < 20){
		int toKill = rand() % ds.numAlive();
		// toKill = 10;
		ds.kill(toKill);
		ds.compactWorkEfficientArbitrary ();

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

int main(){
	//testStreamCompaction();
	srand (time(NULL));
	// naiveCompactGlobal ();
	// naiveCompactSharedArbitrary ();
	naiveSumGlobal ();
	return 0;
}