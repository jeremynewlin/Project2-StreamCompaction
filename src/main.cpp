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
		ds.compact ();

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

int main(){
	//testStreamCompaction();
	srand (time(NULL));
	return 0;
}