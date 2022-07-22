#ifndef YUSUKE_GRAPH_H
#define YUSUKE_GRAPH_H

class graph
{
	public:
	/* No. of edges, represented by m */
	long m;

	/* No. of vertices, represented by n */
	long n;
	
	/* Arrays of size 'm' storing the edge information
	 * A directed edge 'e' (0 <= e < m) from start[e] to end[e]
	 * had an integer weight w[e] */
	long* start;
	long* end;
	double* w;

	graph():m(0),n(0),start(nullptr),end(nullptr),w(nullptr){}
	~graph(){
		free(start);
		free(end);
		free(w);
	}

};

#endif