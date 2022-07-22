#ifndef YUSUKE_CSR_H_
#define YUSUKE_CSR_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "CSC.h"
#include "Triple.h"
#include <string>
#include <sstream>
//#include <tbb/scalable_allocator.h>
#include <cassert>
#include <random>
#include "utility.h"
#include "define.h"

using namespace std;

template <class IT, class NT>
class CSR
{ 
public:
    CSR():nnz(0), rows(0), cols(0),zerobased(true) {}
	CSR(IT mynnz, IT m, IT n):nnz(mynnz),rows(m),cols(n),zerobased(true)
	{
        // Constructing empty Csc objects (size = 0) are allowed (why wouldn't they?).
        assert(rows != 0);
        rowptr = my_malloc<IT>(rows + 1);
		if(nnz > 0) {
            colids = my_malloc<IT>(nnz);
            values = my_malloc<NT>(nnz);
        }
	}
    CSR (IT *row_pointer, IT* col_index, NT* csr_values, IT M, IT N, IT nnz_, int my);
    CSR (graph & G);
    CSR (string filename);
    void construct(string filename);
    CSR (const CSR<IT, NT> &A, IT M_, IT N_, IT M_start, IT N_start);
    CSR (const CSC<IT,NT> & csc);   // CSC -> CSR conversion
    CSR (const CSR<IT,NT> & rhs);	// copy constructor
    CSR (const CSC<IT,NT> & csc, const bool transpose);
	CSR<IT,NT> & operator=(const CSR<IT,NT> & rhs);	// assignment operator
    bool operator==(const CSR<IT,NT> & rhs); // ridefinizione ==
    void shuffleIds(); // Randomly permutating column indices
    void sortIds(); // Permutating column indices in ascending order
    
    void make_empty()
    {
        if(nnz > 0) {
            my_free<IT>(colids);
            my_free<NT>(values);
            nnz = 0;
        }
        if(rows > 0) {
            my_free<IT>(rowptr);
            rows = 0;
        }
        cols = 0;	
    }
    
    ~CSR()
	{
        make_empty();
	}
    bool ConvertOneBased()
    {
        if(!zerobased)	// already one-based
            return false; 
        transform(rowptr, rowptr + rows + 1, rowptr, bind2nd(plus<IT>(), static_cast<IT>(1)));
        transform(colids, colids + nnz, colids, bind2nd(plus<IT>(), static_cast<IT>(1)));
        zerobased = false;
        return true;
    }
    bool ConvertZeroBased()
    {
        if (zerobased)
            return true;
        transform(rowptr, rowptr + rows + 1, rowptr, bind2nd(plus<IT>(), static_cast<IT>(-1)));
        transform(colids, colids + nnz, colids, bind2nd(plus<IT>(), static_cast<IT>(-1)));
        zerobased = true;
        return false;
    }
    bool isEmpty()
    {
        return ( nnz == 0 );
    }
    void Sorted();
    
	IT rows;	
	IT cols;
	IT nnz; // number of nonzeros
    
    IT *rowptr;
    IT *colids;
    NT *values;
    bool zerobased;
};

template <class IT, class NT> 
CSR<IT, NT>::CSR (IT *row_pointer, IT* col_index, NT* csr_values, IT M, IT N, IT nnz_, int my){
    rows = M + my -my;
    cols = N;
    nnz = nnz_;
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);
    memcpy(rowptr, row_pointer, (rows + 1)*sizeof(IT));
    memcpy(colids, col_index, nnz*sizeof(IT));
    memcpy(values, csr_values, nnz*sizeof(NT));
}

// copy constructor
template <class IT, class NT>
CSR<IT,NT>::CSR (const CSR<IT,NT> & rhs): nnz(rhs.nnz), rows(rhs.rows), cols(rhs.cols),zerobased(rhs.zerobased)
{
	if(nnz > 0)
	{
        values = my_malloc<NT>(nnz);
        colids = my_malloc<IT>(nnz);
        copy(rhs.values, rhs.values+nnz, values);
        copy(rhs.colids, rhs.colids+nnz, colids);
	}
	if ( rows > 0)
	{
        rowptr = my_malloc<IT>(rows + 1);
        copy(rhs.rowptr, rhs.rowptr+rows+1, rowptr);
	}
}

template <class IT, class NT>
CSR<IT,NT> & CSR<IT,NT>::operator= (const CSR<IT,NT> & rhs)
{
	if(this != &rhs)		
	{
		if(nnz > 0)	// if the existing object is not empty
		{
            my_free<IT>(colids);
            my_free<NT>(values);
		}
		if(rows > 0)
		{
            my_free<IT>(rowptr);
		}

		nnz	= rhs.nnz;
		rows = rhs.rows;
		cols = rhs.cols;
		zerobased = rhs.zerobased;
		if(rhs.nnz > 0)	// if the copied object is not empty
		{
            values = my_malloc<NT>(nnz);
            colids = my_malloc<IT>(nnz);
            copy(rhs.values, rhs.values+nnz, values);
            copy(rhs.colids, rhs.colids+nnz, colids);
		}
		if(rhs.cols > 0)
		{
            rowptr = my_malloc<IT>(rows + 1);
            copy(rhs.rowptr, rhs.rowptr+rows+1, rowptr);
		}
	}
	return *this;
}

//! Construct a CSR object from a CSC
//! Accepts only zero based CSC inputs
template <class IT, class NT>
CSR<IT,NT>::CSR(const CSC<IT,NT> & csc):nnz(csc.nnz), rows(csc.rows), cols(csc.cols),zerobased(true)
{
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    IT *work = my_malloc<IT>(rows);
    std::fill(work, work+rows, (IT) 0); // initilized to zero
   
    	for (IT k = 0 ; k < nnz ; ++k)
    	{
        	IT tmp =  csc.rowids[k];
        	work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
	}

	if(nnz > 0)
	{
		rowptr[rows] = CumulativeSum (work, rows);		// cumulative sum of w
       	 	copy(work, work+rows, rowptr);

		IT last;
        	for (IT i = 0; i < cols; ++i) 
        	{
       	     		for (IT j = csc.colptr[i]; j < csc.colptr[i+1] ; ++j)
            		{
				colids[ last = work[ csc.rowids[j] ]++ ]  = i ;
				values[last] = csc.values[j] ;
            		}
        	}
	}
    my_free<IT>(work);
}

template <class IT, class NT>
CSR<IT,NT>::CSR(const CSC<IT,NT> & csc, const bool transpose):nnz(csc.nnz), rows(csc.rows), cols(csc.cols),zerobased(true)
{
    if (!transpose) {
        rowptr = my_malloc<IT>(rows + 1);
        colids = my_malloc<IT>(nnz);
        values = my_malloc<NT>(nnz);

        IT *work = my_malloc<IT>(rows);
        std::fill(work, work+rows, (IT) 0); // initilized to zero
   
    	for (IT k = 0 ; k < nnz ; ++k)
            {
                IT tmp =  csc.rowids[k];
                work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
            }

        if(nnz > 0) 
            {
                rowptr[rows] = CumulativeSum (work, rows);		// cumulative sum of w
                copy(work, work+rows, rowptr);

                IT last;
                for (IT i = 0; i < cols; ++i) 
                    {
                        for (IT j = csc.colptr[i]; j < csc.colptr[i+1] ; ++j)
                            {
                                colids[ last = work[ csc.rowids[j] ]++ ]  = i ;
                                values[last] = csc.values[j] ;
                            }
                    }
            }
        my_free<IT>(work);
    }
    else {
        rows = csc.cols;
        cols = csc.rows;
        rowptr = my_malloc<IT>(rows + 1);
        colids = my_malloc<IT>(nnz);
        values = my_malloc<NT>(nnz);

        for (IT k = 0; k < rows + 1; ++k) {
            rowptr[k] = csc.colptr[k];
        }
        for (IT k = 0; k < nnz; ++k) {
            values[k] = csc.values[k];
            colids[k] = csc.rowids[k];
        }
    }
}

template <class IT, class NT>
CSR<IT,NT>::CSR(graph & G):nnz(G.m), rows(G.n), cols(G.n), zerobased(true)
{
	// graph is like a triples object
	// typedef struct {
        // LONG_T m;
        // LONG_T n;
        // // Arrays of size 'm' storing the edge information
        // // A directed edge 'e' (0 <= e < m) from start[e] to end[e]
        // // had an integer weight w[e] 
        // LONG_T* start;
        // LONG_T* end; 
	// WEIGHT_T* w;
	// } graph; 
	cout << "Graph nnz= " << G.m << " and n=" << G.n << endl;

	vector< Triple<IT,NT> > simpleG;
	vector< pair< pair<IT,IT>,NT> > currCol;
	currCol.push_back(make_pair(make_pair(G.start[0], G.end[0]), G.w[0]));
	for (IT k = 0 ; k < nnz-1 ; ++k) {
        if(G.start[k] != G.start[k+1] ) {
            std::sort(currCol.begin(), currCol.end());
            simpleG.push_back(Triple<IT,NT>(currCol[0].first.first, currCol[0].first.second, currCol[0].second));
            for(int i=0; i< currCol.size()-1; ++i) {
                if(currCol[i].first == currCol[i+1].first) {
                    simpleG.back().val += currCol[i+1].second;
                }
                else {	
                    simpleG.push_back(Triple<IT,NT>(currCol[i+1].first.first, currCol[i+1].first.second, currCol[i+1].second));
                }
            }
            vector< pair< pair<IT,IT>,NT> >().swap(currCol);
        }
		currCol.push_back(make_pair(make_pair(G.start[k+1], G.end[k+1]), G.w[k+1]));
    }
    
	// now do the last row
	sort(currCol.begin(), currCol.end());
    simpleG.push_back(Triple<IT,NT>(currCol[0].first.first, currCol[0].first.second, currCol[0].second));
    for(int i=0; i< currCol.size()-1; ++i) {
        if(currCol[i].first == currCol[i+1].first) {
            simpleG.back().val += currCol[i+1].second;
        }
		else {
            simpleG.push_back(Triple<IT,NT>(currCol[i+1].first.first, currCol[i+1].first.second, currCol[i+1].second));
        }
    }

	nnz = simpleG.size();
	cout << "[After duplicate merging] Graph nnz= " << nnz << " and n=" << G.n << endl;

    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    IT *work = my_malloc<IT>(rows);
    std::fill(work, work+rows, (IT) 0); // initilized to zero
    
    for (IT k = 0 ; k < nnz ; ++k) {
        IT tmp =  simpleG[k].row;
        work [ tmp ]++ ;		// col counts (i.e, w holds the "col difference array")
	}

	if(nnz > 0) {
        rowptr[rows] = CumulativeSum (work, rows) ;		// cumulative sum of w
        copy(work, work + rows, rowptr);
        
		IT last;
		for (IT k = 0 ; k < nnz ; ++k) {
            colids[ last = work[ simpleG[k].row ]++ ]  = simpleG[k].col ;
			values[last] = simpleG[k].val ;
        }
	}
    my_free<IT>(work);
}


// check if sorted within rows?
template <class IT, class NT>
void CSR<IT,NT>::Sorted()
{
	bool sorted = true;
	for(IT i=0; i< rows; ++i)
	{
		sorted &= my_is_sorted (colids + rowptr[i], colids + rowptr[i+1], std::less<IT>());
    }
}

template <class IT, class NT>
bool CSR<IT,NT>::operator==(const CSR<IT,NT> & rhs)
{
    bool same;
    if(nnz != rhs.nnz || rows  != rhs.rows || cols != rhs.cols) {
        printf("%d:%d, %d:%d, %d:%d\n", nnz, rhs.nnz, rows, rhs.rows, cols, rhs.cols);
        return false;
    }  
    if (zerobased != rhs.zerobased) {
        IT *tmp_rowptr = my_malloc<IT>(rows + 1);
        IT *tmp_colids = my_malloc<IT>(nnz);
        if (!zerobased) {
            for (int i = 0; i < rows + 1; ++i) {
                tmp_rowptr[i] = rowptr[i] - 1;
            }
            for (int i = 0; i < nnz; ++i) {
                tmp_colids[i] = colids[i] - 1;
            }
            same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rhs.rowptr); 
            same = same && std::equal(tmp_colids, tmp_colids + nnz, rhs.colids);
        }
        else if (!rhs.zerobased) {
            for (int i = 0; i < rows + 1; ++i) {
                tmp_rowptr[i] = rhs.rowptr[i] - 1;
            }
            for (int i = 0; i < nnz; ++i) {
                tmp_colids[i] = rhs.colids[i] - 1;
            }
            same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rowptr); 
            same = same && std::equal(tmp_colids, tmp_colids + nnz, colids);

        }
        my_free<IT>(tmp_rowptr);
        my_free<IT>(tmp_colids);
    }
    else {
        same = std::equal(rowptr, rowptr+rows+1, rhs.rowptr); 
        same = same && std::equal(colids, colids+nnz, rhs.colids);
    }
    
    bool samebefore = same;
    ErrorTolerantEqual<NT> epsilonequal(EPSILON);
    same = same && std::equal(values, values+nnz, rhs.values, epsilonequal );
    //if(samebefore && (!same)) {
#ifdef DEBUG
        vector<NT> error(nnz);
        transform(values, values+nnz, rhs.values, error.begin(), absdiff<NT>());
        vector< pair<NT, NT> > error_original_pair(nnz);
        for(IT i=0; i < nnz; ++i)
            error_original_pair[i] = make_pair(error[i], values[i]);
        if(error_original_pair.size() > 10) { // otherwise would crush for small data
            partial_sort(error_original_pair.begin(), error_original_pair.begin()+10, error_original_pair.end(), greater< pair<NT,NT> >());
            cout << "Highest 10 different entries are: " << endl;
            for(IT i=0; i < 10; ++i)
                cout << "Diff: " << error_original_pair[i].first << " on " << error_original_pair[i].second << endl;
        }
        else {
            sort(error_original_pair.begin(), error_original_pair.end(), greater< pair<NT,NT> >());
            cout << "Highest different entries are: " << endl;
            for(typename vector< pair<NT, NT> >::iterator it=error_original_pair.begin(); it != error_original_pair.end(); ++it)
                cout << "Diff: " << it->first << " on " << it->second << endl;
        }
#endif
            //}
    return same;
}
#define SYMMETRY_GENERAL 0
#define SYMMETRY_SYMMETRY 1
#define SYMMETRY_SKEW_SYMMETRY 2
#define SYMMETRY_HERMITIAN 3
struct matrix_market_banner
{
    std::string matrix; // "matrix" or "vector"
    std::string storage;    // "array" or "coordinate", storage_format
    std::string type;       // "complex", "real", "integer", or "pattern"
    std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric"
};

inline void tokenize(std::vector<std::string>& tokens, const std::string str, const std::string delimiters = "\n\r\t ")
{
    tokens.clear();
    // Skip delimiters at beginning.
    std::string::size_type first_pos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type last_pos     = str.find_first_of(delimiters, first_pos);

    while (std::string::npos != first_pos || std::string::npos != last_pos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(first_pos, last_pos - first_pos));
        // Skip delimiters.  Note the "not_of"
        first_pos = str.find_first_not_of(delimiters, last_pos);
        // Find next "non-delimiter"
        last_pos = str.find_first_of(delimiters, first_pos);
    }
}

template <typename Stream>
void read_mm_banner(Stream& input, matrix_market_banner& banner)
{
    std::string line;
    std::vector<std::string> tokens;

    // read first line
    std::getline(input, line);
    tokenize(tokens, line);

    if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
        throw std::runtime_error("invalid MatrixMarket banner");

    banner.matrix = tokens[1]; // mow just matrix, no vector
    banner.storage  = tokens[2]; // now just coordinate(sparse), no array(dense)
    banner.type     = tokens[3]; // int, real, pattern for double, complex for two double
    banner.symmetry = tokens[4]; // general, symmetry, etc

    if(banner.matrix != "matrix" && banner.matrix != "vector")
        throw std::runtime_error("invalid MatrixMarket matrix type: " + banner.matrix);
    if(banner.matrix == "vector")
        throw std::runtime_error("not impl matrix type: " + banner.matrix);

    if (banner.storage != "array" && banner.storage != "coordinate")
        throw std::runtime_error("invalid MatrixMarket storage format [" + banner.storage + "]");
    if(banner.storage == "array")
        throw std::runtime_error("not impl storage type "+ banner.storage);

    if (banner.type != "complex" && banner.type != "real" && banner.type != "integer" && banner.type != "pattern")
        throw std::runtime_error("invalid MatrixMarket data type [" + banner.type + "]");
    //if(banner.type == "complex")
    //    throw std::runtime_error("not impl data type: " + banner.type);

    if (banner.symmetry != "general" && banner.symmetry != "symmetric" && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
        throw std::runtime_error("invalid MatrixMarket symmetry [" + banner.symmetry + "]");
    if(banner.symmetry == "hermitian")
        throw std::runtime_error("not impl matrix type: " + banner.symmetry);
    
}

template <class IT, class NT>
CSR<IT,NT>::CSR(const string filename): zerobased(true){
    construct(filename);
}

template <class IT, class NT>
void CSR<IT,NT>::construct(const string filename)
{
    std::ifstream ifile(filename.c_str());
    if(!ifile){
        throw std::runtime_error(std::string("unable to open file \"") + filename + std::string("\" for reading"));
    }
    matrix_market_banner banner;
    // read mtx header
    read_mm_banner(ifile, banner);

    // read file contents line by line
    std::string line;

    // skip over banner and comments
    do
    {
        std::getline(ifile, line);
    } while (line[0] == '%');

    // line contains [num_rows num_columns num_entries]
    std::vector<std::string> tokens;
    tokenize(tokens, line);

    if (tokens.size() != 3)
        throw std::runtime_error("invalid MatrixMarket coordinate format");

    std::istringstream(tokens[0]) >> rows;
    std::istringstream(tokens[1]) >> cols;
    std::istringstream(tokens[2]) >> nnz;
    assert(nnz > 0 && "something wrong: nnz is 0");

    IT *I_ = new IT [nnz];
    IT *J_ = new IT [nnz];
    NT *coo_values_ = new NT [nnz];

    IT num_entries_read = 0;

    // read file contents
    if (banner.type == "pattern")
    {
        while(num_entries_read < nnz && !ifile.eof())
        {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            num_entries_read++;
        }
        std::fill(coo_values_, coo_values_ + nnz, NT(1));
    }
    else if (banner.type == "real" || banner.type == "integer")
    {
        while(num_entries_read < nnz && !ifile.eof())
        {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            ifile >> coo_values_[num_entries_read];
            num_entries_read++;
        }
    }
    else if (banner.type == "complex")
    {
        NT tmp;
        while(num_entries_read < nnz && !ifile.eof())
        {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            ifile >> coo_values_[num_entries_read] >> tmp;
            num_entries_read++;
        }
    }
    else
    {
        throw std::runtime_error("invalid MatrixMarket data type");
    }
    ifile.close();

    if(num_entries_read != nnz)
        throw std::runtime_error("read nnz not equal to decalred nnz " + std::to_string(num_entries_read));

    // convert base-1 indices to base-0
    for(IT n = 0; n < nnz; n++){
        I_[n] -= 1;
        J_[n] -= 1;
    }

    // expand symmetric formats to "general" format
    if (banner.symmetry != "general"){
        IT non_diagonals = 0;

        for (IT n = 0; n < nnz; n++)
            if(likely(I_[n] != J_[n]))
                non_diagonals++;

        IT new_nnz = nnz + non_diagonals;

        IT* new_I = new IT [new_nnz];
        IT* new_J = new IT [new_nnz];
        NT *new_coo_values;
        new_coo_values = new NT [new_nnz];
        

        if (banner.symmetry == "symmetric"){
            IT cnt = 0;
            for (IT n = 0; n < nnz; n++){
                // copy entry over
                new_I[cnt] = I_[n];
                new_J[cnt] = J_[n];
                new_coo_values[cnt] = coo_values_[n];
                cnt++;

                // duplicate off-diagonals
                if (I_[n] != J_[n]){
                    new_I[cnt] = J_[n];
                    new_J[cnt] = I_[n];
                    new_coo_values[cnt] = coo_values_[n];
                    cnt++;
                }
            }
            assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
        }
        else if (banner.symmetry == "skew-symmetric"){
            IT cnt = 0;
            for (IT n = 0; n < nnz; n++){
                // copy entry over
                new_I[cnt] = I_[n];
                new_J[cnt] = J_[n];
                new_coo_values[cnt] = coo_values_[n];
                cnt++;

                // duplicate off-diagonals
                if (I_[n] != J_[n]){
                    new_I[cnt] = J_[n];
                    new_J[cnt] = I_[n];
                    new_coo_values[cnt] = -coo_values_[n];
                    cnt++;
                }
            }
            assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
        }
        else if (banner.symmetry == "hermitian"){
            // TODO
            throw std::runtime_error("MatrixMarket I/O does not currently support hermitian matrices");
        }

        // store full matrix in coo
        nnz = new_nnz;
        delete [] I_;
        delete [] J_;
        delete [] coo_values_;
        I_ = new_I;
        J_ = new_J;
        coo_values_ = new_coo_values;
    } // if (banner.symmetry != "general")

    // sort indices by (row,column)
    Pair<long, double> *p = new Pair<long, double> [nnz];
    for(IT i = 0; i < nnz; i++){
        p[i].ind = (long int)cols * I_[i] + J_[i];
        p[i].val = coo_values_[i];
    }
    std::sort(p, p + nnz);
    for(IT i = 0; i < nnz; i++){
        I_[i] = p[i].ind / cols;
        J_[i] = p[i].ind % cols;
        coo_values_[i] = p[i].val;
    }
    delete [] p;
    
    // coo to csr
    rowptr = my_malloc<IT>(rows + 1);
    memset(rowptr, 0, (rows + 1) * sizeof(IT));
    for(IT i = 0; i < nnz; i++){
        rowptr[I_[i]+1]++;
    }
    for(IT i = 1; i <= rows; i++){
        rowptr[i] += rowptr[i-1];
    }
    delete [] I_;
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);
    memcpy(colids, J_, nnz*sizeof(IT));
    memcpy(values, coo_values_, nnz*sizeof(NT));
    delete [] J_;
    delete [] coo_values_;
}

template <class IT, class NT>
void CSR<IT,NT>::shuffleIds()
{
    mt19937_64 mt(0);
    for (IT i = 0; i < rows; ++i) {
        IT offset = rowptr[i];
        IT width = rowptr[i + 1] - rowptr[i];
        uniform_int_distribution<IT> rand_scale(0, width - 1);
        for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IT target = rand_scale(mt);
            IT tmpId = colids[offset + target];
            NT tmpVal = values[offset + target];
            colids[offset + target] = colids[j];
            values[offset + target] = values[j];
            colids[j] = tmpId;
            values[j] = tmpVal;
        }
    }
}

template <class IT, class NT>
CSR<IT,NT>::CSR(const CSR<IT, NT> &A, IT M_, IT N_, IT M_start = 0, IT N_start = 0)
{
    assert(M_ + M_start <= A.rows && "matrix subsect error M");
    assert(N_ + N_start <= A.cols && "matrix subsect error N");
    IT M_end = M_start + M_;
    IT N_end = N_start + N_;
    rows = M_;
    cols = N_;
    IT *row_size = new IT [rows];
    memset(row_size, 0, rows*sizeof(IT));
    for(IT i = M_start; i < M_end; i++){
        for(IT j = A.rowptr[i]; j < A.rowptr[i+1]; j++){
            if(A.colids[j]>= N_start && A.colids[j] < N_end){
                row_size[i - M_start]++;
            }
        }
    }

    //rowptr = new IT [rows + 1];
    rowptr = my_malloc<IT>(rows + 1);
    rowptr[0] = 0;
    for(IT i = 0; i < rows; i++){
        rowptr[i+1] = rowptr[i] + row_size[i];
    }
    nnz = rowptr[rows];
    delete [] row_size;

    //colids = new IT [nnz];
    //values = new NT [nnz];
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);
    for(IT i = M_start; i < M_end; i++){
        IT jj = rowptr[i - M_start];
        for(IT j = A.rowptr[i]; j < A.rowptr[i+1]; j++){
            if(A.colids[j]>= N_start && A.colids[j] < N_end){
                colids[jj] = A.colids[j] - N_start;
                values[jj++] = A.values[j];
            }
        }
    }

}

#endif
