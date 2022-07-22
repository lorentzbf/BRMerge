#ifndef YUSUKE_SORT_H
#define YUSUKE_SORT_H
#include <stdlib.h>


#define bits(x,k,j) ((x>>k) & ~(~0<<j))

void countsort_aux(long q, long *lKey, long *lSorted, \
		   long* auxKey, long* auxSorted, \
		   long R, long bitOff, long m) 
{
	register long j, k, last, temp, offset;
    
	static long *myHisto, *psHisto;

	long *mhp, *mps, *allHisto;

	myHisto  = (long *) malloc(R*sizeof(long));
	psHisto  = (long *) malloc(R*sizeof(long));

	mhp = myHisto;

	for (k=0 ; k<R ; k++)
		mhp[k] = 0;
    
	for (k=0; k<q; k++)
		mhp[bits(lKey[k],bitOff,m)]++;

	for (k=0; k<R; k++) {
		last = psHisto[k] = myHisto[k];
		for (j=1 ; j<1 ; j++) {
			temp = psHisto[j*R + k] = last + myHisto[j*R +  k];
			last = temp;
		}
	}

	allHisto = psHisto;
	
	offset = 0;

	mps = psHisto;

	for (k=0 ; k<R ; k++) {
		mhp[k]  = (mps[k] - mhp[k]) + offset;
		offset += allHisto[k];
	}
    
	for (k=0; k<q; k++) {
		j = bits(lKey[k],bitOff,m);
		lSorted[mhp[j]] = lKey[k];
		auxSorted[mhp[j]] = auxKey[k];
		mhp[j]++;
	}

	free(psHisto);
	free(myHisto);
}


void countingSort(long* keys, long* auxKey1, double* auxKey2, long m)
{
	long *keysSorted;
	long *index, *indexSorted;
	long i, *t1, *t2;
	double* wt;

	keysSorted = (long *) malloc(m*sizeof(long));
	index = (long *) malloc(m*sizeof(long));
	indexSorted = (long *) malloc(m*sizeof(long));

	for (i=0; i<m; i++) {
		index[i] = i;
		keysSorted[i] = 0;
			
	}

	countsort_aux(m, keys, keysSorted, index, indexSorted, (1<<11),  0, 11);

	/* for temp computation, reuse arrays instead of new malloc */
	t1 = keys;
        t2 = index;

	countsort_aux(m, keysSorted, t1, indexSorted, t2, (1<<11), 11, 11);
	countsort_aux(m, t1, keysSorted, t2, indexSorted, (1<<10), 22, 10);
	
	free(t2);

	for (i=0; i<m; i++) {
		keys[i] = keysSorted[i];
	}
	
	t1 = keysSorted;

	for (i = 0; i < m; i++) {
		t1[i] = auxKey1[indexSorted[i]];	
	}
	
	for (i=0; i<m; i++) {
		auxKey1[i] = t1[i];
	}

	free(t1);

	wt = (double *) malloc(m*sizeof(double));

        for (i = 0; i<m; i++) {
                wt[i] = auxKey2[indexSorted[i]];
        }

	for (i = 0; i<m; i++) {
		auxKey2[i] = wt[i];
	}

	free(indexSorted);
	free(wt);

}


#endif
