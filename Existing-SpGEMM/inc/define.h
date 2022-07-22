#ifndef _Z_DEFINE_H_
#define _Z_DEFINE_H_


#define likely(x) __builtin_expect(x,1)
#define unlikely(x) __builtin_expect(x,0)

#define div_up(a, b) ((a+b-1)/b)
#define div_round_up(a, b) ((a+b-1)/b)


#define HASH_SCALE 107

typedef int mint;
typedef double mdouble;


#endif
