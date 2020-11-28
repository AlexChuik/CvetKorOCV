#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_
#include <math.h>

namespace constants {
const double kSqrt2(sqrt(2)); 
const double kSqrt3(sqrt(3)); 
const double kSqrt6(sqrt(6)); 

//длины осей в пространстве lab
const double kAlphaAxisLenght(sqrt(2));
const double kBetaAxisLenght(4.0f / sqrt(6));
const double kLAxisLenght(3.0f / sqrt(3));

//параметр дискретизации
const int kDiscretization = 512;
}

#endif