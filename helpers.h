/** \file
 *  Helper functions for test suite.
 *
 *  @cond TEST
 */
#pragma once
#include <cmath>
#include "nd.h"

///// helpers

/**
 * Templated ndcast() operations.
 */
template<class T> nd_t cast(nd_t a);

/**
 * Saturate value to type max and min.
 */
template<class T> T clamp(double v);

/**
 * Root mean squared error
 */
template<class T>
double RMSE(size_t n, T* a, T* b)
{ double ssq=0.0;
  for(size_t i=0;i<n;++i)
  { double t = (double)b[i]-(double)a[i];
    //if(a[i]!=b[i])
    ssq+=t*t;
  }
  return sqrt(ssq/n);
}
template<class T1,class T2>
double RMSE2(size_t n, T1* a, T2* b)
{ double ssq=0.0;
  for(size_t i=0;i<n;++i)
  { double t = (double)b[i]-(double)a[i];
    //if(a[i]!=b[i])
    ssq+=t*t;
  }
  return sqrt(ssq/n);
}
double ndRMSE(nd_t a, nd_t b);

/**
 * Find first difference. 
 * \returns the index of the first difference, or -1 if none found
 */
template<class Ta,class Tb>
int firstdiff(size_t n, const Ta* a, const Tb* b, const double tol=1e-3)
{ for(size_t i=0;i<n;++i)
    if(fabs((double)a[i]-(double)(Ta)b[i])>tol)
      return (int)i;
  return -1;
}

/**
 * Find first difference. 
 * \returns the index of the first difference, or -1 if none found
 */
template<class Ta,class Tb>
int firstdiff_clamped(size_t n, const Ta* a, const Tb* b, const double tol)
{ for(size_t i=0;i<n;++i)
    if(fabs((double)a[i]-(double)clamp<Ta>(b[i]))>tol)
      return (int)i;
  return -1;
}

/**
 * Allocates an array with shape \a shape and fills it with zeros.
 */
template<class T>
T* zeros(size_t ndim, size_t* shape)
{ size_t i,nelem;
  nelem = shape[0];
  for(i=1;i<ndim;++i)
    nelem*=shape[i];
  T* v = new T[nelem];
  memset(v,0,nelem*sizeof(T));
}

/**
 * Returns index of first value in \a a not equal to \a v.
 * Or -1, if no such index exists.
 */
template<class T>
int all(size_t n, const T* a, const T v)
{ for(size_t i=0;i<n;++i)
    if(a[i]!=v)
      return i;
  return -1;
}
/// @endcond
