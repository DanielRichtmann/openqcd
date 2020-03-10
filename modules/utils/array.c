
/*******************************************************************************
*
* File array.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation of multi-dimensional arrays.
*
*   array_t *alloc_array(unsigned int d,size_t *n,size_t size,
*                        unsigned int p)
*     Allocates a multi-dimensional array of data elements of the specified
*     size and returns a structure containing the array and its parameters.
*     Apart from the data element size, the dimension d and the array sizes
*     n[0],..,n[d-1] must be specified as well as the desired alignment of
*     the data (see the notes). The data elements are initialized to 0.
*
*   void free_array(array_t *a)
*     Frees the array described by the array structure "a", then frees
*     the structure elements and finally the structure itself. The array
*     is assumed to have been allocated by the program alloc_array().
*
*   void write_array(FILE *fdat,array_t *a)
*     Writes the array parameters d, n[0],..,n[d-1], size and the elements
*     of the array in the array structure "a" to the file fdat, assuming the
*     latter is open for binary writing. The array parameters must fit into
*     unsigned 4-byte integers and the data elements must have size equal
*     to 4 or 8. If the machine is big endian, the parameters and data are
*     byte-swapped before they are written to the file.
*
*   void read_array(FILE *fdat,array_t *a)
*     Reads the elements of the structure "a" from the file fdat, assuming
*     the latter is open for binary reading. The array must previously have
*     been allocated by calling alloc_array() and the data on the file must
*     have been written by the program write_array(). An error occurs if
*     the array parameters do not match.
*
* The structures of type array_t have the following elements (see utils.h):
*
*  d                    Dimension of the array (d>=1).
*
*  n[0],..,n[d-1]       Array sizes in dimension 0,..,d-1 (n[i]>=1).
*
*  size                 Size in bytes of the basic data elements [as
*                       determined by sizeof(data_type)].
*
*  p                    Specifies the alignment in memory of the first
*                       data element (p>=0).
*
*  a                    Address of the array.
*
* The aligned data allocation is performed by calling amalloc() [utils.c]
* in such a way that the address of the first data element is a multiple
* of 2^p. If p=0 the data are allocated using the malloc() function.
*
* The array is allocated such that the n[0]*n[1]*..*n[d-1] data elements are
* stored contiguously in memory.
*
*******************************************************************************/

#define ARRAY_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utils.h"

static unsigned int endian=UNKNOWN_ENDIAN;
static size_t max_size;


static void set_constants(void)
{
   if (endian==UNKNOWN_ENDIAN)
   {
      endian=endianness();
      error_loc(endian==UNKNOWN_ENDIAN,1,"set_constants [array.c]",
                "Unknown endianness");
      max_size=0;
      max_size=~max_size;
   }
}


static int check_sizes1(unsigned int d,size_t *n,size_t size,unsigned int p)
{
   unsigned int k;
   size_t m;

   if ((d==0)||(p>(8*sizeof(size_t))))
      return 1;
   m=1;

   for (k=0;k<d;k++)
   {
      if ((n[k]==0)||(m>(max_size/n[k])))
         return 1;
      m*=n[k];
   }

   if ((size==0)||(m>(max_size/size)))
      return 1;

   return 0;
}


static int check_sizes2(unsigned int d,size_t *n,size_t size)
{
   unsigned int k;
   stduint_t u;

   u=(stduint_t)(d);
   if (d!=(unsigned int)(u))
      return 1;

   for (k=0;k<d;k++)
   {
      u=(stduint_t)(n[k]);
      if (n[k]!=(size_t)(u))
         return 1;
   }

   if ((size!=4)&&(size!=8))
      return 1;

   return 0;
}


static array_t *alloc_astruct(unsigned int d)
{
   size_t *n;
   array_t *a;

   a=malloc(1*sizeof(*a));
   n=malloc(d*sizeof(*n));

   error_loc((a==NULL)||(n==NULL),1,"alloc_astruct [array.c]",
             "Unable to allocate array_t structure");

   (*a).d=d;
   (*a).n=n;

   return a;
}


static char *alloc_base(size_t size,unsigned int p)
{
   char *b;

   if (p==0)
      b=malloc(size);
   else
      b=amalloc(size,p);

   error_loc(b==NULL,1,"alloc_base [array.c]",
             "Unable to allocate data array");

   return (char*)(memset(b,0,size));
}


static void **alloc_ptarr(size_t size)
{
   void **p;

   p=malloc(size*sizeof(*p));

   error_loc(p==NULL,1,"alloc_ptarr [array.c]",
             "Unable to allocate pointer array");

   return p;
}


array_t *alloc_array(unsigned int d,size_t *n,size_t size,unsigned int p)
{
   unsigned int i;
   size_t n0,n1,k;
   char *b;
   void **p0,**p1;
   array_t *a;

   set_constants();
   error_loc(check_sizes1(d,n,size,p),1,"alloc_array [arrays.c]",
             "Array parameters are out of range");
   a=alloc_astruct(d);

   for (i=0;i<d;i++)
      (*a).n[i]=n[i];

   (*a).size=size;
   (*a).p=p;

   if (d==1)
      (*a).a=(void*)(alloc_base(n[0]*size,p));
   else
   {
      n0=n[0];
      p0=alloc_ptarr(n0);
      (*a).a=(void*)(p0);

      for (i=1;i<d;i++)
      {
         n1=n0*n[i];

         if (i<(d-1))
         {
            p1=alloc_ptarr(n1);

            for (k=0;k<n0;k++)
               p0[k]=(void*)(p1+k*n[i]);

            p0=p1;
            n0=n1;
         }
         else
         {
            b=alloc_base(n1*size,p);

            for (k=0;k<n0;k++)
               p0[k]=(void*)(b+k*n[i]*size);
         }
      }
   }

   return a;
}


void free_array(array_t *a)
{
   unsigned int d,p,i;
   void **p0,**p1;

   d=(*a).d;
   p=(*a).p;
   p0=(void**)((*a).a);

   for (i=0;i<(d-1);i++)
   {
      p1=(void**)(p0[0]);
      free(p0);
      p0=p1;
   }

   if (p==0)
      free(p0);
   else
      afree(p0);

   free((*a).n);
   free(a);
}


void write_array(FILE *fdat,array_t *a)
{
   unsigned int d,i;
   size_t size,*n;
   size_t m,k,iw;
   stduint_t istd[1];
   char w[8],*wa;
   void **p0;

   set_constants();
   d=(*a).d;
   n=(*a).n;
   size=(*a).size;
   error_loc(check_sizes2(d,n,size),1,"write_array [arrays.c]",
             "Unexpected array parameter or data sizes ");

   p0=(void**)((*a).a);
   m=n[0];

   for (i=1;i<d;i++)
   {
      p0=(void**)(p0[0]);
      m*=n[i];
   }

   istd[0]=(stduint_t)(d);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw=fwrite(istd,sizeof(stduint_t),1,fdat);

   for (i=0;i<d;i++)
   {
      istd[0]=(stduint_t)(n[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stduint_t),1,fdat);
   }

   istd[0]=(stduint_t)(size);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stduint_t),1,fdat);

   wa=(char*)(p0);

   if (endian==BIG_ENDIAN)
   {
      for (k=0;k<m;k++)
      {
         if (size==4)
            bswap_int(1,memcpy(w,wa,size));
         else
            bswap_double(1,memcpy(w,wa,size));

         iw+=fwrite(w,size,1,fdat);
         wa+=size;
      }
   }
   else
      iw+=fwrite(wa,size,m,fdat);

   error_loc(iw!=(2+d+m),1,"write_array [array.c]","Incorrect write count");
}


void read_array(FILE *fdat,array_t *a)
{
   unsigned int d,i,ie;
   size_t size,*n;
   size_t m,k,ir;
   stduint_t istd[1];
   char w[8],*wa;
   void **p0;

   set_constants();
   d=(*a).d;
   n=(*a).n;
   size=(*a).size;

   p0=(void**)((*a).a);
   m=n[0];

   for (i=1;i<d;i++)
   {
      p0=(void**)(p0[0]);
      m*=n[i];
   }

   ir=fread(istd,sizeof(stduint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie=((unsigned int)(istd[0])!=d);

   for (i=0;i<d;i++)
   {
      ir+=fread(istd,sizeof(stduint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie|=((size_t)(istd[0])!=n[i]);
   }

   ir+=fread(istd,sizeof(stduint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie|=((size_t)(istd[0])!=size);

   error_loc(ie!=0,1,"read_array [array.c]","Unexpected array parameters");

   wa=(char*)(p0);

   if (endian==BIG_ENDIAN)
   {
      for (k=0;k<m;k++)
      {
         ir+=fread(w,size,1,fdat);

         if (size==4)
            bswap_int(1,memcpy(wa,w,size));
         else
            bswap_double(1,memcpy(wa,w,size));

         wa+=size;
      }
   }
   else
      ir+=fread(wa,size,m,fdat);

   error_loc(ir!=(2+d+m),1,"read_array [array.c]","Incorrect read count");
}
