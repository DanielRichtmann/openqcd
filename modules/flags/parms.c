
/*******************************************************************************
*
* File parms.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Utility programs for the administration of parameter sets.
*
*   int write_parms(FILE *fdat,int n,int *i,int m,double *r)
*     Writes the array length n, the array elements i[0],..,i[n-1], the array
*     length m and finally the array elements r[0],..,r[m-1] to the file fdat
*     (see the notes). The program returns the number of items written.
*
*   int read_parms(FILE *fdat,int *n,int **i,int *m,double **r)
*     Reads the array length n, allocates the array i[0] with length n and
*     reads the array elements i[0][0],..,i[0][n-1] from the file fdat. If
*     n=0 the array i[0] is set to NULL (see the notes). The program then
*     reads the array length m and the array elements r[0][0],..,r[0][m-1]
*     in the same way. The program returns the number of items read.
*
*   int check_parms(FILE *fdat,int n,int *i,int m,double *r)
*     Checks whether the parameters stored on the file fdat match the array
*     elements i[0],..,i[n-1] and r[0],..,r[m-1]. The program returns 0 if
*     no mismatch is found and 1 otherwise.
*
* These programs assume that the array lengths n,m and the integer data fit
* into 4 byte signed integers. Moreover, the IEEE-754 standard is assumed for
* double-precision floating point numbers. The program write_parms() checks
* whether these conditions are satisfied and terminates with an error message
* if not.
*
* If the machine is big endian, all numbers are byte-swapped before they are
* written to the file. Independently of the machine, they are thus stored in
* little endian format.
*
* The program read_parms() allocates the arrays i[0] and r[0] using the
* malloc() function. They can thus be deallocated by calling free().
*
* The file fdat must be open for binary writing in the case of write_parms()
* and binary reading in the case of read_parms() and check_parms(). In all
* cases the file pointer is left at the position after the last data item
* written or read.
*
*******************************************************************************/

#define PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "flags.h"


static int check_sizes(int n,int *i,int m)
{
   int k,ie;
   stdint_t s;

   ie=((n<0)||(m<0));

   s=(stdint_t)(n);
   ie|=(n!=(int)(s));

   for (k=0;k<n;k++)
   {
      s=(stdint_t)(i[k]);
      ie|=(i[k]!=(int)(s));
   }

   s=(stdint_t)(m);
   ie|=(m!=(int)(s));

   return ie;
}


int write_parms(FILE *fdat,int n,int *i,int m,double *r)
{
   int k,iw,endian;
   stdint_t istd[1];
   double dstd[1];

   endian=endianness();
   error_loc(check_sizes(n,i,m),1,"write_parms [parms.c]",
             "Parameters or the integer data elements are out of range");

   istd[0]=(stdint_t)(n);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   for (k=0;k<n;k++)
   {
      istd[0]=(stdint_t)(i[k]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);
   }

   istd[0]=(stdint_t)(m);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   for (k=0;k<m;k++)
   {
      dstd[0]=r[k];
      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);
      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   return iw;
}


int read_parms(FILE *fdat,int *n,int **i,int *m,double **r)
{
   int k,ir,endian;
   stdint_t istd[1];
   double dstd[1];

   endian=endianness();
   (*n)=0;
   i[0]=NULL;
   (*m)=0;
   r[0]=NULL;
   ir=0;

   if (fread(istd,sizeof(stdint_t),1,fdat)==1)
   {
      ir+=1;
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      (*n)=(int)(istd[0]);

      if ((*n)>0)
         i[0]=malloc((*n)*sizeof(int));

      if ((i[0]==NULL)&&((*n)!=0))
         return ir;

      for (k=0;k<(*n);k++)
      {
         ir+=fread(istd,sizeof(stdint_t),1,fdat);
         if (endian==BIG_ENDIAN)
            bswap_int(1,istd);
         i[0][k]=(int)(istd[0]);
      }
   }

   if (fread(istd,sizeof(stdint_t),1,fdat)==1)
   {
      ir+=1;
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      (*m)=(int)(istd[0]);

      if ((*m)>0)
         r[0]=malloc((*m)*sizeof(double));

      if ((r[0]==NULL)&&((*m)!=0))
         return ir;

      for (k=0;k<(*m);k++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);
         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
         r[0][k]=dstd[0];
      }
   }

   return ir;
}


int check_parms(FILE *fdat,int n,int *i,int m,double *r)
{
   int k,ir,ie,endian;
   stdint_t istd[1];
   double dstd[1];

   endian=endianness();
   ir=0;
   ie=0;

   if (fread(istd,sizeof(stdint_t),1,fdat)==1)
   {
      ir+=1;
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      if (n==(int)(istd[0]))
      {
         for (k=0;k<n;k++)
         {
            ir+=fread(istd,sizeof(stdint_t),1,fdat);
            if (endian==BIG_ENDIAN)
               bswap_int(1,istd);
            ie|=(i[k]!=(int)(istd[0]));
         }
      }
      else
         return 1;
   }
   else
      return 1;

   if (fread(istd,sizeof(stdint_t),1,fdat)==1)
   {
      ir+=1;
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      if (m==(int)(istd[0]))
      {
         for (k=0;k<m;k++)
         {
            ir+=fread(dstd,sizeof(double),1,fdat);
            if (endian==BIG_ENDIAN)
               bswap_double(1,dstd);
            ie|=(r[k]!=dstd[0]);
         }
      }
      else
         return 1;
   }
   else
      return 1;

   return (ir!=(2+n+m))|ie;
}
