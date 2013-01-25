
/*******************************************************************************
*
* File ranlux.c
*
* Copyright (C) 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Initialization of the ranlux generators
*
* The externally accessible function is
*
*   void start_ranlux(int level,int seed)
*     Initializes the random number generators ranlxs and ranlxd on all
*     processes in different ways. The luxury level should be 0 (recommended)
*     or 1 (exceptional) and the seed can be any positive integer less than
*     or equal to INT_MAX/NPROC. An error occurs if the seed is not in this
*     range.
*
* Notes:
*
* This program guarantees that all generators are initialized with different
* seed values. Moreover, the initialization is guaranteed to be pairwise
* different from the previous one when start_ranlux() is called a second
* time with another value of "seed". The program acts globally and must be
* called from all processes simultaneously.
*
*******************************************************************************/

#define RANLUX_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"
#include "utils.h"
#include "random.h"
#include "global.h"


void start_ranlux(int level,int seed)
{
   int my_rank,max_seed,loc_seed,iprms[2];

   if (NPROC>1)
   {
      iprms[0]=level;
      iprms[1]=seed;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);
   
      error((iprms[0]!=level)||(iprms[1]!=seed),1,
            "start_ranlux [ranlux.c]","Input parameters are not global");
   }
   
   max_seed=INT_MAX/NPROC;

   error_root((level<0)||(level>1)||(seed<1)||(seed>max_seed),1,
              "start_ranlux [ranlux.c]","Parameters are out of range");

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   loc_seed=seed+my_rank*max_seed;

   rlxs_init(level,loc_seed);
   rlxd_init(level+1,loc_seed);
}
