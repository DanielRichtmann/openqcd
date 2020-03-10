
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2011-2013, 2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the program sw_term().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sw_term.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,bc,is,count,nt;
   double phi[2],phi_prime[2],theta[3];
   double wt1,wt2,wdt;
   sw_parms_t swp;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time1.log","w",stdout);
      printf("\n");
      printf("Timing of the program sw_term()\n");
      printf("-------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

#if (defined AVX)
#if (defined FMA3)
      printf("Using AVX and FMA3 instructions\n");
#else
      printf("Using AVX instructions\n");
#endif
#elif (defined x64)
      printf("Using SSE3 instructions and up to 16 xmm registers\n\n");
#endif

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [time1.c]",
                    "Syntax: check1 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [time1.c]",
                    "Syntax: check1 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   set_lat_parms(5.5,1.0,0,NULL,is,1.978);
   print_lat_parms();

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.301,0.789,phi,phi_prime,theta);
   print_bc_parms(2);

   start_ranlux(0,12345);
   geometry();
   set_sw_parms(-0.0123);
   random_ud();

   swp=sw_parms();

   if (my_rank==0)
   {
      if (swp.isw)
      {
         printf("Exponential variant of the SW term\n");
         printf("m0=%.4e, csw=%.4e, N=%d\n\n",swp.m0,swp.csw,sw_order());
      }
      else
         printf("Traditional form of the SW term\n");
   }

   nt=(int)(5.0e5/(double)(VOLUME));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      for (count=0;count<nt;count++)
      {
         set_flags(UPDATED_UD);
         (void)sw_term(NO_PTS);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   wdt=2.0e6*wdt/((double)(nt)*(double)(VOLUME));

   if (my_rank==0)
   {
      printf("Time per lattice point: %4.3f micro sec",wdt);

      if (swp.isw)
      {
         printf(" (%d Mflops [%d bit arithmetic])\n",
                (int)((12552.0+2.0*((double)(sw_order())-5.0)*10.0)/wdt),
                (int)(sizeof(spinor_dble))/3);
      }
      else
      {
         printf(" (%d Mflops [%d bit arithmetic])\n",
                (int)(9936.0/wdt),(int)(sizeof(spinor_dble))/3);
      }
   }

   nt=(int)(2.0e6/(double)(VOLUME));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      for (count=0;count<nt;count++)
      {
         set_flags(ERASED_SWD);
         (void)sw_term(NO_PTS);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   wdt=2.0e6*wdt/((double)(nt)*(double)(VOLUME));

   if (my_rank==0)
   {
      printf("                        %4.3f micro sec",wdt);
      printf(" (field tensor is up-to-date)\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
