
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2007-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs in the module dfl_subspace.c.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "block.h"
#include "linalg.h"
#include "sflds.h"
#include "vflds.h"
#include "dfl.h"
#include "global.h"


static double check_basis(int Ns)
{
   int nb,isw,i,j;
   double dev,dmx;
   complex_dble z;
   complex_qflt zq;
   block_t *b,*bm;

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   bm=b+nb;

   dmx=0.0;

   for (;b<bm;b++)
   {
      for (i=1;i<=Ns;i++)
      {
         assign_s2sd((*b).vol,(*b).s[i],(*b).sd[0]);

         for (j=1;j<=i;j++)
         {
            assign_s2sd((*b).vol,(*b).s[j],(*b).sd[1]);
            zq=spinor_prod_dble((*b).vol,0,(*b).sd[0],(*b).sd[1]);
            z.re=zq.re.q[0];
            z.im=zq.im.q[0];
            dev=sqrt(z.re*z.re+z.im*z.im);

            if (i==j)
               dev=fabs(1.0-dev);

            if (dev>dmx)
               dmx=dev;
         }
      }
   }

   if (NPROC>1)
   {
      dev=dmx;
      MPI_Reduce(&dev,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return dmx;
}


int main(int argc,char *argv[])
{
   int my_rank,bc,i;
   int bs[4],Ns,nv;
   double phi[2],phi_prime[2],theta[3];
   double dev,dmx[3];
   complex **vm,**wv,z;
   spinor **ws;
   FILE *fin=NULL,*flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check2.in","r",stdin);

      printf("\n");
      printf("Check of the programs in the module dfl_subspace.c\n");
      printf("--------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check2.c]",
                    "Syntax: check2 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.0;
   phi[1]=0.0;
   phi_prime[0]=0.0;
   phi_prime[1]=0.0;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0);

   start_ranlux(0,123456);
   geometry();
   set_dfl_parms(bs,Ns);

   alloc_ws(Ns+1);
   alloc_wv(2);

   ws=reserve_ws(Ns+1);
   vm=vflds()+Ns;
   wv=reserve_wv(2);
   nv=Ns*VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }

   dfl_subspace(ws);
   dmx[0]=check_basis(Ns);
   dmx[1]=0.0;
   dmx[2]=0.0;

   for (i=0;i<Ns;i++)
   {
      dfl_v2s(vm[i],ws[Ns]);
      mulr_spinor_add(VOLUME,ws[Ns],ws[i],-1.0f);
      dev=(double)(norm_square(VOLUME,1,ws[Ns])/
                   norm_square(VOLUME,1,ws[i]));
      if (dev>dmx[1])
         dmx[1]=dev;
   }

   for (i=0;i<10;i++)
   {
      random_v(nv,wv[0],1.0f);
      dfl_v2s(wv[0],ws[Ns]);
      dfl_s2v(ws[Ns],wv[1]);
      z.re=-1.0f;
      z.im=0.0f;
      mulc_vadd(nv,wv[0],wv[1],z);
      dev=(double)(vnorm_square(nv,1,wv[0])/vnorm_square(nv,1,wv[1]));

      if (dev>dmx[2])
         dmx[2]=dev;
   }

   dmx[2]=sqrt(dmx[2]);

   if (my_rank==0)
   {
      printf("Orthonormality of the basis vectors:        %.1e\n",dmx[0]);
      printf("Check of the single-precision vector modes: %.1e\n",dmx[1]);
      printf("Check of dfl_s2v() and dfl_v2s():           %.1e\n\n",dmx[2]);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
