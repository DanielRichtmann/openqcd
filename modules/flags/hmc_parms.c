
/*******************************************************************************
*
* File hmc_parms.c
*
* Copyright (C) 2009, 2010, 2011, 2013, 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic HMC parameters.
*
*   hmc_parms_t set_hmc_parms(int nact,int *iact,int npf,int nmu,double *mu,
*                             int nlv,double tau)
*     Sets some basic parameters of the HMC algorithm. The parameters are
*
*       nact        Number of terms in the total action.
*
*       iact        Indices iact[i] of the action terms (i=0,..,nact-1).
*
*       npf         Number of pseudo-fermion fields on which the action
*                   depends.
*
*       nmu         Number of twisted mass parameters on which the
*                   pseudo-fermion actions and forces depend.
*
*       mu          Twisted masses mu[i] (i=0,..,nmu-1).
*
*       nlv         Number of levels of the molecular-dynamics integrator
*                   (nlv must be at least 1).
*
*       tau         Molecular-dynamics trajectory length.
*
*     The program returns a structure that contains the parameters listed
*     above.
*
*   hmc_parms_t hmc_parms(void)
*     Returns a structure containing the current values of the parameters
*     listed above.
*
*   void print_hmc_parms(void)
*     Prints the HMC parameters to stdout on MPI process 0.
*
*   void write_hmc_parms(FILE *fdat)
*     Writes the HMC parameters to the file fdat on MPI process 0.
*
*   void check_hmc_parms(FILE *fdat)
*     Compares the HMC parameters with the values stored on the file fdat
*     on MPI process 0, assuming the latter were written to the file by
*     the program write_hmc_parms().
*
* The type hmc_parms_t is defined in the file flags.h.
*
* To ensure the consistency of the data base, the HMC parameters must be set
* simultaneously on all processes. They may only be set once and may not be
* set at all if the SMD parameters are already set (see flags/smd_parms.c).
*
*******************************************************************************/

#define HMC_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static hmc_parms_t hmc={0,0,0,0,NULL,0.0,NULL};


hmc_parms_t set_hmc_parms(int nact,int *iact,int npf,int nmu,double *mu,
                          int nlv,double tau)
{
   int iprms[4],i,ie;
   double dprms[1];
   smd_parms_t smd;

   smd=smd_parms();

   error_root(smd.nlv!=0,1,"set_hmc_parms [hmc_parms.c]",
              "Cannot simultaneously set HMC and SMD parameters");

   error_root(hmc.nlv!=0,1,"set_hmc_parms [hmc_parms.c]",
              "Parameters may only be set once");

   if (NPROC>1)
   {
      iprms[0]=nact;
      iprms[1]=npf;
      iprms[2]=nmu;
      iprms[3]=nlv;
      dprms[0]=tau;

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=nact)||(iprms[1]!=npf)||(iprms[2]!=nmu)||
            (iprms[3]!=nlv)||(dprms[0]!=tau),1,
            "set_hmc_parms [hmc_parms.c]","Parameters are not global");

      ie=0;

      for (i=0;i<nact;i++)
      {
         iprms[0]=iact[i];
         MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
         ie|=(iprms[0]!=iact[i]);
      }

      for (i=0;i<nmu;i++)
      {
         dprms[0]=mu[i];
         MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
         ie|=(dprms[0]!=mu[i]);
      }

      error(ie!=0,2,"set_hmc_parms [hmc_parms.c]",
            "Parameters are not global");
   }

   error_root((nact<0)||(npf<0)||(nmu<0)||(nlv<1),1,
              "set_hmc_parms [hmc_parms.c]",
              "Parameter nact,npf,nmu or nlv is out of range");

   if (nact>0)
   {
      hmc.iact=malloc(nact*sizeof(int));
      error(hmc.iact==NULL,1,"set_hmc_parms [hmc_parms.c]",
            "Unable to allocate parameter array");
   }

   if (nmu>0)
   {
      hmc.mu=malloc(nmu*sizeof(double));
      error(hmc.mu==NULL,2,"set_hmc_parms [hmc_parms.c]",
            "Unable to allocate parameter array");
   }

   hmc.nact=nact;
   hmc.npf=npf;
   hmc.nmu=nmu;
   hmc.nlv=nlv;
   hmc.tau=tau;

   for (i=0;i<nact;i++)
      hmc.iact[i]=iact[i];

   for (i=0;i<nmu;i++)
      hmc.mu[i]=mu[i];

   return hmc;
}


hmc_parms_t hmc_parms(void)
{
   return hmc;
}


void print_hmc_parms(void)
{
   int my_rank,n,i;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("HMC parameters:\n");
      printf("actions =");
      if (hmc.nact>0)
      {
         for (i=0;i<hmc.nact;i++)
            printf(" %d",hmc.iact[i]);
         printf("\n");
      }
      else
         printf(" none\n");
      printf("npf = %d\n",hmc.npf);
      printf("mu =");
      if (hmc.nmu>0)
      {
         for (i=0;i<hmc.nmu;i++)
         {
            n=fdigits(hmc.mu[i]);
            printf(" %.*f",IMAX(n,1),hmc.mu[i]);
         }
         printf("\n");
      }
      else
         printf(" none\n");
      printf("nlv = %d\n",hmc.nlv);
      n=fdigits(hmc.tau);
      printf("tau = %.*f\n\n",IMAX(n,1),hmc.tau);
   }
}


void write_hmc_parms(FILE *fdat)
{
   int my_rank,endian;
   int nact,nmu,i,iw;
   stdint_t *istd;
   double *dstd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      nact=hmc.nact;
      nmu=hmc.nmu;

      istd=malloc((nact+4)*sizeof(*istd));
      dstd=malloc((nmu+1)*sizeof(*dstd));
      error_root((istd==NULL)||(dstd==NULL),1,"write_hmc_parms [hmc_parms.c]",
                 "Unable to allocate auxiliary arrays");

      istd[0]=(stdint_t)(hmc.nact);
      istd[1]=(stdint_t)(hmc.npf);
      istd[2]=(stdint_t)(hmc.nmu);
      istd[3]=(stdint_t)(hmc.nlv);
      dstd[0]=hmc.tau;

      for (i=0;i<nact;i++)
         istd[4+i]=(stdint_t)(hmc.iact[i]);

      for (i=0;i<nmu;i++)
         dstd[1+i]=hmc.mu[i];

      if (endian==BIG_ENDIAN)
      {
         bswap_int(nact+4,istd);
         bswap_double(nmu+1,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),nact+4,fdat);
      iw+=fwrite(dstd,sizeof(double),nmu+1,fdat);
      error_root(iw!=(nact+nmu+5),1,"write_hmc_parms [hmc_parms.c]",
                 "Incorrect write count");

      free(istd);
      free(dstd);
   }
}


void check_hmc_parms(FILE *fdat)
{
   int my_rank,endian;
   int nact,nmu,i,ie,ir;
   stdint_t *istd;
   double *dstd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      nact=hmc.nact;
      nmu=hmc.nmu;

      istd=malloc((nact+4)*sizeof(*istd));
      dstd=malloc((nmu+1)*sizeof(*dstd));
      error_root((istd==NULL)||(dstd==NULL),1,"check_hmc_parms [hmc_parms.c]",
                 "Unable to allocate auxiliary arrays");

      ir=fread(istd,sizeof(stdint_t),nact+4,fdat);
      ir+=fread(dstd,sizeof(double),nmu+1,fdat);
      error_root(ir!=(nact+nmu+5),1,"check_hmc_parms [hmc_parms.c]",
                 "Incorrect read count");

      if (endian==BIG_ENDIAN)
      {
         bswap_int(nact+4,istd);
         bswap_double(nmu+1,dstd);
      }

      ie=0;
      ie|=(istd[0]!=(stdint_t)(hmc.nact));
      ie|=(istd[1]!=(stdint_t)(hmc.npf));
      ie|=(istd[2]!=(stdint_t)(hmc.nmu));
      ie|=(istd[3]!=(stdint_t)(hmc.nlv));
      ie|=(dstd[0]!=hmc.tau);

      for (i=0;i<nact;i++)
         ie|=(istd[4+i]!=(stdint_t)(hmc.iact[i]));

      for (i=0;i<nmu;i++)
         ie|=(dstd[1+i]!=hmc.mu[i]);

      error_root(ie!=0,1,"check_hmc_parms [hmc_parms.c]",
                 "Parameters do not match");

      free(istd);
      free(dstd);
   }
}
