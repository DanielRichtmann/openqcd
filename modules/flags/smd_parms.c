
/*******************************************************************************
*
* File smd_parms.c
*
* Copyright (C) 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic SMD parameters.
*
*   smd_parms_t set_smd_parms(int nact,int *iact,int npf,int nmu,double *mu,
*                             int nlv,double gamma,double eps,int iacc)
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
*       gamma       SMD friction parameter (gamma must be positive).
*
*       eps         Simulation step size (eps must be positive).
*
*       iacc        Rejection-acceptance step on|off (iacc=1 or iacc=0).
*
*     The program returns a structure that contains the parameters listed
*     above.
*
*   smd_parms_t smd_parms(void)
*     Returns a structure containing the current values of the parameters
*     listed above.
*
*   void print_smd_parms(void)
*     Prints the SMD parameters to stdout on MPI process 0.
*
*   void write_smd_parms(FILE *fdat)
*     Writes the SMD parameters to the file fdat on MPI process 0.
*
*   void check_smd_parms(FILE *fdat)
*     Compares the SMD parameters with the values stored on the file fdat
*     on MPI process 0, assuming the latter were written to the file by
*     the program write_smd_parms().
*
* The type smd_parms_t is defined in the file flags.h.
*
* To ensure the consistency of the data base, the SMD parameters must be set
* simultaneously on all processes. They may only be set once and may not be
* set at all if the HMC parameters are already set (see flags/hmc_parms.c).
*
*******************************************************************************/

#define SMD_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static smd_parms_t smd={0,0,0,0,0,NULL,0.0,0.0,NULL};


smd_parms_t set_smd_parms(int nact,int *iact,int npf,int nmu,double *mu,
                          int nlv,double gamma,double eps,int iacc)
{
   int iprms[5],i,ie;
   double dprms[2];
   hmc_parms_t hmc;

   hmc=hmc_parms();

   error_root(hmc.nlv!=0,1,"set_smd_parms [smd_parms.c]",
              "Cannot simultaneously set HMC and SMD parameters");

   error_root(smd.nlv!=0,1,"set_smd_parms [smd_parms.c]",
              "Parameters may only be set once");

   if (NPROC>1)
   {
      iprms[0]=nact;
      iprms[1]=npf;
      iprms[2]=nmu;
      iprms[3]=nlv;
      iprms[4]=iacc;
      dprms[0]=gamma;
      dprms[1]=eps;

      MPI_Bcast(iprms,5,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,2,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=nact)||(iprms[1]!=npf)||(iprms[2]!=nmu)||
            (iprms[3]!=nlv)||(iprms[4]!=iacc)||(dprms[0]!=gamma)||
            (dprms[1]!=eps),1,"set_smd_parms [smd_parms.c]",
            "Parameters are not global");

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

      error(ie!=0,2,"set_smd_parms [smd_parms.c]",
            "Parameters are not global");
   }

   error_root((nact<0)||(npf<0)||(nmu<0)||(nlv<1)||(iacc<0)||(iacc>1),1,
              "set_smd_parms [smd_parms.c]",
              "Parameter nact,npf,nmu,nlv or iacc is out of range");

   error_root((gamma<=0.0)||(eps<=0.0),1,"set_smd_parms [smd_parms.c]",
              "Parameters gamma and eps must be positive");

   if (nact>0)
   {
      smd.iact=malloc(nact*sizeof(int));
      error(smd.iact==NULL,1,"set_smd_parms [smd_parms.c]",
            "Unable to allocate parameter array");
   }

   if (nmu>0)
   {
      smd.mu=malloc(nmu*sizeof(double));
      error(smd.mu==NULL,2,"set_smd_parms [smd_parms.c]",
            "Unable to allocate parameter array");
   }

   smd.nact=nact;
   smd.npf=npf;
   smd.nmu=nmu;
   smd.nlv=nlv;
   smd.iacc=iacc;
   smd.gamma=gamma;
   smd.eps=eps;

   for (i=0;i<nact;i++)
      smd.iact[i]=iact[i];

   for (i=0;i<nmu;i++)
      smd.mu[i]=mu[i];

   return smd;
}


smd_parms_t smd_parms(void)
{
   return smd;
}


void print_smd_parms(void)
{
   int my_rank,n,i;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("SMD parameters:\n");
      printf("actions =");
      if (smd.nact>0)
      {
         for (i=0;i<smd.nact;i++)
            printf(" %d",smd.iact[i]);
         printf("\n");
      }
      else
         printf(" none\n");
      printf("npf = %d\n",smd.npf);
      printf("mu =");
      if (smd.nmu>0)
      {
         for (i=0;i<smd.nmu;i++)
         {
            n=fdigits(smd.mu[i]);
            printf(" %.*f",IMAX(n,1),smd.mu[i]);
         }
         printf("\n");
      }
      else
         printf(" none\n");
      printf("nlv = %d\n",smd.nlv);
      n=fdigits(smd.gamma);
      printf("gamma = %.*f\n",IMAX(n,1),smd.gamma);
      n=fdigits(smd.eps);
      printf("eps = %.*f\n",IMAX(n,1),smd.eps);
      printf("iacc = %d\n\n",smd.iacc);
   }
}


void write_smd_parms(FILE *fdat)
{
   int my_rank,endian;
   int nact,nmu,i,iw;
   stdint_t *istd;
   double *dstd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      nact=smd.nact;
      nmu=smd.nmu;

      istd=malloc((nact+5)*sizeof(*istd));
      dstd=malloc((nmu+2)*sizeof(*dstd));
      error_root((istd==NULL)||(dstd==NULL),1,"write_smd_parms [smd_parms.c]",
                 "Unable to allocate auxiliary arrays");

      istd[0]=(stdint_t)(smd.nact);
      istd[1]=(stdint_t)(smd.npf);
      istd[2]=(stdint_t)(smd.nmu);
      istd[3]=(stdint_t)(smd.nlv);
      istd[4]=(stdint_t)(smd.iacc);
      dstd[0]=smd.gamma;
      dstd[1]=smd.eps;

      for (i=0;i<nact;i++)
         istd[5+i]=(stdint_t)(smd.iact[i]);

      for (i=0;i<nmu;i++)
         dstd[2+i]=smd.mu[i];

      if (endian==BIG_ENDIAN)
      {
         bswap_int(nact+5,istd);
         bswap_double(nmu+2,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),nact+5,fdat);
      iw+=fwrite(dstd,sizeof(double),nmu+2,fdat);
      error_root(iw!=(nact+nmu+7),1,"write_smd_parms [smd_parms.c]",
                 "Incorrect write count");

      free(istd);
      free(dstd);
   }
}


void check_smd_parms(FILE *fdat)
{
   int my_rank,endian;
   int nact,nmu,i,ie,ir;
   stdint_t *istd;
   double *dstd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      nact=smd.nact;
      nmu=smd.nmu;

      istd=malloc((nact+5)*sizeof(*istd));
      dstd=malloc((nmu+2)*sizeof(*dstd));
      error_root((istd==NULL)||(dstd==NULL),1,"check_smd_parms [smd_parms.c]",
                 "Unable to allocate auxiliary arrays");

      ir=fread(istd,sizeof(stdint_t),nact+5,fdat);
      ir+=fread(dstd,sizeof(double),nmu+2,fdat);
      error_root(ir!=(nact+nmu+7),1,"check_smd_parms [smd_parms.c]",
                 "Incorrect read count");

      if (endian==BIG_ENDIAN)
      {
         bswap_int(nact+5,istd);
         bswap_double(nmu+2,dstd);
      }

      ie=0;
      ie|=(istd[0]!=(stdint_t)(smd.nact));
      ie|=(istd[1]!=(stdint_t)(smd.npf));
      ie|=(istd[2]!=(stdint_t)(smd.nmu));
      ie|=(istd[3]!=(stdint_t)(smd.nlv));
      ie|=(istd[4]!=(stdint_t)(smd.iacc));
      ie|=(dstd[0]!=smd.gamma);
      ie|=(dstd[1]!=smd.eps);

      for (i=0;i<nact;i++)
         ie|=(istd[5+i]!=(stdint_t)(smd.iact[i]));

      for (i=0;i<nmu;i++)
         ie|=(dstd[2+i]!=smd.mu[i]);

      error_root(ie!=0,1,"check_smd_parms [smd_parms.c]",
                 "Parameters do not match");

      free(istd);
      free(dstd);
   }
}
