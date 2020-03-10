
/*******************************************************************************
*
* File msize.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Size of the memory occupied by the simulation programs.
*
*   size_t msize(void)
*     Returns an estimate of the size (in bytes per MPI process) of the
*     memory used by the current HMC or SMD simulation. The program assumes
*     that all parameters are set, the workspaces are allocated and the
*     chronological solver is set up.
*
*   void print_msize(int ipr)
*     Calls msize() and prints the estimated size of the memory used to
*     stdout on MPI process 0 in human-readable form. A more detailed
*     report on the memory occupation is printed if ipr=1 and an even
*     more detailed one if ipr=2.
*
* The value returned by msize() includes the memory used for the fields,
* the large index arrays and the various buffers created by openQCD. Most
* of the additional memory required for online measurements is included
* in the workspace and thus taken into account. In practice, the memory
* actually used is somewhat larger than calculated, because of the OS and
* MPI overheads (shared libraries, system communication buffers, etc.).
*
* The units used by print_msize() are
*
*  1 KiB = 1024 bytes, 1 MiB = (1024)^2 bytes, 1 GiB = (1024)^3 bytes, ...
*
* The programs in this module may perform global communications and must
* be called simultaneously on all MPI processes.
*
*******************************************************************************/

#define MSIZE_C

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "mdflds.h"
#include "update.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static const int fsz[4]={FACE0,FACE1,FACE2,FACE3};
static const int lsz[4]={L0,L1,L2,L3};
static const int nsz[4]={N0,N1,N2,N3};

static size_t nbuf,nchr,ndfl,nglb,nmdf,nsap,nwsp;
static char *unit[5]={"KiB","MiB","GiB","TiB","PiB"};


static void blk_numbers(int *bs,int *nb,int *nbb)
{
   int v,vb;

   v=bs[0]*bs[1]*bs[2]*bs[3];
   vb=FACE0*bs[0]+FACE1*bs[1]+FACE2*bs[2]+FACE3*bs[3];

   (*nb)=VOLUME/v;
   (*nbb)=2*(vb/v);
}


static size_t mdflds_msize(int *iswd)
{
   int npf,ipf,*eo;
   size_t nall,vol,bndry;
   spinor_dble **pf;
   mdflds_t *mdfs;

   mdfs=mdflds();
   npf=(*mdfs).npf;
   eo=(*mdfs).eo;
   pf=(*mdfs).pf;

   vol=(size_t)(VOLUME);
   bndry=(size_t)(BNDRY);

   (*iswd)=0;
   nall=(8*vol+7*(bndry/4))*sizeof(su3_alg_dble);

   for (ipf=0;ipf<npf;ipf++)
   {
      if (pf[ipf]!=NULL)
      {
         (*iswd)=1;

         if (eo[ipf])
            nall+=(vol/2)*sizeof(spinor_dble);
         else
            nall+=vol*sizeof(spinor_dble);
      }
   }

   return nall;
}


static void check_solver(solver_t solver,int *isap,int *idfl,int *itmcg)
{
   (*isap)|=((solver==SAP_GCR)||(solver==DFL_SAP_GCR));
   (*idfl)|=(solver==DFL_SAP_GCR);
   (*itmcg)|=(solver==CGNE);
}


static void solver_types(int nlv,int nact,int *iact,
                         int *isap,int *idfl,int *itmcg)
{
   int i,j,nfr,*ifr;
   mdint_parms_t mdp;
   action_parms_t ap;
   solver_parms_t sp;
   force_parms_t fp;

   (*isap)=0;
   (*idfl)=0;
   (*itmcg)=0;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action!=ACG)
      {
         sp=solver_parms(ap.isp[0]);
         check_solver(sp.solver,isap,idfl,itmcg);
      }

      if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO))
      {
         sp=solver_parms(ap.isp[1]);
         check_solver(sp.solver,isap,idfl,itmcg);
      }
   }

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);
      nfr=mdp.nfr;
      ifr=mdp.ifr;

      for (j=0;j<nfr;j++)
      {
         fp=force_parms(ifr[j]);

         if (fp.force!=FRG)
         {
            sp=solver_parms(fp.isp[0]);
            check_solver(sp.solver,isap,idfl,itmcg);
         }
      }
   }
}


static size_t global_msize(int iswd,int isw)
{
   size_t nall,vol,bndry,edge;

   vol=(size_t)(VOLUME);
   bndry=(size_t)(BNDRY);
   edge=(size_t)(L0*L1+L0*L2+L0*L3+L1*L2+L1*L3+L2*L3);

   nall=(10*vol+bndry)*sizeof(int);
   nall+=(4*vol+7*(bndry/4))*sizeof(su3_dble);

   if ((iswd)||(isw))
   {
      nall+=4*vol*sizeof(su3_dble);
      nall+=2*(6*vol+3*(bndry/2)+edge)*sizeof(u3_alg_dble);
      nall+=2*vol*sizeof(pauli_dble);
   }

   if (isw)
   {
      nall+=4*vol*sizeof(su3);
      nall+=2*vol*sizeof(pauli);
   }

   return nall;
}


static size_t sap_msize(void)
{
   int nb,vol,bndry,*bs;
   size_t nall;
   sap_parms_t sap;

   sap=sap_parms();
   bs=sap.bs;

   vol=bs[0]*bs[1]*bs[2]*bs[3];
   bndry=2*(bs[1]*bs[2]*bs[3]+bs[2]*bs[3]*bs[0]+
            bs[3]*bs[0]*bs[1]+bs[0]*bs[1]*bs[2]);
   nb=VOLUME/vol;

   nall=(9*vol+2*bndry)*sizeof(int);
   nall+=nb*(vol+bndry)*sizeof(int);
   nall+=nb*(4*vol+bndry)*sizeof(su3);
   nall+=nb*2*vol*sizeof(pauli);
   nall+=3*vol*sizeof(spinor);

   nall+=nb*bndry*sizeof(int);
   nall+=(nb*bndry+BNDRY)*sizeof(weyl);

   return nall;
}


static size_t dfl_msize(void)
{
   int Ns,nb,nbb,vol,bndry,*bs;
   size_t nall;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   bs=dfl.bs;
   blk_numbers(bs,&nb,&nbb);

   vol=bs[0]*bs[1]*bs[2]*bs[3];
   bndry=2*(bs[1]*bs[2]*bs[3]+bs[2]*bs[3]*bs[0]+
            bs[3]*bs[0]*bs[1]+bs[0]*bs[1]*bs[2]);

   nall=(9*vol+2*bndry)*sizeof(int);
   nall+=nb*(vol+bndry)*sizeof(int);
   nall+=4*vol*sizeof(su3_dble);
   nall+=2*vol*sizeof(pauli_dble);
   nall+=(Ns+1)*nb*vol*sizeof(spinor);
   nall+=2*vol*sizeof(spinor_dble);

   nall+=18*nb*Ns*Ns*sizeof(complex);
   nall+=(18*nb+3*(nbb/2))*Ns*Ns*sizeof(complex_dble);

   nall+=2*Ns*Ns*nb*sizeof(complex);
   nall+=Ns*Ns*nb*sizeof(complex_dble);

   return nall;
}


static size_t buf_msize(int iswd,int isw,int idfl)
{
   int mu,fmx,lmx,nmx;
   int Ns,*bs;
   int nb,nbb,vbb,vbm,vb[4];
   size_t nall;
   dfl_parms_t dfl;
   lat_parms_t lat;
   smd_parms_t smd;

   dfl=dfl_parms();
   lat=lat_parms();
   smd=smd_parms();

   fmx=0;
   lmx=0;
   nmx=0;

   for (mu=0;mu<4;mu++)
   {
      if (fsz[mu]>fmx)
         fmx=fsz[mu];

      if (lsz[mu]>lmx)
         lmx=lsz[mu];

      if (nsz[mu]>nmx)
         nmx=nsz[mu];
   }

   nall=4*(lmx+nmx)*sizeof(su3_dble);

   if (smd.nlv)
      nall+=4*(L3+N3)*sizeof(su3_alg_dble);

   nall+=7*(BNDRY/4)*sizeof(int);
   nall+=3*fmx*sizeof(su3_dble);

   if (lat.c0!=1.0)
      nall+=3*(BNDRY+fmx)*sizeof(su3_dble);

   nall+=7*(BNDRY/4)*sizeof(su3_alg_dble);

   if (iswd)
   {
      nall+=BNDRY*sizeof(weyl_dble);

      nall+=6*BNDRY*sizeof(int);
      nall+=fmx*sizeof(u3_alg_dble);

      if (smd.nlv)
         nall+=(L3+N3)*sizeof(spinor_dble);
   }

   if (isw)
      nall+=BNDRY*sizeof(weyl);

   if (idfl)
   {
      Ns=dfl.Ns;
      bs=dfl.bs;
      blk_numbers(bs,&nb,&nbb);

      vb[0]=bs[1]*bs[2]*bs[3];
      vb[1]=bs[2]*bs[3]*bs[0];
      vb[2]=bs[3]*bs[0]*bs[1];
      vb[3]=bs[0]*bs[1]*bs[2];
      vbb=0;
      vbm=0;

      for (mu=0;mu<4;mu++)
      {
         vbb+=vb[mu];

         if (vb[mu]>vbm)
            vbm=vb[mu];
      }

      nall+=nb*vbb*sizeof(int);
      nall+=3*(nbb/2)*Ns*Ns*sizeof(complex_dble);
      nall+=3*Ns*vbm*sizeof(spinor_dble);

      nall+=Ns*nbb*sizeof(complex_dble);
      nall+=Ns*nbb*sizeof(complex);
   }

   return nall;
}


size_t msize(void)
{
   int nlv,nact,*iact;
   int isap,idfl,itmcg;
   int iswd,isw;
   size_t nall;
   hmc_parms_t hmc;
   smd_parms_t smd;

   nbuf=0;
   nchr=0;
   ndfl=0;
   nglb=0;
   nmdf=0;
   nsap=0;
   nwsp=0;
   nall=0;

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv)
   {
      nact=hmc.nact;
      nlv=hmc.nlv;
      iact=hmc.iact;
   }
   else if (smd.nlv)
   {
      nact=smd.nact;
      nlv=smd.nlv;
      iact=smd.iact;
   }
   else
   {
      nact=0;
      nlv=0;
      iact=NULL;
   }

   error(nlv==0,1,"msize [msize.c]",
         "The parameters of the simulation algorithm are not set");

   nmdf=mdflds_msize(&iswd);
   nall=nmdf;

   solver_types(nlv,nact,iact,&isap,&idfl,&itmcg);
   isw=((idfl)||(itmcg));

   nglb=global_msize(iswd,isw);
   nall+=nglb;

   if (isap)
   {
      nsap=sap_msize();
      nall+=nsap;
   }

   if (idfl)
   {
      ndfl=dfl_msize();
      nall+=ndfl;
   }

   nbuf=buf_msize(iswd,isw,idfl);
   nall+=nbuf;

   nchr=chrono_msize();
   nall+=nchr;

   nwsp=wsp_msize();
   nall+=nwsp;

   return nall;
}


static void scale_msize(double *r,int *iu)
{
   int k;

   (*r)/=1024.0;
   (*iu)=0;

   for (k=0;k<4;k++)
   {
      if ((*r)>=1024.0)
      {
         (*r)/=1024.0;
         (*iu)+=1;
      }
      else
         return;
   }
}


void print_msize(int ipr)
{
   int my_rank,iu;
   size_t nall;
   double np,rp,ra,rp0;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   nall=msize();

   if (my_rank==0)
   {
      np=(double)(NPROC);
      rp=(double)(nall);
      ra=rp*np;
      rp0=rp;

      scale_msize(&ra,&iu);
      printf("Total memory used = %.1f %s ",ra,unit[iu]);
      scale_msize(&rp,&iu);
      printf("(= %.1f %s per MPI process)\n",rp,unit[iu]);

      if ((ipr==1)||(ipr==2))
      {
         rp=100.0*(double)(nglb)/rp0;
         printf("Global fields            = %5.1f%%\n",rp);

         rp=100.0*(double)(nmdf)/rp0;
         printf("MD fields                = %5.1f%%\n",rp);

         if (nbuf)
         {
            rp=100.0*(double)(nbuf)/rp0;
            printf("Communication buffers    = %5.1f%%\n",rp);
         }

         if (nchr)
         {
            rp=100.0*(double)(nchr)/rp0;
            printf("Chronological solver     = %5.1f%%\n",rp);
         }

         if (nsap)
         {
            rp=100.0*(double)(nsap)/rp0;
            printf("SAP preconditioner       = %5.1f%%\n",rp);
         }

         if (ndfl)
         {
            rp=100.0*(double)(ndfl)/rp0;
            printf("Deflation preconditioner = %5.1f%%\n",rp);
         }

         if (nwsp)
         {
            rp=100.0*(double)(nwsp)/rp0;
            printf("Workspace                = %5.1f%%\n",rp);
         }
      }

      printf("\n");
   }

   if (ipr==2)
      print_wsp();
}
