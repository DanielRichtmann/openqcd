
/*******************************************************************************
*
* File chrono.c
*
* Copyright (C) 2007, 2011, 2012, 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs needed for the propagation of solutions of the Dirac equation
* along the molecular-dynamics trajectories.
*
*   void setup_chrono(void)
*     Allocates the required memory space for the stacks of previous
*     solutions to be used in the course of the molecular-dynamics
*     trajectories. The number and size of the stacks is inferred from
*     the parameter data base.
*
*   double mdtime(void)
*     Returns the current molecular-dynamics time.
*
*   void step_mdtime(double dt)
*     Advances the molecular-dynamics time by dt.
*
*   void add_chrono(int icr,spinor_dble *psi)
*     Adds the solution psi obtained at the current molecular-dynamics
*     time to the stack number icr of previously calculated solutions.
*     If the MD time has not changed since the last addition, the last
*     saved field is replaced by psi.
*
*   int get_chrono(int icr,spinor_dble *psi)
*     Extrapolates the solutions stored in the stack number icr to the
*     current molecular-dynamics time. The program returns 0 and leaves
*     psi unchanged if the stack does not contain any previous solutions.
*     Otherwise the program assigns the extrapolated solution to psi and
*     returns 1.
*
*   void reset_chrono(int n)
*     Deletes the stored solutions in all stacks except for the last n
*     solutions. All previous solutions are kept in the stacks that
*     currently contain less than n solutions. If n=0 the stacks are
*     all reset to their initial state.
*
*   size_t chrono_msize(void)
*     Returns the total local size in bytes of the stacks of solutions.
*
* The propagation of the solutions of the Dirac equation was proposed by
*
*   R.C. Brower et al., "Chronological inversion method for the Dirac
*   matrix in Hybrid Monte Carlo", Nucl. Phys. B484 (1997) 353
*
* Here the solutions are propagated using a polynomial extrapolation. The
* maximal number of solutions to be kept in memory can be chosen for each
* solution stack separately. How many stacks should be allocated, and how
* many fields are kept in each case, is inferred from the force parameters
* in the parameter data base (see flags/force_parms.c).
*
* The stacks are labeled by an index icr=0,1,2,.., where the empty stack
* has index icr=0 and all other stacks have index icr>0. The indices are
* included in the force parameter sets and are set internally when the
* force parameters are entered in the data base.
*
* Depending on the type of force, only the part of the fields on the even
* sites of the lattice are stored and propagated in time.
*
* This module includes a clock that serves to keep track of the molecular-
* dynamics times at which the Dirac equation is solved. The clock is never
* reset. It starts at time 0 and is advanced by the molecular-dynamics
* integrator (see update/update.c).
*
* All programs in this module should be called simultaneously on all MPI
* processes.
*
*******************************************************************************/

#define CHRONO_C

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "update.h"
#include "global.h"

typedef struct
{
   int ncr,vol,isd,nsd;
   double *ta;
   spinor_dble **sd;
} stack_t;

static int nst=0;
static double mdt=0.0;
static stack_t *st=NULL;


static int check_levels(void)
{
   int nlv,ilv,ie;
   hmc_parms_t hmc;
   smd_parms_t smd;
   mdint_parms_t mdp;

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv>0)
      nlv=hmc.nlv;
   else if (smd.nlv>0)
      nlv=smd.nlv;
   else
      nlv=0;

   ie=0;

   for (ilv=0;ilv<nlv;ilv++)
   {
      mdp=mdint_parms(ilv);
      ie|=(mdp.nstep==0);
   }

   error((nlv==0)||(ie!=0),1,"check_levels [chrono.c]",
         "Simulation or the MD integrator parameters are not set");

   return nlv;
}


static void free_stacks(void)
{
   int icr;

   for (icr=1;icr<nst;icr++)
   {
      free(st[icr].ta);
      afree(st[icr].sd[0]);
      free(st[icr].sd);
   }

   if (nst>0)
      free(st);

   nst=0;
   st=NULL;
}


static void alloc_stacks(int nlv)
{
   int ilv,icr,j,k,ie;
   mdint_parms_t mdp;
   force_parms_t fp;

   nst=0;

   for (ilv=0;ilv<nlv;ilv++)
   {
      mdp=mdint_parms(ilv);

      for (j=0;j<mdp.nfr;j++)
      {
         fp=force_parms(mdp.ifr[j]);

         for (k=0;k<4;k++)
         {
            if (fp.icr[k]>nst)
               nst=fp.icr[k];
         }
      }
   }

   nst+=1;
   st=malloc(nst*sizeof(*st));
   error(st==NULL,1,"alloc_stacks [chrono.c]",
         "Unable to allocate stack structures");

   for (j=0;j<nst;j++)
   {
      st[j].ncr=0;
      st[j].vol=0;
      st[j].isd=0;
      st[j].nsd=0;
      st[j].ta=NULL;
      st[j].sd=NULL;
   }

   ie=0;

   for (ilv=0;ilv<nlv;ilv++)
   {
      mdp=mdint_parms(ilv);

      for (j=0;j<mdp.nfr;j++)
      {
         fp=force_parms(mdp.ifr[j]);

         for (k=0;k<4;k++)
         {
            if (fp.icr[k]>0)
            {
               ie|=((fp.force==FRG)||(fp.force==FORCES));
               ie|=((fp.ncr[k]==0)||(st[fp.icr[k]].ncr!=0));
               st[fp.icr[k]].ncr=fp.ncr[k];

               if ((fp.force==FRF_TM1)||(fp.force==FRF_TM2))
                  st[fp.icr[k]].vol=VOLUME;
               else
                  st[fp.icr[k]].vol=VOLUME/2;
            }
            else
               ie|=(fp.ncr[k]!=0);
         }
      }
   }

   for (icr=1;icr<nst;icr++)
      ie|=(st[icr].ncr==0);

   error(ie!=0,1,"alloc_stacks [chrono.c]",
         "Unexpected entries in the force parameter data base");
}


void setup_chrono(void)
{
   int nlv,ncr,vol,icr,k;
   double *ta;
   spinor_dble **sd,*s;

   nlv=check_levels();
   free_stacks();
   alloc_stacks(nlv);

   for (icr=1;icr<nst;icr++)
   {
      ncr=st[icr].ncr;
      vol=st[icr].vol;

      ta=malloc(ncr*sizeof(*ta));
      sd=malloc(ncr*sizeof(*sd));
      s=amalloc(ncr*vol*sizeof(*s),ALIGN);

      if ((ta==NULL)||(sd==NULL)||(s==NULL))
         break;

      st[icr].ta=ta;
      st[icr].sd=sd;

      for (k=0;k<ncr;k++)
      {
         ta[k]=0.0;
         sd[k]=s;
         set_sd2zero(vol,s);
         s+=vol;
      }
   }

   error(icr<nst,1,"setup_chrono [chrono.c]",
         "Unable to allocate field stacks");
}


double mdtime(void)
{
   return mdt;
}


void step_mdtime(double dt)
{
   mdt+=dt;
}


void add_chrono(int icr,spinor_dble *psi)
{
   int ncr,vol,isd,nsd;
   int ird,iprms[1];

   error_root((icr<0)||(icr>=nst),1,"add_chrono [chrono.c]",
              "Unknown field stack");

   if (NPROC>1)
   {
      iprms[0]=icr;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      error(iprms[0]!=icr,1,"add_chrono [chrono.c]",
            "Parameter icr is not global");
   }

   if ((icr>0)&&(icr<nst))
   {
      ncr=st[icr].ncr;
      vol=st[icr].vol;
      isd=st[icr].isd;
      nsd=st[icr].nsd;

      if (nsd>0)
      {
         ird=isd-1;
         if (ird<0)
            ird=ncr-1;

         if (mdt==st[icr].ta[ird])
         {
            assign_sd2sd(vol,psi,st[icr].sd[ird]);
            return;
         }
      }

      st[icr].ta[isd]=mdt;
      assign_sd2sd(vol,psi,st[icr].sd[isd]);

      if (isd<(ncr-1))
         st[icr].isd+=1;
      else
         st[icr].isd=0;

      if (nsd<ncr)
         st[icr].nsd+=1;
   }
}


int get_chrono(int icr,spinor_dble *psi)
{
   int ncr,vol,nsd,isd;
   int k,l,ksd,lsd,ird,iprms[1];
   double *ta,c;
   spinor_dble **sd;

   if (nst==0)
      return 0;

   error_root((icr<0)||(icr>=nst),1,"get_chrono [chrono.c]",
              "Unknown field stack");

   if (NPROC>1)
   {
      iprms[0]=icr;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      error(iprms[0]!=icr,1,"get_chrono [chrono.c]",
            "Parameter icr is not global");
   }

   if ((icr>0)&&(icr<nst))
   {
      nsd=st[icr].nsd;

      if (nsd>0)
      {
         ncr=st[icr].ncr;
         vol=st[icr].vol;
         isd=st[icr].isd;
         ta=st[icr].ta;
         sd=st[icr].sd;

         ird=isd-1;
         if (ird<0)
            ird=ncr-1;

         if ((nsd==1)||(ta[ird]==mdt))
            assign_sd2sd(vol,sd[ird],psi);
         else
         {
            set_sd2zero(vol,psi);
            ksd=ird;

            for (k=0;k<nsd;k++)
            {
               c=1.0;
               lsd=ird;

               for (l=0;l<nsd;l++)
               {
                  if (l!=k)
                     c*=((mdt-ta[lsd])/(ta[ksd]-ta[lsd]));

                  lsd-=1;
                  if (lsd<0)
                     lsd=ncr-1;
               }

               mulr_spinor_add_dble(vol,psi,sd[ksd],c);

               ksd-=1;
               if (ksd<0)
                  ksd=ncr-1;
            }
         }

         return 1;
      }
   }

   return 0;
}


void reset_chrono(int n)
{
   int icr,nsd,iprms[1];

   error_root(n<0,1,"reset_chrono [chrono.c]",
              "Parameter n is out of range");

   if (NPROC>1)
   {
      iprms[0]=n;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      error(iprms[0]!=n,1,"reset_chrono [chrono.c]",
            "Parameter n is not global");
   }

   if (n>=0)
   {
      for (icr=1;icr<nst;icr++)
      {
         nsd=st[icr].nsd;

         if (nsd>n)
            st[icr].nsd=n;

         if (n==0)
            st[icr].isd=0;
      }
   }
}


size_t chrono_msize(void)
{
   int icr;
   size_t nall;

   nall=0;

   for (icr=1;icr<nst;icr++)
      nall+=(size_t)(st[icr].ncr)*(size_t)(st[icr].vol);

   return nall*sizeof(spinor_dble);
}
