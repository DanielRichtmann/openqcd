
/*******************************************************************************
*
* File mdint.c
*
* Copyright (C) 2011-2013, 2017, 2018  Stefan Schaefer, Martin Luescher,
*                                      John Bulava
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Integration of the molecular-dynamics equations.
*
*   void run_mdint(void)
*     Integrates the molecular-dynamics equations.
*
* The integrator used is the one defined by the array of elementary operations
* returned by mdsteps() (see update/mdsteps.c). It is assumed that the fields,
* the integrator, the status counters and the chronological propagation of the
* the solutions of the Dirac equation have been properly initialized.
*
* In the course of the integration, the solver iteration numbers are added
* to the appropriate counters provided by the module update/counters.c. The
* deflation subspace is updated according to the parameter data base (see
* flags/dfl_parms.c).
*
* This program does not change the phase of the link variables. If phase-
* periodic boundary conditions are chosen, it is up to the calling program
* to ensure that the gauge field is in the proper phase-set condition (see
* uflds/uflds.c).
*
* The program in this module performs global communications and must be
* called simultaneously on all MPI processes.
*
* Some debugging information is printed to stdout if the macro MDINT_DBG is
* defined. The norm of the forces printed is the uniform norm.
*
*******************************************************************************/

#define MDINT_C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "mdflds.h"
#include "linalg.h"
#include "forces.h"
#include "update.h"
#include "global.h"


static void chk_mode_regen(int *status)
{
   int i,is;

   is=status[2];

   for (i=2;i<4;i++)
      status[i]=status[i+1];

   status[4]=is;

   if (status[4]>0)
      add2counter("modes",2,status+4);
   if (status[5]>0)
      add2counter("modes",2,status+5);
}

#ifdef MDINT_DBG

static void print_force_step(mdstep_t *s,double wdt)
{
   int my_rank;
   double nrm,eps;
   force_parms_t fp;
   mdflds_t *mdfs;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   mdfs=mdflds();
   nrm=unorm_alg(4*VOLUME,1,(*mdfs).frc);
   fp=force_parms((*s).iop);
   eps=(*s).eps;

   if (my_rank==0)
   {
      if (fp.force==FRG)
         printf("Force FRG:              ");
      else if (fp.force==FRF_TM1)
         printf("Force FRF_TM1:          ");
      else if (fp.force==FRF_TM1_EO)
         printf("Force FRF_TM1_EO:       ");
      else if (fp.force==FRF_TM1_EO_SDET)
         printf("Force FRF_TM1_EO_SDET:  ");
      else if (fp.force==FRF_TM2)
         printf("Force FRF_TM2:          ");
      else if (fp.force==FRF_TM2_EO)
         printf("Force FRF_TM2_EO:       ");
      else if (fp.force==FRF_RAT)
         printf("Force FRF_RAT:          ");
      else if (fp.force==FRF_RAT_SDET)
         printf("Force FRF_RAT_SDET:     ");

      printf("|frc| = %.2e, eps = % .2e, |eps*frc| = %.2e, "
             "time = %.2e sec\n",nrm/fabs(eps),eps,nrm,wdt);
   }
}

#endif

static void mdint(double *mu)
{
   int nop,itu;
   int iop,status[6];
   double eps;
#ifdef MDINT_DBG
   double wt1,wt2;
#endif
   mdstep_t *s,*sm;
   force_parms_t fp;
   solver_parms_t sp;

   s=mdsteps(&nop,&itu);
   sm=s+nop;

   for (;s<sm;s++)
   {
      iop=(*s).iop;
      eps=(*s).eps;

      if (iop<itu)
      {
         fp=force_parms(iop);

#ifdef MDINT_DBG
         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();
#endif
         if (fp.force==FRG)
            force0(eps);
         else
         {
            sp=solver_parms(fp.isp[0]);
            if (sp.solver==DFL_SAP_GCR)
               dfl_upd();
            set_sw_parms(sea_quark_mass(fp.im0));
            status[2]=0;
            status[5]=0;

            if (fp.force==FRF_TM1)
               force1(mu[fp.imu[0]],fp.ipf,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM1_EO)
               force4(mu[fp.imu[0]],fp.ipf,0,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM1_EO_SDET)
               force4(mu[fp.imu[0]],fp.ipf,1,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM2)
               force2(mu[fp.imu[0]],mu[fp.imu[1]],fp.ipf,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM2_EO)
               force5(mu[fp.imu[0]],mu[fp.imu[1]],fp.ipf,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_RAT)
               force3(fp.irat,fp.ipf,0,fp.isp[0],
                      eps,status);
            else if (fp.force==FRF_RAT_SDET)
               force3(fp.irat,fp.ipf,1,fp.isp[0],
                      eps,status);
            else
               error_root(1,1,"mdint [mdint.c]","Unknown force");

            if (sp.solver==DFL_SAP_GCR)
               chk_mode_regen(status);
            add2counter("force",iop,status);
         }

#ifdef MDINT_DBG
         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         print_force_step(s,wt2-wt1);
         update_mom();
         set_frc2zero();
#endif
      }
      else if (iop==itu)
      {
         update_mom();
         update_ud(eps);
      }
      else
         update_mom();
   }
}


void run_mdint(void)
{
   hmc_parms_t hmc;
   smd_parms_t smd;

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv!=0)
      mdint(hmc.mu);
   else if (smd.nlv!=0)
      mdint(smd.mu);
   else
      error_root(1,1,"run_mdint [mdint.c]",
                 "Simulation parameters are not set");
}
