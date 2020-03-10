
/*******************************************************************************
*
* File smd.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* SMD simulation algorithm.
*
*   void smd_reset_dfl(void)
*     Regenerates the deflation subspace if deflation is used.
*
*   void smd_init(void)
*     Initializes the momentum and pseudo-fermion fields to random values
*     with the correct distributions.
*
*   void smd_action(qflt *act)
*     Computes the local parts of the momentum, gauge and pseudo-fermion
*     actions (see the notes).
*
*   int run_smd(qflt *act0,qflt *act1)
*     Applies a complete SMD update cycle to the current fields and assigns
*     the local parts of the momentum, gauge and pseudo-fermion actions at
*     the beginning and the end of the molecular-dynamics evolution to act0
*     and act1 (see the notes). The program returns 1 if the generated new
*     configuration was accepted and 0 otherwise. The gauge field is phase-
*     unset on exit.
*
*   void run_smd_noacc0(qflt *act0,qflt *act1)
*     Same as run_smd() w/o accept-reject step.
*
*   void run_smd_noacc1(void)
*     Same as run_smd() w/o accept-reject step and without computation of
*     the actions act0 and act1.
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The elements of the action arrays act0 and act1 are
*
*  act[0]        action of the momentum field,
*  act[1]        gauge field action,
*  act[2+n]      pseudo-fermion action number n,
*
* where the pseudo-fermion actions are counted from n=0 in steps of 1 as
* they appear in the action array smd.iact in the structure smd returned
* by smd_parms(). The arrays must thus have at least smd.nact+1 elements.
*
* The SMD algorithm is implemented as specified by the parameter data base.
* Accepted new gauge field configurations are renormalized to SU(3) on all
* active links. If a configuration is rejected, the fields are reset to the
* values they had at the beginning of the molecular-dynamics evolution and
* the momentum field is sign-flipped.
*
* The programs in this module assume that the counters, the chronological
* solver and the deflation subspace have been properly initialized. If a
* new configuration is rejected, the chronological solver is reset and the
* deflation subspace is regenerated. The latter is regularly updated along
* the molecular-dynamics trajectory as specified in the parameter data base
* [flags/dfl_parms.c].
*
* All programs internally take care of the phase-setting [see uflds/uflds.c].
* In the case of the programs smd_reset_dfl(), smd_init() and smd_action(),
* the phase of the gauge field is unchanged on exit, and the field is left
* in the phase-unset state at the end of the programs run_smd*().
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define SMD_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "lattice.h"
#include "utils.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
#include "global.h"

static su3_dble ubnd[3] ALIGNED16;


static void chk_mode_regen(int isp,int *status)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if ((sp.solver==DFL_SAP_GCR)&&(status[2]>0))
      add2counter("modes",2,status+2);
}


void smd_reset_dfl(void)
{
   int is,status[2];
   dfl_parms_t dfl;

   dfl=dfl_parms();

   if (dfl.Ns)
   {
      is=query_flags(UD_PHASE_SET);
      if (is==0)
         set_ud_phase();

      dfl_modes2(status);
      error_root((status[1]<0)||((status[1]==0)&&(status[0]<0)),1,
                 "smd_reset_dfl [smd.c]","Deflation subspace generation "
                 "failed (status = %d;%d)",status[0],status[1]);

      if (status[1]==0)
         add2counter("modes",0,status);
      else
         add2counter("modes",2,status+1);

      start_dfl_upd();

      if (is==0)
         unset_ud_phase();
   }
}


void smd_init(void)
{
   int i,is,nact,*iact;
   int status[3];
   double *mu;
   smd_parms_t smd;
   action_parms_t ap;

   random_mom();
   smd=smd_parms();

   if (smd.npf)
   {
      is=query_flags(UD_PHASE_SET);
      if (is==0)
         set_ud_phase();

      nact=smd.nact;
      iact=smd.iact;
      mu=smd.mu;

      for (i=0;i<nact;i++)
      {
         ap=action_parms(iact[i]);

         if (ap.action!=ACG)
         {
            set_sw_parms(sea_quark_mass(ap.im0));
            status[2]=0;

            if (ap.action==ACF_TM1)
               (void)(setpf1(mu[ap.imu[0]],ap.ipf,0));
            else if (ap.action==ACF_TM1_EO)
               (void)(setpf4(mu[ap.imu[0]],ap.ipf,0,0));
            else if (ap.action==ACF_TM1_EO_SDET)
               (void)(setpf4(mu[ap.imu[0]],ap.ipf,1,0));
            else if (ap.action==ACF_TM2)
            {
               (void)(setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                             0,status));
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else if (ap.action==ACF_TM2_EO)
            {
               (void)(setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                             0,status));
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else if (ap.action==ACF_RAT)
            {
               (void)(setpf3(ap.irat,ap.ipf,0,ap.isp[1],0,status));
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else if (ap.action==ACF_RAT_SDET)
            {
               (void)(setpf3(ap.irat,ap.ipf,1,ap.isp[1],0,status));
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else
               error(1,1,"smd_init [smd.c]","Unknown action");
         }
      }

      if (is==0)
         unset_ud_phase();
   }
}


void smd_action(qflt *act)
{
   int i,is,n,nact,*iact;
   int status[3];
   double *mu;
   smd_parms_t smd;
   action_parms_t ap;
   force_parms_t fp;

   act[0]=momentum_action(0);
   act[1]=action0(0);

   smd=smd_parms();

   if (smd.npf)
   {
      is=query_flags(UD_PHASE_SET);
      if (is==0)
         set_ud_phase();

      nact=smd.nact;
      iact=smd.iact;
      mu=smd.mu;
      n=2;

      for (i=0;i<nact;i++)
      {
         ap=action_parms(iact[i]);

         if (ap.action!=ACG)
         {
            set_sw_parms(sea_quark_mass(ap.im0));
            fp=force_parms(matching_force(iact[i]));
            status[2]=0;

            if (ap.action==ACF_TM1)
               act[n]=action1(mu[ap.imu[0]],ap.ipf,ap.isp[0],fp.icr[0],
                              0,status);
            else if (ap.action==ACF_TM1_EO)
               act[n]=action4(mu[ap.imu[0]],ap.ipf,0,ap.isp[0],fp.icr[0],
                              0,status);
            else if (ap.action==ACF_TM1_EO_SDET)
               act[n]=action4(mu[ap.imu[0]],ap.ipf,1,ap.isp[0],fp.icr[0],
                              0,status);
            else if (ap.action==ACF_TM2)
               act[n]=action2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                              fp.icr[0],0,status);
            else if (ap.action==ACF_TM2_EO)
               act[n]=action5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                              fp.icr[0],0,status);
            else if (ap.action==ACF_RAT)
               act[n]=action3(ap.irat,ap.ipf,0,ap.isp[0],0,status);
            else if (ap.action==ACF_RAT_SDET)
               act[n]=action3(ap.irat,ap.ipf,1,ap.isp[0],0,status);
            else
               error(1,1,"smd_action [smd.c]","Unknown action");

            chk_mode_regen(ap.isp[0],status);
            add2counter("action",iact[i],status);
            n+=1;
         }
      }

      if (is==0)
         unset_ud_phase();
   }
}


static void rotate_flds(void)
{
   int i,nact,*iact;
   int status[6];
   double s,c1,c2,*mu;
   smd_parms_t smd;
   action_parms_t ap;
   force_parms_t fp;

   smd=smd_parms();

   s=sinh(smd.gamma*smd.eps);
   c1=1.0/(s+sqrt(1.0+s*s));
   c2=sqrt(2.0*c1*s);
   rotate_mom(c1,c2);

   if (smd.npf)
   {
      set_ud_phase();
      reset_chrono(1);

      nact=smd.nact;
      iact=smd.iact;
      mu=smd.mu;

      for (i=0;i<nact;i++)
      {
         ap=action_parms(iact[i]);

         if (ap.action!=ACG)
         {
            set_sw_parms(sea_quark_mass(ap.im0));
            fp=force_parms(matching_force(iact[i]));
            status[2]=0;

            if (ap.action==ACF_TM1)
               rotpf1(mu[ap.imu[0]],ap.ipf,fp.isp[0],fp.icr[0],c1,c2,status);
            else if ((ap.action==ACF_TM1_EO)||(ap.action==ACF_TM1_EO_SDET))
               rotpf4(mu[ap.imu[0]],ap.ipf,fp.isp[0],fp.icr[0],c1,c2,status);
            else if (ap.action==ACF_TM2)
            {
               rotpf2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,fp.isp[0],ap.isp[1],
                      fp.icr[0],c1,c2,status);
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else if (ap.action==ACF_TM2_EO)
            {
               rotpf5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,fp.isp[0],ap.isp[1],
                      fp.icr[0],c1,c2,status);
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            {
               rotpf3(ap.irat,ap.ipf,ap.isp[1],c1,c2,status);
               chk_mode_regen(ap.isp[1],status);
               add2counter("field",ap.ipf,status);
            }
            else
               error(1,1,"rotate_flds [smd.c]","Unknown action");
         }
      }
   }
}


static int accept_smd(qflt *act0,qflt *act1,su3_alg_dble *mold,
                      su3_dble *uold)
{
   int my_rank,i,iac;
   double da,r,*qsm[1];
   qflt act,dact;
   su3_alg_dble *mom;
   su3_dble *udb;
   mdflds_t *mdfs;
   smd_parms_t smd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   smd=smd_parms();
   iac=0;
   dact.q[0]=0.0;
   dact.q[1]=0.0;

   for (i=0;i<=smd.nact;i++)
   {
      act.q[0]=-act0[i].q[0];
      act.q[1]=-act0[i].q[1];
      add_qflt(act1[i].q,act.q,act.q);
      add_qflt(act.q,dact.q,dact.q);
   }

   if (NPROC>1)
   {
      qsm[0]=dact.q;
      global_qsum(1,qsm,qsm);
   }

   if (my_rank==0)
   {
      ranlxd(&r,1);
      da=dact.q[0];

      if (da<=0.0)
         iac=1;
      else if (r<=exp(-da))
         iac=1;
   }

   if (NPROC>1)
      MPI_Bcast(&iac,1,MPI_INT,0,MPI_COMM_WORLD);

   if (iac==0)
   {
      mdfs=mdflds();
      mom=(*mdfs).mom;
      flip_assign_alg2alg(4*VOLUME,mold,mom);

      udb=udfld();
      cm3x3_assign(4*VOLUME,uold,udb);
      if ((cpr[0]==(NPROC0-1))&&((bc_type()==1)||(bc_type()==2)))
         cm3x3_assign(3,ubnd,udb+4*VOLUME+7*(BNDRY/4));
      set_flags(UPDATED_UD);
      set_flags(UNSET_UD_PHASE);

      if (smd.npf)
      {
         reset_chrono(0);
         smd_reset_dfl();
      }
   }
   else
   {
      unset_ud_phase();
      renormalize_ud();
   }

   return iac;
}


int run_smd(qflt *act0,qflt *act1)
{
   int iac;
   su3_alg_dble *mom,**mold;
   su3_dble *udb,**uold;
   mdflds_t *mdfs;

   mold=reserve_wfd(1);
   uold=reserve_wud(1);

   mdfs=mdflds();
   mom=(*mdfs).mom;
   assign_alg2alg(4*VOLUME,mom,mold[0]);

   unset_ud_phase();
   udb=udfld();
   cm3x3_assign(4*VOLUME,udb,uold[0]);

   if ((cpr[0]==(NPROC0-1))&&((bc_type()==1)||(bc_type()==2)))
      cm3x3_assign(3,udb+4*VOLUME+7*(BNDRY/4),ubnd);

   rotate_flds();
   smd_action(act0);
   run_mdint();
   smd_action(act1);
   iac=accept_smd(act0,act1,mold[0],uold[0]);

   release_wud();
   release_wfd();

   return iac;
}


void run_smd_noacc0(qflt *act0,qflt *act1)
{
   rotate_flds();
   smd_action(act0);
   run_mdint();
   smd_action(act1);
   unset_ud_phase();
   renormalize_ud();
}


void run_smd_noacc1(void)
{
   rotate_flds();
   run_mdint();
   unset_ud_phase();
   renormalize_ud();
}
