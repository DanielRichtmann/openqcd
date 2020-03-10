
/*******************************************************************************
*
* File update.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Molecular-dynamics elementary integration steps.
*
*   void update_mom(void)
*     Subtracts the current force field from the momentum field (see the
*     notes). The operation is performed on the active links only.
*
*   void update_ud(double eps)
*     Replaces the gauge field variables ud by exp{eps*mom}*ud, where mom
*     is the current momentum field (see the notes). Only the active link
*     variables are updated and the molecular-dynamics time is advanced by
*     eps.
*
*   void start_dfl_upd(void)
*     Starts (or restarts) the deflation-subspace update cycle (see the
*     notes).
*
*   void dfl_upd(void)
*     Reads the molecular-dynamics time and updates the deflation subspace
*     if an update is due according to the chosen update scheme.
*
* The programs update_mom() and update_ud() act on the global fields, i.e.
* the gauge, momentum and force fields that can be accessed through udfld()
* [uflds/uflds.c] and mdflds() [mdflds/mdflds.c], respectively.
*
* The update scheme for the deflation subspace is defined by the parameter
* data base [flags/dfl_parms.c]. No subspace initialization is performed
* by start_dfl_upd(), only the update time is set to the current time. The
* program dfl_upd() does nothing if start_dfl_upd() has not been called.
*
* The program dfl_upd() assumes that the deflation subspace and the counters
* have been properly initialized. If phase-periodic boundary conditions are
* chosen, the calling program must ensure that the gauge field is phase-set.
*
* The programs in this module perform global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define UPDATE_C

#include <stdio.h>
#include <stdlib.h>
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "su3fcts.h"
#include "dfl.h"
#include "update.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int nsm=0;
static double dtau=1.0,rtau=0.0;


void update_mom(void)
{
   int bc,ix,t,ifc;
   su3_alg_dble *mom,*frc;
   mdflds_t *mdfs;

   bc=bc_type();
   mdfs=mdflds();
   mom=(*mdfs).mom;
   frc=(*mdfs).frc;

   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (t==0)
      {
         _su3_alg_sub_assign(mom[0],frc[0]);
         mom+=1;
         frc+=1;

         if (bc!=0)
         {
            _su3_alg_sub_assign(mom[0],frc[0]);
         }

         mom+=1;
         frc+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            if (bc!=1)
            {
               _su3_alg_sub_assign(mom[0],frc[0]);
            }

            mom+=1;
            frc+=1;
         }
      }
      else if (t==(N0-1))
      {
         if (bc!=0)
         {
            _su3_alg_sub_assign(mom[0],frc[0]);
         }

         mom+=1;
         frc+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            _su3_alg_sub_assign(mom[0],frc[0]);
            mom+=1;
            frc+=1;
         }
      }
      else
      {
         for (ifc=0;ifc<8;ifc++)
         {
            _su3_alg_sub_assign(mom[0],frc[0]);
            mom+=1;
            frc+=1;
         }
      }
   }
}


void update_ud(double eps)
{
   int bc,ix,t,ifc;
   su3_dble *u;
   su3_alg_dble *mom;
   mdflds_t *mdfs;

   bc=bc_type();
   mdfs=mdflds();
   mom=(*mdfs).mom;
   u=udfld();

   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (t==0)
      {
         expXsu3(eps,mom,u);
         u+=1;
         mom+=1;

         if (bc!=0)
            expXsu3(eps,mom,u);
         u+=1;
         mom+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            if (bc!=1)
               expXsu3(eps,mom,u);
            u+=1;
            mom+=1;
         }
      }
      else if (t==(N0-1))
      {
         if (bc!=0)
            expXsu3(eps,mom,u);
         u+=1;
         mom+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            expXsu3(eps,mom,u);
            u+=1;
            mom+=1;
         }
      }
      else
      {
         for (ifc=0;ifc<8;ifc++)
         {
            expXsu3(eps,mom,u);
            u+=1;
            mom+=1;
         }
      }
   }

   step_mdtime(eps);
   set_flags(UPDATED_UD);
}


void start_dfl_upd(void)
{
   dfl_upd_parms_t dup;

   dup=dfl_upd_parms();
   dtau=dup.dtau;
   nsm=dup.nsm;
   rtau=mdtime();
}


void dfl_upd(void)
{
   int status[2];
   double tau;
   dfl_parms_t dfl;

   tau=mdtime();

   if ((nsm>0)&&((tau-rtau)>dtau))
   {
      dfl=dfl_parms();

      if (dfl.Ns)
      {
         dfl_update2(nsm,status);
         error_root((status[1]<0)||((status[1]==0)&&(status[0]<0)),1,
                    "dfl_upd [update.c]","Deflation subspace update "
                    "failed (status = %d;%d)",status[0],status[1]);

         if (status[1]==0)
            add2counter("modes",1,status);
         else
            add2counter("modes",2,status+1);

         rtau=tau;
      }
   }
}
