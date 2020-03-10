
/*******************************************************************************
*
* File auxfcts.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Common functions used in the force check programs.
*
*   void save_ranlux(void)
*     Saves the state of the ranlux random number generator.
*
*   void restore_ranlux(void)
*     Restores the state of the ranlux random number generator to the
*     last saved state.
*
*   void rot_ud(double eps)
*     Multiplies the global double-precision link variables U(x,mu) by
*     exp(eps*mom(x,mu)) on active links (x,mu).
*
*   void check_bnd_fld(su3_alg_dble *fld)
*     Checks whether the link field fld vanishes on the inactive links.
*
* All these programs act globally and must be called on all MPI processes
* simultaneously.
*
*******************************************************************************/

#define AUXFCTS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "random.h"
#include "uflds.h"
#include "mdflds.h"
#include "sflds.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int *rlxd_state,*rlxs_state=NULL;


void save_ranlux(void)
{
   int nlxs,nlxd;

   if (rlxs_state==NULL)
   {
      nlxs=rlxs_size();
      nlxd=rlxd_size();

      rlxs_state=malloc((nlxs+nlxd)*sizeof(int));
      rlxd_state=rlxs_state+nlxs;

      error(rlxs_state==NULL,1,"save_ranlux [auxfcts.c]",
            "Unable to allocate state arrays");
   }

   rlxs_get(rlxs_state);
   rlxd_get(rlxd_state);
}


void restore_ranlux(void)
{
   error(rlxs_state==NULL,1,"restore_ranlux [auxfcts.c]",
         "The state of the ranlux generator was not previously saved");

   rlxs_reset(rlxs_state);
   rlxd_reset(rlxd_state);
}


void rot_ud(double eps)
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
         mom+=1;
         u+=1;

         if (bc!=0)
            expXsu3(eps,mom,u);
         mom+=1;
         u+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            if (bc!=1)
               expXsu3(eps,mom,u);
            mom+=1;
            u+=1;
         }
      }
      else if (t==(N0-1))
      {
         if (bc!=0)
            expXsu3(eps,mom,u);
         mom+=1;
         u+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            expXsu3(eps,mom,u);
            mom+=1;
            u+=1;
         }
      }
      else
      {
         for (ifc=0;ifc<8;ifc++)
         {
            expXsu3(eps,mom,u);
            mom+=1;
            u+=1;
         }
      }
   }

   set_flags(UPDATED_UD);
}


static int is_zero(su3_alg_dble *f)
{
   int ie;

   ie=1;
   ie&=((*f).c1==0.0);
   ie&=((*f).c2==0.0);
   ie&=((*f).c3==0.0);
   ie&=((*f).c4==0.0);
   ie&=((*f).c5==0.0);
   ie&=((*f).c6==0.0);
   ie&=((*f).c7==0.0);
   ie&=((*f).c8==0.0);

   return ie;
}


void check_bnd_fld(su3_alg_dble *fld)
{
   int bc,ix,t,ifc,ie;

   bc=bc_type();
   ie=0;

   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if ((t==0)&&(bc==0))
      {
         ie|=is_zero(fld);
         fld+=1;

         ie|=(is_zero(fld)^0x1);
         fld+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            ie|=is_zero(fld);
            fld+=1;
         }
      }
      else if ((t==0)&&(bc==1))
      {
         ie|=is_zero(fld);
         fld+=1;

         ie|=is_zero(fld);
         fld+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            ie|=(is_zero(fld)^0x1);
            fld+=1;
         }
      }
      else if ((t==(N0-1))&&(bc==0))
      {
         ie|=(is_zero(fld)^0x1);
         fld+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            ie|=is_zero(fld);
            fld+=1;
         }
      }
      else
      {
         for (ifc=0;ifc<8;ifc++)
         {
            ie|=is_zero(fld);
            fld+=1;
         }
      }
   }

   error(ie!=0,1,"check_bnd_fld [auxfcts.c]",
         "Link field vanishes on an incorrect set of links");
}
