
/*******************************************************************************
*
* File dfl_subspace.c
*
* Copyright (C) 2007-2013, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic programs related to the deflation subspace.
*
*   void dfl_s2v(spinor *s,complex *v)
*     Assigns the components of the global spinor field s along the
*     deflation subspace to the vector field v.
*
*   void dfl_v2s(complex *v,spinor *s)
*     Assigns the element of the deflation subspace corresponding to
*     the vector field v to the global spinor field s.
*
*   void dfl_subspace(spinor **mds)
*     Copies the global spinor fields mds[0],..,mds[Ns-1] to the fields
*     b.s[1],..,b.s[Ns] on the blocks b of the DFL_BLOCKS grid. The block
*     fields are then orthonormalized using the Gram-Schmidt process with
*     double-precision accuracy.
*      In this basis of fields, the modes mds[0],..,mds[Ns-1] are given by
*     fields vmds[0],..,vmds[Ns-1] of Ns*nb complex numbers, where nb is
*     the number of blocks in the block grid. These fields are assigned to
*     the last Ns single-precision vector fields of the array returned by
*     vflds() [vflds/vflds.c].
*
* The deflation subspace is spanned by the fields (*b).s[1],..,(*b).s[Ns]
* on the blocks b of the DFL_BLOCKS grid. The number Ns of fields is set by
* the program dfl_set_parms() [flags/dfl_parms.c].
*
* Any spinor field in the deflation subspace is a linear combination of the
* basis elements on the blocks. The associated complex coefficients form a
* vector field of the type described in vflds/vflds.c. Such fields are thus
* in one-to-one correspondence with the deflation modes. In particular, the
* deflation subspace contains the global spinor fields from which it was
* created by the program dfl_subspace().
*
* The program dfl_subspace() allocates the DFL_BLOCKS block grid if it is
* not already allocated. This program involves global operations and must be
* called simultaneously on all processes.
*
*******************************************************************************/

#define DFL_SUBSPACE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "block.h"
#include "dfl.h"
#include "global.h"


void dfl_s2v(spinor *s,complex *v)
{
   int Ns,nb,nbh,isw;
   int n,m,i,vol;
   block_t *b;
   spinor **sb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

   for (n=0;n<nb;n++)
   {
      if (n<nbh)
         m=n+isw*nbh;
      else
         m=n-isw*nbh;

      assign_s2sblk(DFL_BLOCKS,m,ALL_PTS,s,0);
      sb=b[m].s;

      for (i=1;i<=Ns;i++)
      {
         (*v)=spinor_prod(vol,0,sb[i],sb[0]);
         v+=1;
      }
   }
}


void dfl_v2s(complex *v,spinor *s)
{
   int Ns,nb,nbh,isw;
   int n,m,i,vol;
   block_t *b;
   spinor **sb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

   for (n=0;n<nb;n++)
   {
      if (n<nbh)
         m=n+isw*nbh;
      else
         m=n-isw*nbh;

      sb=b[m].s;
      set_s2zero(vol,sb[0]);

      for (i=1;i<=Ns;i++)
      {
         mulc_spinor_add(vol,sb[0],sb[i],*v);
         v+=1;
      }

      assign_sblk2s(DFL_BLOCKS,m,ALL_PTS,0,s);
   }
}


void dfl_subspace(spinor **mds)
{
   int Ns,nb,nbh,isw;
   int n,m,i,j,vol;
   complex **vs,*v;
   complex_dble z;
   complex_qflt cqsm;
   block_t *b;
   spinor **sb;
   spinor_dble **sdb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;

   error_root(Ns==0,1,"dfl_subspace [dfl_subspace.c]",
              "Deflation subspace parameters are not set");

   b=blk_list(DFL_BLOCKS,&nb,&isw);

   if (nb==0)
   {
      alloc_bgr(DFL_BLOCKS);
      b=blk_list(DFL_BLOCKS,&nb,&isw);
   }

   nbh=nb/2;
   vol=(*b).vol;
   vs=vflds()+Ns;

   for (n=0;n<nb;n++)
   {
      if (n<nbh)
         m=n+isw*nbh;
      else
         m=n-isw*nbh;

      sb=b[m].s;
      sdb=b[m].sd;

      for (i=1;i<=Ns;i++)
      {
         assign_s2sdblk(DFL_BLOCKS,m,ALL_PTS,mds[i-1],0);
         v=vs[i-1]+Ns*n;

         for (j=1;j<i;j++)
         {
            assign_s2sd(vol,sb[j],sdb[1]);
            cqsm=spinor_prod_dble(vol,0,sdb[1],sdb[0]);
            z.re=cqsm.re.q[0];
            z.im=cqsm.im.q[0];

            (*v).re=(float)(z.re);
            (*v).im=(float)(z.im);
            v+=1;

            z.re=-(z).re;
            z.im=-(z).im;
            mulc_spinor_add_dble(vol,sdb[0],sdb[1],z);
         }

         (*v).re=(float)(normalize_dble(vol,0,sdb[0]));
         (*v).im=0.0f;
         v+=1;

         for (j=(i+1);j<=Ns;j++)
         {
            (*v).re=0.0f;
            (*v).im=0.0f;
            v+=1;
         }

         assign_sd2s(vol,sdb[0],sb[i]);
      }
   }

   set_flags(ERASED_AW);
   set_flags(ERASED_AWHAT);
}
