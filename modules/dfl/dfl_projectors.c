
/*******************************************************************************
*
* File dfl_projectors.c
*
* Copyright (C) 2007, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the deflation projectors
*
* The externally accessible functions are
*
*   void dfl_Ppro_dble(spinor_dble *sd,spinor_dble *rd)
*     Applies the orthogonal projector P to the deflation subspace to the 
*     global double-precision spinor field sd and assigns the result to 
*     the field rd (which may coincide with sd).
*
*   void dfl_Qpro_dble(spinor_dble *sd,spinor_dble *rd)
*     Applies the orthogonal projector Q=1-P to the orthogonal complement
*     of the deflation subspace to the global double-precision spinor field
*     sd and assigns the result to the field rd (which may coincide with sd).
*
*   void dfl_Lpro_dble(int nkv,int nmx,double res,double mu,
*                      spinor_dble *sd,spinor_dble *rd,int *status)
*     Applies the deflation projector P_L to the global double-precision
*     spinor field sd. Along the way, the little Dirac equation is solved
*     with source P*sd. On exit, P_L*sd is assigned to the field sd and 
*     the solution of the little system to the field rd (the meaning of
*     the other arguments is explained in the notes).
*
*   void dfl_RLpro_dble(int nkv,int nmx,double res,double mu,
*                       spinor_dble *sd,spinor_dble *rd,int *status)
*     Applies the deflation projectors P_R and P_L to the global double-
*     precision spinor fields sd and rd, respectively, assuming rd is   
*     equal to (Dw+i*mu*gamma_5*1e)*sd if the twisted-mass flag is set or
*     (Dw+i*mu*gamma_5*1e)*sd if not. The meaning of the other arguments
*     is explained in the notes.
*
*   void dfl_RLpro(int nkv,int nmx,double res,double mu,
*                   spinor *s,spinor *r,int *status)
*     Applies the deflation projectors P_R and P_L to the global single-
*     precision spinor fields s and r, respectively, assuming r is equal
*     to (Dw+i*mu*gamma_5*1e)*s if the twisted-mass flag is set or
*     (Dw+i*mu*gamma_5)*s if not. The meaning of the other arguments is
*     explained in the notes.
*
* Notes:
*
* The deflation projectors calculated in this module were introduced in
*
*  M. Luescher: "Local coherence and deflation of the low quark modes
*                in lattice QCD", JHEP 0707 (2007) 081
*
* The programs dfl_Lpro_dble(), dfl_RLpro_dble() and dfl_RLpro() require
* the solution of the little Dirac equation for a given source field. The
* solution is obtained using the solver program ltl_gcr(), with arguments
* nkv,nmx,res,mu and status (see ltl_grc.c).
*
* All programs in this module assume that the deflation subspace has been
* defined using the program dfl_subspace(). The workspaces of fields needed
* are
*
*      program             complex      complex_dble     spinor_dble
*
*  dfl_Lpro_dble()        2*nkv+1           4                1
*  dfl_RLpro_dble()       2*nkv+1           4                0
*  dfl_RLpro()            2*nkv+2           4                0
*
* (see utils/wspace.c).
*
*******************************************************************************/

#define DFL_PROJECTORS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "flags.h"
#include "block.h"
#include "uflds.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "little.h"
#include "dfl.h"
#include "global.h"

static int Ns=0,nv,vol;
static complex_dble *cs;


static void set_constants(void)
{
   int *bs;
   dfl_parms_t dfl;
   dfl_grid_t grd;
   
   dfl=dfl_parms();   
   grd=dfl_geometry();

   Ns=dfl.Ns;
   nv=Ns*grd.nb;

   bs=dfl.bs;
   vol=bs[0]*bs[1]*bs[2]*bs[3];

   cs=amalloc(Ns*sizeof(*cs),ALIGN);
   error(cs==NULL,1,"set_constants [dfl_projectors.c]",
         "Unable to allocate auxiliary array");
}


void dfl_Ppro_dble(spinor_dble *sd,spinor_dble *rd)
{
   int nb,isw,n,k;
   spinor_dble **sdb;
   block_t *b;

   if (Ns==0)
      set_constants();
   
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   
   for (n=0;n<nb;n++)
   {
      assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,sd,0);
      sdb=(*b).sd;
      
      for (k=0;k<Ns;k++)
         cs[k]=spinor_prod_dble(vol,0,sdb[k+1],sdb[0]);

      set_sd2zero(vol,sdb[0]);

      for (k=0;k<Ns;k++)
         mulc_spinor_add_dble(vol,sdb[0],sdb[k+1],cs[k]);
         
      assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,rd);
      b+=1;
   }
}


void dfl_Qpro_dble(spinor_dble *sd,spinor_dble *rd)
{
   int nb,isw,n,k;
   spinor_dble **sdb;
   block_t *b;

   if (Ns==0)
      set_constants();
   
   b=blk_list(DFL_BLOCKS,&nb,&isw);

   for (n=0;n<nb;n++)
   {
      assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,sd,0);
      sdb=(*b).sd;
      
      for (k=0;k<Ns;k++)
         project_dble(vol,0,sdb[0],sdb[k+1]);

      assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,rd);
      b+=1;      
   }
}


void dfl_Lpro_dble(int nkv,int nmx,double res,double mu,
                   spinor_dble *sd,spinor_dble *rd,int *status)
{
   complex_dble **wvd;
   spinor_dble **wsd;

   if (Ns==0)
      set_constants();
   
   wvd=reserve_wvd(1);
   dfl_sd2vd(sd,wvd[0]);
   ltl_gcr(nkv,nmx,res,mu,wvd[0],wvd[0],status);
   dfl_vd2sd(wvd[0],rd);
   release_wvd();

   sw_term(NO_PTS);
   wsd=reserve_wsd(1);
   Dw_dble(mu,rd,wsd[0]);
   mulr_spinor_add_dble(VOLUME,sd,wsd[0],-1.0);
   release_wsd();
}


void dfl_RLpro_dble(int nkv,int nmx,double res,double mu,
                    spinor_dble *sd,spinor_dble *rd,int *status)
{
   complex_dble **wvd;

   if (Ns==0)
      set_constants();
   
   wvd=reserve_wvd(1);
   dfl_sd2vd(rd,wvd[0]);
   ltl_gcr(nkv,nmx,res,mu,wvd[0],wvd[0],status);
   dfl_sub_vd2sd(wvd[0],sd);
   release_wvd();

   sw_term(NO_PTS);
   Dw_dble(mu,sd,rd);
}


void dfl_RLpro(int nkv,int nmx,double res,double mu,
               spinor *s,spinor *r,int *status)
{
   complex **wv;
   complex_dble **wvd;

   if (Ns==0)
      set_constants();
   
   wv=reserve_wv(1);   
   wvd=reserve_wvd(1);   
   dfl_s2v(r,wv[0]);
   assign_v2vd(nv,wv[0],wvd[0]);
   ltl_gcr(nkv,nmx,res,mu,wvd[0],wvd[0],status);
   assign_vd2v(nv,wvd[0],wv[0]);
   dfl_sub_v2s(wv[0],s);
   release_wvd();
   release_wv();

   if (query_flags(U_MATCH_UD)!=1)
      assign_ud2u();
   
   if ((query_flags(SW_UP2DATE)!=1)||(query_flags(SW_E_INVERTED)!=0)||
       (query_flags(SW_O_INVERTED)!=0))
   {
      sw_term(NO_PTS);
      assign_swd2sw();
   }

   Dw((float)(mu),s,r);
}
