
/*******************************************************************************
*
* File msfcts.h
*
* Copyright (C) 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef MSFCTS_H
#define MSFCTS_H

#ifndef SU3_H
#include "su3.h"
#endif

/* FARCHIVE */
extern void write_flds(char *out,int n,double **f);
extern void read_flds(char *in,int n,double **f);
extern void export_flds(char *out,int n,double **f);
extern void import_flds(char *in,int n,double **f);
extern void blk_export_flds(int *bs,char *out,int n,double **f);
extern void blk_import_flds(char *in,int n,double **f);

/* FLDOPS_C */
extern void gather_fld(double *f,complex_dble *rf);
extern void scatter_fld(complex_dble *rf,double *f);
extern void apply_fft(int type,complex_dble *rf,complex_dble *rft);
extern void convolute_flds(int *s,double *f,double *g,
                           complex_dble *rf,complex_dble *rg,double *fg);
extern void shift_fld(int *s,double *f,complex_dble *rf,double *g);
extern void copy_flds(double *f,double *g);
extern void add_flds(double *f,double *g);
extern void mul_flds(double *f,double *g);
extern void mulr_fld(double r,double *f);

/* LATAVG_C */
extern void sphere_fld(int r,double *f);
extern void sphere3d_fld(int r,double *f);
extern void sphere_sum(int dmax,double *f,double *sm);
extern void sphere3d_sum(int dmax,double *f,double *sm);
extern double avg_fld(double *f);
extern double center_fld(double *f);
extern void cov_fld(int dmax,double *f,double *g,
                    complex_dble *rf,complex_dble *rg,double *w,double *cov);
extern void cov3d_fld(int dmax,double *f,double *g,
                      complex_dble *rf,complex_dble *rg,double *w,double *cov);

/* PT2PT_C */
void pt2pt(int i3d,int r,double *f,
           complex_dble *rf,double *w1,double *w2,double *obs);

#endif
