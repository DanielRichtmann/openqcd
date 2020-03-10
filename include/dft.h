
/*******************************************************************************
*
* File dft.h
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef DFT_H
#define DFT_H

#ifndef FLAGS_H
#include "flags.h"
#endif

/* DFT4D_C */
extern void dft4d(int id,complex_dble *f,complex_dble *ft);
extern void inv_dft4d(int id,complex_dble *ft,complex_dble *f);

/* DFT_COM_C */
extern void dft_gather(int mu,int *nx,int *mf,complex_dble **lf,
                       complex_dble **f);
extern void dft_scatter(int mu,int *nx,int *mf,complex_dble **f,
                        complex_dble **lf);

/* DFT_SHUF_C */
extern void dft_shuf(int nx,int ny,int csize,complex_dble *f,complex_dble *rf);

/* FFT_C */
extern void fft(dft_parms_t *dp,int nf,complex_dble **f,complex_dble **ft);
extern void inv_fft(dft_parms_t *dp,int nf,complex_dble **f,complex_dble **ft);

/* SMALL_DFT_C */
extern void small_dft(int s,int n,int nf,complex_dble *w,complex_dble **f,
                      complex_dble **ft);

#endif
