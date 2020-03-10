
/*******************************************************************************
*
* File utils.h
*
* Copyright (C) 2011, 2016-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <limits.h>
#include <float.h>
#include <stdlib.h>

#ifndef SU3_H
#include "su3.h"
#endif

#if (SHRT_MAX==0x7fffffff)
typedef short int stdint_t;
typedef unsigned short int stduint_t;
#elif (INT_MAX==0x7fffffff)
typedef int stdint_t;
typedef unsigned int stduint_t;
#elif (LONG_MAX==0x7fffffff)
typedef long int stdint_t;
typedef unsigned long int stduint_t;
#else
#error : There is no four-byte integer type on this machine
#endif

#undef UNKNOWN_ENDIAN
#undef LITTLE_ENDIAN
#undef BIG_ENDIAN

#define UNKNOWN_ENDIAN 0
#define LITTLE_ENDIAN 1
#define BIG_ENDIAN 2

#undef IMAX
#define IMAX(n,m) ((n)+((m)-(n))*((m)>(n)))

typedef enum
{
   ALL_PTS,EVEN_PTS,ODD_PTS,NO_PTS,PT_SETS
} ptset_t;

typedef struct
{
   unsigned int d,p;
   size_t size,*n;
   void *a;
} array_t;

/* ARRAY */
extern array_t *alloc_array(unsigned int d,size_t *n,size_t size,
                            unsigned int p);
extern void free_array(array_t *a);
extern void write_array(FILE *fdat,array_t *a);
extern void read_array(FILE *fdat,array_t *a);

/* ENDIAN_C */
extern int endianness(void);
extern void bswap_int(int n,void *a);
extern void bswap_double(int n,void *a);

/* ERROR_C */
extern void set_error_file(char *path,int loc_flag);
extern void error(int test,int no,char *name,char *format,...);
extern void error_root(int test,int no,char *name,char *format,...);
extern void error_loc(int test,int no,char *name,char *format,...);

/* HSUM_C */
extern int init_hsum(int n);
extern void reset_hsum(int id);
extern void add_to_hsum(int id,double *x);
extern void local_hsum(int id,double *sx);
extern void global_hsum(int id,double *sx);

/* MUTILS_C */
extern void check_machine(void);
extern int find_opt(int argc,char *argv[],char *opt);
extern int fdigits(double x);
extern void check_dir(char* dir);
extern void check_dir_root(char* dir);
extern int check_file(char* file,char* mode);
extern int name_size(char *format,...);
extern long find_section(char *title);
extern long read_line(char *tag,char *format,...);
extern int count_tokens(char *tag);
extern void read_iprms(char *tag,int n,int *iprms);
extern void read_dprms(char *tag,int n,double *dprms);
extern void copy_file(char *in,char *out);

/* QSUM_C */
extern void acc_qflt(double u,double *qr);
extern void add_qflt(double *qu,double *qv,double *qr);
extern void global_qsum(int n,double **qu,double **qr);
extern void scl_qflt(double u,double *qr);
extern void mul_qflt(double *qu,double *qv,double *qr);

/* UTILS_C */
extern int safe_mod(int x,int y);
extern void *amalloc(size_t size,int p);
extern void afree(void *addr);
extern int mpi_permanent_tag(void);
extern int mpi_tag(void);
extern void message(char *format,...);

/* WSPACE_C */
extern void alloc_wud(int n);
extern su3_dble **reserve_wud(int n);
extern int release_wud(void);
extern void alloc_wfd(int n);
extern su3_alg_dble **reserve_wfd(int n);
extern int release_wfd(void);
extern void alloc_ws(int n);
extern spinor **reserve_ws(int n);
extern int release_ws(void);
extern void alloc_wsd(int n);
extern void wsd_uses_ws(void);
extern spinor_dble **reserve_wsd(int n);
extern int release_wsd(void);
extern void alloc_wv(int n);
extern complex **reserve_wv(int n);
extern int release_wv(void);
extern void alloc_wvd(int n);
extern complex_dble **reserve_wvd(int n);
extern int release_wvd(void);
extern size_t wsp_msize(void);
extern void print_wsp(void);

#endif
