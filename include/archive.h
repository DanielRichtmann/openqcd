
/*******************************************************************************
*
* File archive.h
*
* Copyright (C) 2011, 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef ARCHIVE_H
#define ARCHIVE_H

#ifndef SU3_H
#include "su3.h"
#endif

/* ARCHIVE_C */
extern void set_nio_streams(int nio);
extern int nio_streams(void);
extern void write_cnfg(char *out);
extern void read_cnfg(char *in);
extern void lat_sizes(char *in,int *nl);
extern void export_cnfg(char *out);
extern void import_cnfg(char *in,int mask);
extern void blk_sizes(char *in,int *nl,int *bs);
extern int blk_index(int *nl,int *bs,int *nb);
extern int blk_root_process(int *nl,int *bs,int *bo,int *nb,int *ib);
extern void blk_export_cnfg(int *bs,char *out);
extern void blk_import_cnfg(char *in,int mask);

/* MARCHIVE_C */
extern void write_mfld(char *out);
extern void read_mfld(char *in);
extern void export_mfld(char *out);
extern void import_mfld(char *in);
extern void blk_export_mfld(int *bs,char *out);
extern void blk_import_mfld(char *in);

/* SARCHIVE_C */
extern void write_sfld(char *out,int eo,spinor_dble *sd);
extern void read_sfld(char *in,int eo,spinor_dble *sd);
extern void export_sfld(char *out,int eo,spinor_dble *sd);
extern void import_sfld(char *in,int eo,spinor_dble *sd);
extern void blk_export_sfld(int *bs,char *out,int eo,spinor_dble *sd);
extern void blk_import_sfld(char *in,int eo,spinor_dble *sd);

#endif
