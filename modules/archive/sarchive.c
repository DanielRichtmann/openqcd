
/*******************************************************************************
*
* File sarchive.c
*
* Copyright (C) 2005-2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write global double-precision spinor fields.
*
*   void write_sfld(char *out,int eo,spinor_dble *sd)
*     Writes the lattice sizes, the process grid sizes, the coordinates
*     of the calling process, the local square-norm of the spinor field
*     sd and the local part of the field to the file "out".
*
*   void read_sfld(char *in,int eo,spinor_dble *sd)
*     Reads the local part of the spinor field sd from the file "in",
*     assuming the field was written to the file by write_sfld().
*
*   void export_sfld(char *out,int eo,spinor_dble *sd)
*     Writes the lattice sizes, the square-norm of the spinor field sd
*     and the field to the file "out" from process 0. The field variables
*     are written in the "universal" order specified in main/README.io.
*
*   void import_sfld(char *in,int eo,spinor_dble *sd)
*     Reads the spinor field sd from the file "in" on process 0, assuming
*     the field was written by the program export_sfld().
*
*   void blk_export_sfld(int *bs,char *out,int eo,spinor_dble *sd)
*     Divides the lattice into logical blocks with sizes bs[0],..,bs[3].
*     The MPI processes, where the local lattice contains the base point of
*     a block, then write the lattice sizes, the block sizes, the base point
*     coordinates, the square-norm of the spinor field sd and the part of
*     the field residing on the block to the file "out". The field variables
*     are written in the "universal" order (see main/README.io).
*
*   void blk_import_sfld(char *in,int eo,spinor_dble *sd)
*     Reads the spinor field sd from the file "in", assuming the field was
*     written by the program blk_export_sfld(). The program reads the block
*     size from the configuration file on the MPI process with coordinates
*     (0,0,0,0) in the process grid. Each MPI process containing the base
*     point of a block then reads the field variables on that block from
*     the configuration file.
*
* The spinor fields are assumed to be global quark fields as described in
* main/README.global. Only their physical components (i.e. the spinors on
* the local lattices) are written and read. The parameter "eo" specifies
* whether the I/O includes the spinors on all points (eo=0) or only the
* even points (eo=1). The value of eo is stored on the configuration files
* just before the square-norm of the field.
*
* See main/README.io for a description of the configuration I/O strategies
* implemented in openQCD and the specification of the "universal" storage
* format. The square-norm of the fields are computed by calling the program
* norm_square_dble() [linalg/salg_dble.c].
*
* The programs that perform parallel I/O write the field variables residing
* on B0xB1xB2xB3 blocks of lattice points to different files. When one of
* these functions is executed, the number of parallel I/O streams last set
* by set_nio_streams() must divide the number of blocks. Moreover, the block
* sizes must be multiples of the local lattice sizes. An error occurs if
* any of these conditions is violated.
*
* Independently of the machine, the export functions write the data to the
* output file in little-endian byte order. Integers and double-precision
* numbers on the output file occupy 4 and 8 bytes, respectively, the latter
* being formatted according to the IEEE-754 standard. The import function
* assumes the data on the input file to be little endian and converts them
* to big-endian order if the machine is big endian. Exported fields can
* thus be safely exchanged between different machines.
*
* All programs in this module involve communications and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define SARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "lattice.h"
#include "linalg.h"
#include "sflds.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static const int nsz[4]={N0,N1,N2,N3};
static const int lsz[4]={L0,L1,L2,L3};
static const int diy[4]={L1*L2*L3,L2*L3,L3,1};

static int my_rank,tag0,tag1,endian;
static spinor_dble *sfld,*sbuf=NULL,*vbuf;


void write_sfld(char *out,int eo,spinor_dble *sd)
{
   int nio,vol,n,i,iw,ldat[17];
   double norm;
   qflt qnrm;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   error(sd==NULL,1,"write_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   error(iup[0][0]==0,1,"write_sfld [sarchive.c]",
         "Geometry arrays are not set");
   error((eo<0)||(eo>1),1,"write_sfld [sarchive.c]",
         "The parameter eo is out of range");

   ldat[0]=NPROC0;
   ldat[1]=NPROC1;
   ldat[2]=NPROC2;
   ldat[3]=NPROC3;

   ldat[4]=L0;
   ldat[5]=L1;
   ldat[6]=L2;
   ldat[7]=L3;

   ldat[8]=NPROC0_BLK;
   ldat[9]=NPROC1_BLK;
   ldat[10]=NPROC2_BLK;
   ldat[11]=NPROC3_BLK;

   ldat[12]=cpr[0];
   ldat[13]=cpr[1];
   ldat[14]=cpr[2];
   ldat[15]=cpr[3];

   ldat[16]=eo;

   if (eo==0)
      vol=VOLUME;
   else
      vol=VOLUME/2;

   nio=nio_streams();
   error_root((nio<1)||(NPROC%nio),1,"write_sfld [sarchive.c]",
              "Improper number of parallel I/O streams");
   n=NPROC/nio;
   qnrm=norm_square_dble(vol,0,sd);
   norm=qnrm.q[0];

   for (i=0;i<n;i++)
   {
      if (i==(my_rank%n))
      {
         fout=fopen(out,"wb");
         error_loc(fout==NULL,1,"write_sfld [sarchive.c]",
                   "Unable to open output file");

         iw=fwrite(ldat,sizeof(int),17,fout);
         iw+=fwrite(&norm,sizeof(double),1,fout);
         iw+=fwrite(sd,sizeof(spinor_dble),vol,fout);
         error_loc(iw!=(18+vol),1,"write_sfld [sarchive.c]",
                   "Incorrect write count");

         fclose(fout);
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }
}


void read_sfld(char *in,int eo,spinor_dble *sd)
{
   int nio,vol,n,i,ir,ie,ldat[17];
   double norm0,norm1,eps;
   qflt qnrm;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   error(sd==NULL,1,"read_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   error(iup[0][0]==0,1,"read_sfld [sarchive.c]",
         "Geometry arrays are not set");
   error((eo<0)||(eo>1),1,"read_sfld [sarchive.c]",
         "The parameter eo is out of range");

   if (eo==0)
      vol=VOLUME;
   else
      vol=VOLUME/2;

   nio=nio_streams();
   error_root((nio<1)||(NPROC%nio),1,"read_sfld [sarchive.c]",
              "Improper number of parallel I/O streams");
   n=NPROC/nio;
   norm0=0.0;

   for (i=0;i<n;i++)
   {
      if (i==(my_rank%n))
      {
         fin=fopen(in,"rb");
         error_loc(fin==NULL,1,"read_sfld [sarchive.c]",
                   "Unable to open input file");

         ir=fread(ldat,sizeof(int),17,fin);

         ie=0;
         ie|=((ldat[0]!=NPROC0)||(ldat[1]!=NPROC1)||
              (ldat[2]!=NPROC2)||(ldat[3]!=NPROC3));
         ie|=((ldat[4]!=L0)||(ldat[5]!=L1)||
              (ldat[6]!=L2)||(ldat[7]!=L3));
         ie|=((ldat[8]!=NPROC0_BLK)||(ldat[9]!=NPROC1_BLK)||
              (ldat[10]!=NPROC2_BLK)||(ldat[11]!=NPROC3_BLK));
         ie|=((ldat[12]!=cpr[0])||(ldat[13]!=cpr[1])||
              (ldat[14]!=cpr[2])||(ldat[15]!=cpr[3])||(ldat[16]!=eo));
         error_loc(ie!=0,1,"read_sfld [sarchive.c]","Unexpected lattice data");

         ir+=fread(&norm0,sizeof(double),1,fin);
         ir+=fread(sd,sizeof(spinor_dble),vol,fin);
         error_loc(ir!=(18+vol),1,"read_sfld [sarchive.c]",
                   "Incorrect read count");

         fclose(fin);
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

   qnrm=norm_square_dble(vol,0,sd);
   norm1=qnrm.q[0];
   eps=64.0*(norm0+norm1)*DBL_EPSILON;
   error(fabs(norm1-norm0)>eps,1,"read_sfld [sarchive.c]",
         "Incorrect square norm");
}


static void alloc_sbuf(void)
{
   error(iup[0][0]==0,1,"alloc_sbuf [sarchive.c]",
         "Geometry arrays are not set");
   sbuf=amalloc((L3+N3)*sizeof(spinor_dble),ALIGN);
   vbuf=sbuf+L3;
   error(sbuf==NULL,1,"alloc_sbuf [sarchive.c]",
         "Unable to allocate auxiliary array");
   set_sd2zero(L3+N3,sbuf);

   tag0=mpi_permanent_tag();
   tag1=mpi_permanent_tag();
   endian=endianness();
}


static void get_spinors(int eo,int iy)
{
   int iym;
   spinor_dble *sb;

   sb=sbuf;
   iym=iy+L3;

   if (eo==0)
   {
      for (;iy<iym;iy++)
      {
         sb[0]=sfld[ipt[iy]];
         sb+=1;
      }
   }
   else
   {
      if (ipt[iy]>=(VOLUME/2))
         iy+=1;

      for (;iy<iym;iy+=2)
      {
         sb[0]=sfld[ipt[iy]];
         sb+=1;
      }
   }
}


static void set_spinors(int eo,int iy)
{
   int iym;
   spinor_dble *sb;

   sb=sbuf;
   iym=iy+L3;

   if (eo==0)
   {
      for (;iy<iym;iy++)
      {
         sfld[ipt[iy]]=sb[0];
         sb+=1;
      }
   }
   else
   {
      if (ipt[iy]>=(VOLUME/2))
         iy+=1;

      for (;iy<iym;iy+=2)
      {
         sfld[ipt[iy]]=sb[0];
         sb+=1;
      }
   }
}


static void get_line(int ip0,int eo,int *x,int dx)
{
   int nu,iy,l3,np[4];
   int k,kmn,kmx,dmy,ip1;
   MPI_Status stat;

   iy=0;

   for (nu=0;nu<4;nu++)
   {
      iy+=(x[nu]%lsz[nu])*diy[nu];
      np[nu]=x[nu]/lsz[nu];
   }

   if (eo==0)
      l3=L3;
   else
      l3=L3/2;

   kmn=np[3];
   kmx=kmn+(dx/L3);
   dmy=1;

   for (k=kmn;k<kmx;k++)
   {
      np[3]=k;
      ip1=ipr_global(np);

      if (my_rank==ip1)
         get_spinors(eo,iy);

      if (ip1!=ip0)
      {
         if (my_rank==ip0)
         {
            MPI_Send(&dmy,1,MPI_INT,ip1,tag0,MPI_COMM_WORLD);
            MPI_Recv(vbuf+k*l3,l3*24,MPI_DOUBLE,ip1,tag1,MPI_COMM_WORLD,&stat);
         }
         else if (my_rank==ip1)
         {
            MPI_Recv(&dmy,1,MPI_INT,ip0,tag0,MPI_COMM_WORLD,&stat);
            MPI_Send(sbuf,l3*24,MPI_DOUBLE,ip0,tag1,MPI_COMM_WORLD);
         }
      }
      else if (my_rank==ip0)
         assign_sd2sd(l3,sbuf,vbuf+k*l3);
   }
}


static void set_line(int ip0,int eo,int *x,int dx)
{
   int nu,iy,l3,np[4];
   int k,kmn,kmx,dmy,ip1;
   MPI_Status stat;

   iy=0;

   for (nu=0;nu<4;nu++)
   {
      iy+=(x[nu]%lsz[nu])*diy[nu];
      np[nu]=x[nu]/lsz[nu];
   }

   if (eo==0)
      l3=L3;
   else
      l3=L3/2;

   kmn=np[3];
   kmx=kmn+(dx/L3);
   dmy=1;

   for (k=kmn;k<kmx;k++)
   {
      np[3]=k;
      ip1=ipr_global(np);

      if (ip1!=ip0)
      {
         if (my_rank==ip0)
         {
            MPI_Send(vbuf+k*l3,l3*24,MPI_DOUBLE,ip1,tag1,MPI_COMM_WORLD);
            MPI_Recv(&dmy,1,MPI_INT,ip1,tag0,MPI_COMM_WORLD,&stat);
         }
         else if (my_rank==ip1)
         {
            MPI_Recv(sbuf,l3*24,MPI_DOUBLE,ip0,tag1,MPI_COMM_WORLD,&stat);
            MPI_Send(&dmy,1,MPI_INT,ip0,tag0,MPI_COMM_WORLD);
         }
      }
      else if (my_rank==ip1)
         assign_sd2sd(l3,vbuf+k*l3,sbuf);

      if (my_rank==ip1)
         set_spinors(eo,iy);
   }
}


void export_sfld(char *out,int eo,spinor_dble *sd)
{
   int iw,iwa,vol,n3,x[4],iprms[1];
   stdint_t lsize[5];
   double norm;
   qflt qnrm;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   error_root((eo<0)||(eo>1),1,"export_sfld [sarchive.c]",
              "Parameter eo is out of range");

   if (NPROC>1)
   {
      iprms[0]=eo;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=eo,1,"export_sfld [sarchive.c]",
            "Parameters are not global");
   }

   if (sbuf==NULL)
      alloc_sbuf();

   error(sd==NULL,1,"export_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   sfld=sd;

   if (eo==0)
   {
      vol=VOLUME;
      n3=N3;
   }
   else
   {
      vol=VOLUME/2;
      n3=N3/2;
   }

   qnrm=norm_square_dble(vol,1,sd);
   norm=qnrm.q[0];

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_sfld [sarchive.c]",
                 "Unable to open output file");

      lsize[0]=N0;
      lsize[1]=N1;
      lsize[2]=N2;
      lsize[3]=N3;
      lsize[4]=(stdint_t)(eo);

      if (endian==BIG_ENDIAN)
      {
         bswap_int(5,lsize);
         bswap_double(1,&norm);
      }

      iw=fwrite(lsize,sizeof(stdint_t),5,fout);
      iw+=fwrite(&norm,sizeof(double),1,fout);
      error_root(iw!=6,1,"export_sfld [sarchive.c]","Incorrect write count");
   }

   iwa=0;
   x[3]=0;

   for (x[0]=0;x[0]<N0;x[0]++)
   {
      for (x[1]=0;x[1]<N1;x[1]++)
      {
         for (x[2]=0;x[2]<N2;x[2]++)
         {
            get_line(0,eo,x,N3);

            if (my_rank==0)
            {
               if (endian==BIG_ENDIAN)
                  bswap_double(n3*24,vbuf);

               iw=fwrite(vbuf,sizeof(spinor_dble),n3,fout);
               iwa|=(iw!=n3);
            }
         }
      }
   }

   if (my_rank==0)
   {
      error_root(iwa!=0,1,"export_sfld [sarchive.c]","Incorrect write count");
      fclose(fout);
   }
}


void import_sfld(char *in,int eo,spinor_dble *sd)
{
   int ir,ira,vol,n3,x[4],iprms[1];
   stdint_t lsize[5];
   double norm0,norm1,eps;
   qflt qnrm;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   error_root((eo<0)||(eo>1),1,"import_sfld [sarchive.c]",
              "Parameter eo is out of range");

   if (NPROC>1)
   {
      iprms[0]=eo;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=eo,1,"import_sfld [sarchive.c]",
            "Parameters are not global");
   }

   if (sbuf==NULL)
      alloc_sbuf();

   error(sd==NULL,1,"import_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   sfld=sd;

   if (eo==0)
   {
      vol=VOLUME;
      n3=N3;
   }
   else
   {
      vol=VOLUME/2;
      n3=N3/2;
   }

   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_sfld [sarchive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),5,fin);
      ir+=fread(&norm0,sizeof(double),1,fin);
      error_root(ir!=6,1,"import_sfld [sarchive.c]","Incorrect read count");

      if (endian==BIG_ENDIAN)
      {
         bswap_int(5,lsize);
         bswap_double(1,&norm0);
      }

      error_root((lsize[0]!=N0)||(lsize[1]!=N1)||(lsize[2]!=N2)||
                 (lsize[3]!=N3)||(lsize[4]!=(stdint_t)(eo)),1,
                 "import_sfld [sarchive.c]","Lattice sizes do not match");
   }

   MPI_Bcast(&norm0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   ira=0;
   x[3]=0;

   for (x[0]=0;x[0]<N0;x[0]++)
   {
      for (x[1]=0;x[1]<N1;x[1]++)
      {
         for (x[2]=0;x[2]<N2;x[2]++)
         {
            if (my_rank==0)
            {
               ir=fread(vbuf,sizeof(spinor_dble),n3,fin);
               ira|=(ir!=n3);

               if (endian==BIG_ENDIAN)
                  bswap_double(n3*24,vbuf);
            }

            set_line(0,eo,x,N3);
         }
      }
   }

   if (my_rank==0)
   {
      error_root(ira!=0,1,"import_sfld [sarchive.c]","Incorrect read count");
      fclose(fin);
   }

   qnrm=norm_square_dble(vol,1,sd);
   norm1=qnrm.q[0];
   eps=64.0*(norm0+norm1)*DBL_EPSILON;
   error_root(fabs(norm1-norm0)>eps,1,"import_sfld [sarchive.c]",
              "Incorrect square norm");
}


void blk_export_sfld(int *bs,char *out,int eo,spinor_dble *sd)
{
   int ip0,nio,iw,iwa;
   int nl[4],bo[4];
   int mu,vol,bs3,bo3,x[4],y[4];
   int n,i,nb,ib,iprms[5];
   stdint_t lsize[13];
   double norm;
   qflt qnrm;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (NPROC>1)
   {
      for (mu=0;mu<4;mu++)
         iprms[mu]=bs[mu];

      iprms[4]=eo;

      MPI_Bcast(iprms,5,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=bs[0])||(iprms[1]!=bs[1])||(iprms[2]!=bs[2])||
            (iprms[3]!=bs[3])||(iprms[4]!=eo),1,"blk_export_sfld [sarchive.c]",
            "Parameters are not global");
   }

   if (sbuf==NULL)
      alloc_sbuf();

   error(sd==NULL,1,"blk_export_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   sfld=sd;

   for (mu=0;mu<4;mu++)
      nl[mu]=nsz[mu];
   ip0=blk_root_process(nl,bs,bo,&nb,&ib);

   if (eo==0)
   {
      vol=VOLUME;
      bs3=bs[3];
      bo3=bo[3];
   }
   else
   {
      vol=VOLUME/2;
      bs3=bs[3]/2;
      bo3=bo[3]/2;
   }

   nio=nio_streams();
   error_root((nio<1)||((nb%nio)!=0),1,"blk_export_sfld [sarchive.c]",
              "Improper number of parallel I/O streams");
   n=nb/nio;
   qnrm=norm_square_dble(vol,1,sd);
   norm=qnrm.q[0];

   for (i=0;i<n;i++)
   {
      if (i==(ib%n))
      {
         if (my_rank==ip0)
         {
            fout=fopen(out,"wb");
            error_loc(fout==NULL,1,"blk_export_sfld [sarchive.c]",
                      "Unable to open output file");

            for (mu=0;mu<4;mu++)
            {
               lsize[  mu]=(stdint_t)(nl[mu]);
               lsize[4+mu]=(stdint_t)(bs[mu]);
               lsize[8+mu]=(stdint_t)(bo[mu]);
            }

            lsize[12]=eo;

            if (endian==BIG_ENDIAN)
            {
               bswap_int(13,lsize);
               bswap_double(1,&norm);
            }

            iw=fwrite(lsize,sizeof(stdint_t),13,fout);
            iw+=fwrite(&norm,sizeof(double),1,fout);

            error_loc(iw!=14,1,"blk_export_sfld [sarchive.c]",
                      "Incorrect write count");
         }

         iwa=0;
         y[3]=0;

         for (y[0]=0;y[0]<bs[0];y[0]++)
         {
            for (y[1]=0;y[1]<bs[1];y[1]++)
            {
               for (y[2]=0;y[2]<bs[2];y[2]++)
               {
                  for (mu=0;mu<4;mu++)
                     x[mu]=bo[mu]+y[mu];

                  get_line(ip0,eo,x,bs[3]);

                  if (my_rank==ip0)
                  {
                     if (endian==BIG_ENDIAN)
                        bswap_double(bs3*24,vbuf+bo3);
                     iw=fwrite(vbuf+bo3,sizeof(spinor_dble),bs3,fout);
                     iwa|=(iw!=bs3);
                  }
               }
            }
         }

         if (my_rank==ip0)
         {
            error_loc(iwa!=0,1,"blk_export_sfld [sarchive.c]",
                      "Incorrect write count");
            fclose(fout);
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }
}


void blk_import_sfld(char *in,int eo,spinor_dble *sd)
{
   int ip0,nio,ir,ira,ie;
   int nl[4],bs[4],bo[4];
   int mu,vol,bs3,bo3,x[4],y[4];
   int n,i,nb,ib,iprms[1];
   stdint_t lsize[13];
   double norm0,norm1,eps;
   qflt qnrm;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (NPROC>1)
   {
      iprms[0]=eo;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=eo),1,"blk_import_sfld [sarchive.c]",
            "Parameters are not global");
   }

   blk_sizes(in,nl,bs);
   error_root((nl[0]!=N0)||(nl[1]!=N1)||(nl[2]!=N2)||(nl[3]!=N3),1,
              "blk_import_sfld [sarchive.c]","Lattice sizes do not match");

   if (sbuf==NULL)
      alloc_sbuf();

   error(sd==NULL,1,"blk_export_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   sfld=sd;
   ip0=blk_root_process(nl,bs,bo,&nb,&ib);

   if (eo==0)
   {
      vol=VOLUME;
      bs3=bs[3];
      bo3=bo[3];
   }
   else
   {
      vol=VOLUME/2;
      bs3=bs[3]/2;
      bo3=bo[3]/2;
   }

   nio=nio_streams();
   error_root((nio<1)||((nb%nio)!=0),1,"blk_import_sfld [sarchive.c]",
              "Improper number of parallel I/O streams");
   n=nb/nio;
   norm0=0.0;

   for (i=0;i<n;i++)
   {
      if (i==(ib%n))
      {
         if (my_rank==ip0)
         {
            fin=fopen(in,"rb");
            error_loc(fin==NULL,1,"blk_import_sfld [sarchive.c]",
                      "Unable to open input file");

            ir=fread(lsize,sizeof(stdint_t),13,fin);
            ir+=fread(&norm0,sizeof(double),1,fin);
            error_loc(ir!=14,1,"blk_import_sfld [sarchive.c]",
                      "Incorrect read count");

            if (endian==BIG_ENDIAN)
            {
               bswap_int(13,lsize);
               bswap_double(1,&norm0);
            }

            ie=0;

            for (mu=0;mu<4;mu++)
            {
               ie|=(nl[mu]!=(int)(lsize[  mu]));
               ie|=(bs[mu]!=(int)(lsize[4+mu]));
               ie|=(bo[mu]!=(int)(lsize[8+mu]));
            }

            ie|=(eo!=(int)(lsize[12]));

            error_loc(ie!=0,1,"blk_import_sfld [sarchive.c]",
                      "Unexpected file header data");
         }

         ira=0;
         y[3]=0;

         for (y[0]=0;y[0]<bs[0];y[0]++)
         {
            for (y[1]=0;y[1]<bs[1];y[1]++)
            {
               for (y[2]=0;y[2]<bs[2];y[2]++)
               {
                  for (mu=0;mu<4;mu++)
                     x[mu]=bo[mu]+y[mu];

                  if (my_rank==ip0)
                  {
                     ir=fread(vbuf+bo3,sizeof(spinor_dble),bs3,fin);
                     ira|=(ir!=bs3);

                     if (endian==BIG_ENDIAN)
                        bswap_double(bs3*24,vbuf+bo3);
                  }

                  set_line(ip0,eo,x,bs[3]);
               }
            }
         }

         if (my_rank==ip0)
         {
            error_loc(ira!=0,1,"blk_import_sfld [sarchive.c]",
                      "Incorrect read count");
            fclose(fin);
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

   qnrm=norm_square_dble(vol,1,sd);
   norm1=qnrm.q[0];
   eps=64.0*(norm0+norm1)*DBL_EPSILON;
   error((my_rank==ip0)&&(fabs(norm1-norm0)>eps),1,
         "blk_import_sfld [sarchive.c]","Incorrect square norm");
}
