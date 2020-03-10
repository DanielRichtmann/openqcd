
/*******************************************************************************
*
* File marchive.c
*
* Copyright (C) 2005-2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write momentum-field configurations.
*
*   void write_mfld(char *out)
*     Writes the lattice sizes, the process grid sizes, the coordinates
*     of the calling process, the local momentum action and the local part
*     of the momentum field to the file "out".
*
*   void read_mfld(char *in)
*     Reads the local part of the momentum field from the file "in",
*     assuming the field was written to the file by write_mfld(). The
*     program then checks that the restored field is compatible with
*     the chosen boundary conditions.
*
*   void export_mfld(char *out)
*     Writes the lattice sizes, the global momentum action and the global
*     momentum field to the file "out" from process 0. The field variables
*     are written in the "universal" order specified in main/README.io.
*
*   void import_mfld(char *in)
*     Reads the momentum field from the file "in" on process 0, assuming
*     the field was written by the program export_mfld(). The program then
*     checks that the restored field is compatible with the chosen boundary
*     conditions.
*
*   void blk_export_mfld(int *bs,char *out)
*     Divides the lattice into logical blocks with sizes bs[0],..,bs[3].
*     The MPI processes, where the local lattice contains the base point of
*     a block, then write the lattice sizes, the block sizes, the base point
*     coordinates, the global momentum action and the part of the momentum
*     field residing on the block to the file "out". The field variables are
*     written in the "universal" order (see main/README.io).
*
*   void blk_import_mfld(char *in)
*     Reads the momentum field from the file "in", assuming the field was
*     written by the program blk_export_mfld(). The program reads the block
*     size from the configuration file on the MPI process with coordinates
*     (0,0,0,0) in the process grid. Each MPI process containing the base
*     point of a block then reads the field variables on that block from
*     the configuration file. The compatibility of the imported field with
*     the boundary conditions is checked.
*
* The momentum field is the one that can be accessed through the program
* mdflds() [mdflds/mdflds.c]. See main/README.io for a description of the
* configuration I/O strategies implemented in openQCD and the specification
* of the "universal" storage format. The momentum action is computed by
* calling the program momentum_action() [mdflds/mdflds.c].
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
* Compatibility of the momentum field with the chosen boundary conditions is
* established by checking that the field components on the inactive links
* vanish. In addition, the current value of the momentum action is compared
* with the value read from the data file.
*
* All programs in this module involve communications and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define MARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "linalg.h"
#include "mdflds.h"
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
static su3_alg_dble *mom,*mbuf=NULL,*vbuf;


static int is_zero(su3_alg_dble *X)
{
   int it;

   it=1;
   it&=((*X).c1==0.0);
   it&=((*X).c2==0.0);
   it&=((*X).c3==0.0);
   it&=((*X).c4==0.0);
   it&=((*X).c5==0.0);
   it&=((*X).c6==0.0);
   it&=((*X).c7==0.0);
   it&=((*X).c8==0.0);

   return it;
}


static int check_bcmom(void)
{
   int bc,ie;
   int nlks,*lks,*lkm;
   int npts,*pts,*ptm;
   su3_alg_dble *m,*mm;
   mdflds_t *mdfs;

   bc=bc_type();
   mdfs=mdflds();
   mom=(*mdfs).mom;
   ie=1;

   if (bc==0)
   {
      lks=bnd_lks(&nlks);
      lkm=lks+nlks;

      for (;lks<lkm;lks++)
         ie&=is_zero(mom+(*lks));
   }
   else if (bc==1)
   {
      pts=bnd_pts(&npts);
      ptm=pts+npts;
      pts+=(npts/2);

      for (;pts<ptm;pts++)
      {
         m=mom+8*((*pts)-(VOLUME/2))+2;
         mm=m+6;

         for (;m<mm;m++)
            ie&=is_zero(m);
      }
   }

   return ie;
}


void write_mfld(char *out)
{
   int nio,n,i,iw,ldat[16];
   double act;
   qflt qact;
   mdflds_t *mdfs;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

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

   nio=nio_streams();
   error_root((nio<1)||(NPROC%nio),1,"write_mfld [marchive.c]",
              "Improper number of parallel I/O streams");
   n=NPROC/nio;
   mdfs=mdflds();
   mom=(*mdfs).mom;
   qact=momentum_action(0);
   act=qact.q[0];

   for (i=0;i<n;i++)
   {
      if (i==(my_rank%n))
      {
         fout=fopen(out,"wb");
         error_loc(fout==NULL,1,"write_mfld [marchive.c]",
                   "Unable to open output file");

         iw=fwrite(ldat,sizeof(int),16,fout);
         iw+=fwrite(&act,sizeof(double),1,fout);
         iw+=fwrite(mom,sizeof(su3_alg_dble),4*VOLUME,fout);

         error_loc(iw!=(17+4*VOLUME),1,"write_mfld [marchive.c]",
                   "Incorrect write count");
         fclose(fout);
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }
}


void read_mfld(char *in)
{
   int nio,n,i,ir,ie,ldat[16];
   double act0,act1,eps;
   qflt qact;
   mdflds_t *mdfs;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   nio=nio_streams();
   error_root((nio<1)||(NPROC%nio),1,"read_mfld [marchive.c]",
              "Improper number of parallel I/O streams");
   n=NPROC/nio;
   mdfs=mdflds();
   mom=(*mdfs).mom;
   act0=0.0;

   for (i=0;i<n;i++)
   {
      if (i==(my_rank%n))
      {
         fin=fopen(in,"rb");
         error_loc(fin==NULL,1,"read_mfld [marchive.c]",
                   "Unable to open input file");

         ir=fread(ldat,sizeof(int),16,fin);

         ie=0;
         ie|=((ldat[0]!=NPROC0)||(ldat[1]!=NPROC1)||
              (ldat[2]!=NPROC2)||(ldat[3]!=NPROC3));
         ie|=((ldat[4]!=L0)||(ldat[5]!=L1)||
              (ldat[6]!=L2)||(ldat[7]!=L3));
         ie|=((ldat[8]!=NPROC0_BLK)||(ldat[9]!=NPROC1_BLK)||
              (ldat[10]!=NPROC2_BLK)||(ldat[11]!=NPROC3_BLK));
         ie|=((ldat[12]!=cpr[0])||(ldat[13]!=cpr[1])||
              (ldat[14]!=cpr[2])||(ldat[15]!=cpr[3]));
         error_loc(ie!=0,1,"read_mfld [marchive.c]","Unexpected lattice data");

         ir+=fread(&act0,sizeof(double),1,fin);
         ir+=fread(mom,sizeof(su3_alg_dble),4*VOLUME,fin);
         error_loc(ir!=(17+4*VOLUME),1,"read_mfld [marchive.c]",
                   "Incorrect read count");

         fclose(fin);
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

   error(check_bcmom()!=1,1,"write_mfld [marchive.c]",
         "Unexpected boundary values");
   qact=momentum_action(0);
   act1=qact.q[0];
   eps=64.0*(act0+act1)*DBL_EPSILON;
   error(fabs(act1-act0)>eps,1,"read_mfld [marchive.c]",
         "Incorrect momentum action");
}


static void alloc_mbuf(void)
{
   mbuf=amalloc(4*(L3+N3)*sizeof(su3_alg_dble),ALIGN);
   vbuf=mbuf+4*L3;
   error(mbuf==NULL,1,"alloc_mbuf [marchive.c]",
         "Unable to allocate auxiliary array");
   set_alg2zero(4*(L3+N3),mbuf);

   tag0=mpi_permanent_tag();
   tag1=mpi_permanent_tag();
   endian=endianness();
}


static void get_links(int iy)
{
   int iym,ifc;
   su3_alg_dble *m,*mb;

   mb=mbuf;
   iym=iy+L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (;iy<iym;iy+=2)
   {
      m=mom+8*(ipt[iy]-(VOLUME/2));

      for (ifc=0;ifc<8;ifc++)
      {
         mb[0]=m[0];
         mb+=1;
         m+=1;
      }
   }
}


static void set_links(int iy)
{
   int iym,ifc;
   su3_alg_dble *m,*mb;

   mb=mbuf;
   iym=iy+L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (;iy<iym;iy+=2)
   {
      m=mom+8*(ipt[iy]-(VOLUME/2));

      for (ifc=0;ifc<8;ifc++)
      {
         m[0]=mb[0];
         m+=1;
         mb+=1;
      }
   }
}


static void get_line(int ip0,int *x,int dx)
{
   int nu,iy,np[4];
   int k,kmn,kmx,dmy,ip1;
   MPI_Status stat;

   iy=0;

   for (nu=0;nu<4;nu++)
   {
      iy+=(x[nu]%lsz[nu])*diy[nu];
      np[nu]=x[nu]/lsz[nu];
   }

   kmn=np[3];
   kmx=kmn+(dx/L3);
   dmy=1;

   for (k=kmn;k<kmx;k++)
   {
      np[3]=k;
      ip1=ipr_global(np);

      if (my_rank==ip1)
         get_links(iy);

      if (ip1!=ip0)
      {
         if (my_rank==ip0)
         {
            MPI_Send(&dmy,1,MPI_INT,ip1,tag0,MPI_COMM_WORLD);
            MPI_Recv(vbuf+k*4*L3,4*L3*8,MPI_DOUBLE,ip1,tag1,MPI_COMM_WORLD,
                     &stat);
         }
         else if (my_rank==ip1)
         {
            MPI_Recv(&dmy,1,MPI_INT,ip0,tag0,MPI_COMM_WORLD,&stat);
            MPI_Send(mbuf,4*L3*8,MPI_DOUBLE,ip0,tag1,MPI_COMM_WORLD);
         }
      }
      else if (my_rank==ip0)
         assign_alg2alg(4*L3,mbuf,vbuf+k*4*L3);
   }
}


static void set_line(int ip0,int *x,int dx)
{
   int nu,iy,np[4];
   int k,kmn,kmx,dmy,ip1;
   MPI_Status stat;

   iy=0;

   for (nu=0;nu<4;nu++)
   {
      iy+=(x[nu]%lsz[nu])*diy[nu];
      np[nu]=x[nu]/lsz[nu];
   }

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
            MPI_Send(vbuf+k*4*L3,4*L3*8,MPI_DOUBLE,ip1,tag1,MPI_COMM_WORLD);
            MPI_Recv(&dmy,1,MPI_INT,ip1,tag0,MPI_COMM_WORLD,&stat);
         }
         else if (my_rank==ip1)
         {
            MPI_Recv(mbuf,4*L3*8,MPI_DOUBLE,ip0,tag1,MPI_COMM_WORLD,&stat);
            MPI_Send(&dmy,1,MPI_INT,ip0,tag0,MPI_COMM_WORLD);
         }
      }
      else if (my_rank==ip1)
         assign_alg2alg(4*L3,vbuf+k*4*L3,mbuf);

      if (my_rank==ip1)
         set_links(iy);
   }
}


void export_mfld(char *out)
{
   int iw,iwa,x[4];
   stdint_t lsize[4];
   double act;
   qflt qact;
   mdflds_t *mdfs;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (mbuf==NULL)
      alloc_mbuf();

   mdfs=mdflds();
   mom=(*mdfs).mom;
   qact=momentum_action(1);
   act=qact.q[0];

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_mfld [marchive.c]",
                 "Unable to open output file");

      lsize[0]=N0;
      lsize[1]=N1;
      lsize[2]=N2;
      lsize[3]=N3;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&act);
      }

      iw=fwrite(lsize,sizeof(stdint_t),4,fout);
      iw+=fwrite(&act,sizeof(double),1,fout);
      error_root(iw!=5,1,"export_mfld [marchive.c]","Incorrect write count");
   }

   iwa=0;
   x[3]=0;

   for (x[0]=0;x[0]<N0;x[0]++)
   {
      for (x[1]=0;x[1]<N1;x[1]++)
      {
         for (x[2]=0;x[2]<N2;x[2]++)
         {
            get_line(0,x,N3);

            if (my_rank==0)
            {
               if (endian==BIG_ENDIAN)
                  bswap_double(4*N3*8,vbuf);

               iw=fwrite(vbuf,sizeof(su3_alg_dble),4*N3,fout);
               iwa|=(iw!=(4*N3));
            }
         }
      }
   }

   if (my_rank==0)
   {
      error_root(iwa!=0,1,"export_mfld [marchive.c]","Incorrect write count");
      fclose(fout);
   }
}


void import_mfld(char *in)
{
   int ir,ira,x[4];
   stdint_t lsize[4];
   double act0,act1,eps;
   qflt qact;
   mdflds_t *mdfs;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (mbuf==NULL)
      alloc_mbuf();

   mdfs=mdflds();
   mom=(*mdfs).mom;

   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_mfld [marchive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&act0,sizeof(double),1,fin);
      error_root(ir!=5,1,"import_mfld [marchive.c]","Incorrect read count");

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&act0);
      }

      error_root((lsize[0]!=N0)||(lsize[1]!=N1)||(lsize[2]!=N2)||
                 (lsize[3]!=N3),1,"import_mfld [marchive.c]",
                 "Lattice sizes do not match");
   }

   MPI_Bcast(&act0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
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
               ir=fread(vbuf,sizeof(su3_alg_dble),4*N3,fin);
               ira|=(ir!=(4*N3));

               if (endian==BIG_ENDIAN)
                  bswap_double(4*N3*8,vbuf);
            }

            set_line(0,x,N3);
         }
      }
   }

   if (my_rank==0)
   {
      error_root(ira!=0,1,"import_mfld [marchive.c]","Incorrect read count");
      fclose(fin);
   }

   error(check_bcmom()!=1,1,"import_mfld [marchive.c]",
         "Unexpected boundary values");
   qact=momentum_action(1);
   act1=qact.q[0];
   eps=64.0*(act0+act1)*DBL_EPSILON;
   error_root(fabs(act1-act0)>eps,1,"import_mfld [marchive.c]",
              "Incorrect momentum action");
}


void blk_export_mfld(int *bs,char *out)
{
   int ip0,nio,iw,iwa;
   int nl[4],bo[4];
   int mu,x[4],y[4];
   int n,i,nb,ib,iprms[4];
   stdint_t lsize[12];
   double act;
   qflt qact;
   mdflds_t *mdfs;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (NPROC>1)
   {
      for (mu=0;mu<4;mu++)
         iprms[mu]=bs[mu];

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=bs[0])||(iprms[1]!=bs[1])||(iprms[2]!=bs[2])||
            (iprms[3]!=bs[3]),1,"blk_export_mfld [marchive.c]",
            "Parameters are not global");
   }

   if (mbuf==NULL)
      alloc_mbuf();

   mdfs=mdflds();
   mom=(*mdfs).mom;
   qact=momentum_action(1);
   act=qact.q[0];
   nio=nio_streams();

   for (mu=0;mu<4;mu++)
      nl[mu]=nsz[mu];
   ip0=blk_root_process(nl,bs,bo,&nb,&ib);
   error_root((nio<1)||((nb%nio)!=0),1,"blk_export_mfld [marchive.c]",
              "Improper number of parallel I/O streams");
   n=nb/nio;

   for (i=0;i<n;i++)
   {
      if (i==(ib%n))
      {
         if (my_rank==ip0)
         {
            fout=fopen(out,"wb");
            error_loc(fout==NULL,1,"blk_export_mfld [marchive.c]",
                      "Unable to open output file");

            for (mu=0;mu<4;mu++)
            {
               lsize[  mu]=(stdint_t)(nl[mu]);
               lsize[4+mu]=(stdint_t)(bs[mu]);
               lsize[8+mu]=(stdint_t)(bo[mu]);
            }

            if (endian==BIG_ENDIAN)
            {
               bswap_int(12,lsize);
               bswap_double(1,&act);
            }

            iw=fwrite(lsize,sizeof(stdint_t),12,fout);
            iw+=fwrite(&act,sizeof(double),1,fout);

            error_loc(iw!=13,1,"blk_export_mfld [marchive.c]",
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

                  get_line(ip0,x,bs[3]);

                  if (my_rank==ip0)
                  {
                     if (endian==BIG_ENDIAN)
                        bswap_double(4*bs[3]*8,vbuf+4*bo[3]);
                     iw=fwrite(vbuf+4*bo[3],sizeof(su3_alg_dble),4*bs[3],fout);
                     iwa|=(iw!=(4*bs[3]));
                  }
               }
            }
         }

         if (my_rank==ip0)
         {
            error_loc(iwa!=0,1,"blk_export_mfld [marchive.c]",
                      "Incorrect write count");
            fclose(fout);
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }
}


void blk_import_mfld(char *in)
{
   int ip0,nio,ir,ira,ie;
   int nl[4],bs[4],bo[4];
   int mu,x[4],y[4];
   int n,i,nb,ib;
   stdint_t lsize[12];
   double act0,act1,eps;
   qflt qact;
   mdflds_t *mdfs;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   blk_sizes(in,nl,bs);
   error_root((nl[0]!=N0)||(nl[1]!=N1)||(nl[2]!=N2)||
              (nl[3]!=N3),1,"blk_import_mfld [marchive.c]",
              "Lattice sizes do not match");

   if (mbuf==NULL)
      alloc_mbuf();

   mdfs=mdflds();
   mom=(*mdfs).mom;
   nio=nio_streams();
   ip0=blk_root_process(nl,bs,bo,&nb,&ib);
   error_root((nio<1)||((nb%nio)!=0),1,"blk_import_mfld [marchive.c]",
              "Improper number of parallel I/O streams");
   n=nb/nio;
   act0=0.0;

   for (i=0;i<n;i++)
   {
      if (i==(ib%n))
      {
         if (my_rank==ip0)
         {
            fin=fopen(in,"rb");
            error_loc(fin==NULL,1,"blk_import_mfld [marchive.c]",
                      "Unable to open input file");

            ir=fread(lsize,sizeof(stdint_t),12,fin);
            ir+=fread(&act0,sizeof(double),1,fin);
            error_loc(ir!=13,1,"blk_import_mfld [marchive.c]",
                      "Incorrect read count");

            if (endian==BIG_ENDIAN)
            {
               bswap_int(12,lsize);
               bswap_double(1,&act0);
            }

            ie=0;

            for (mu=0;mu<4;mu++)
            {
               ie|=(nl[mu]!=(int)(lsize[  mu]));
               ie|=(bs[mu]!=(int)(lsize[4+mu]));
               ie|=(bo[mu]!=(int)(lsize[8+mu]));
            }

            error_loc(ie!=0,1,"blk_import_mfld [marchive.c]",
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
                     ir=fread(vbuf+4*bo[3],sizeof(su3_alg_dble),4*bs[3],fin);
                     ira|=(ir!=(4*bs[3]));

                     if (endian==BIG_ENDIAN)
                        bswap_double(4*bs[3]*8,vbuf+4*bo[3]);
                  }

                  set_line(ip0,x,bs[3]);
               }
            }
         }

         if (my_rank==ip0)
         {
            error_loc(ira!=0,1,"blk_import_mfld [marchive.c]",
                      "Incorrect read count");
            fclose(fin);
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

   error(check_bcmom()!=1,1,"blk_import_mfld [marchive.c]",
         "Unexpected boundary values");
   qact=momentum_action(1);
   act1=qact.q[0];
   eps=64.0*(act0+act1)*DBL_EPSILON;
   error((my_rank==ip0)&&(fabs(act1-act0)>eps),1,
         "blk_import_mfld [marchive.c]","Incorrect momentum action");
}
