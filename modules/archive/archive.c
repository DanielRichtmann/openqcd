
/*******************************************************************************
*
* File archive.c
*
* Copyright (C) 2005-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write gauge-field configurations.
*
*   void set_nio_streams(int nio)
*     Sets the number nio of simultaneous I/O operations performed by the
*     programs that read or write the parts of the field configurations
*     residing on blocks of lattice points in parallel.
*
*   int nio_streams(void)
*     Returns the number of simultaneous I/O operations last set by the
*     program set_nio_streams() (or zero if that program has not been
*     called before).
*
*   void write_cnfg(char *out)
*     Writes the lattice sizes, the process grid sizes, the coordinates
*     of the calling process, the state of the random number generators,
*     the local plaquette sum and the local double-precision gauge field
*     to the file "out".
*
*   void read_cnfg(char *in)
*     Reads the local double-precision gauge field from the file "in",
*     assuming it was written to the file by the program write_cnfg().
*     The program then resets the random number generator and checks
*     whether the restored field is compatible with the chosen boundary
*     conditions.
*
*   void lat_sizes(char *in,int *nl)
*     Reads the lattice sizes nl[0],..,nl[3] from the configuration file
*     "in" on the MPI process with rank 0, assuming the configuration was
*     written to the file by the program export_cnfg().
*
*   void export_cnfg(char *out)
*     Writes the lattice sizes, the average plaquette and the global
*     double-precision gauge field to the file "out" from process 0.
*     The gauge field variables are written in the "universal" order
*     specified in main/README.io.
*
*   void import_cnfg(char *in,int mask)
*     Reads the global double-precision gauge field from the file "in" on
*     process 0, assuming the field was written by the program export_cnfg().
*     If the lattice sizes recorded on the file are smaller than the current
*     lattice sizes, the field is anti-periodically or periodically extended
*     in direction mu=0,..,3 depending on whether the bit mask&(2^mu) is set
*     or not. The program then checks whether the configuration is compatible
*     with the chosen boundary conditions (see the notes).
*
*  void blk_sizes(char *in,int *nl,int *bs)
*     Reads the lattice and block sizes nl[0],..,nl[3] and bs[0],..,bs[3]
*     from the configuration file "in" on the MPI process with coordinates
*     (0,0,0,0) in the process grid, assuming the configuration was written
*     to the file by the program blk_export_cnfg(), blk_export_mfld() or
*     blk_export_sfld().
*
*   int blk_index(int *nl,int *bs,int *nb)
*     Determines the number nb of blocks of size bs[0],..,bs[3] contained
*     in the block of size nl[0],..,nl[3] with base point (0,0,0,0). These
*     nb blocks are ordered lexicographically according to the position of
*     their base points. If the local lattice is contained in one of them,
*     the program returns the index of that block. Otherwise nb is returned.
*     An error occurs if nl[mu] is not an integer multiple of bs[mu] or if
*     bs[mu] is not a multiple of the local lattice size in direction mu.
*
*   int blk_root_process(int *nl,int *bs,int *bo,int *nb,int *ib)
*     Calls blk_index(nl,bs,nb) and assigns the calculated index to ib.
*     The program then assigns the coordinates of the base point of the
*     block containing the local lattice to bo[0],..,bo[3] and returns the
*     rank of the MPI process whose local lattice contains that point.
*
*   void blk_export_cnfg(int *bs,char *out)
*     Divides the lattice into logical blocks with sizes bs[0],..,bs[3].
*     The MPI processes, where the local lattice contains the base point of
*     a block, then write the lattice sizes, the block sizes, the base
*     point coordinates, the average plaquette and the part of the global
*     double-precision gauge field residing on the block to the file "out".
*     The gauge field variables are written in the "universal" order (see
*     main/README.io).
*
*   void blk_import_cnfg(char *in,int mask)
*     Reads the global double-precision gauge field from the file "in",
*     assuming the field was written by the program blk_export_cnfg().
*     The program reads the block size from the configuration file on the
*     MPI process with coordinates (0,0,0,0) in the process grid. Each MPI
*     process containing the base point of a block then reads the field
*     variables on that block from the configuration file. If needed the
*     field is extended as in the case of the program import_cnfg() and
*     the compatibility of the imported field with the boundary conditions
*     is checked.
*
* See main/README.io for a description of the configuration I/O strategies
* implemented in openQCD and the specification of the "universal" storage
* format.
*
* The programs that perform parallel I/O write the field variables residing
* on B0xB1xB2xB3 blocks of lattice points to different files. When one of
* these functions is executed, the number of parallel I/O streams last set
* by set_nio_streams() must divide the number of blocks. Moreover, the block
* sizes must be multiples of the local lattice sizes. An error occurs if
* any of these conditions is violated.
*
* The average plaquette is calculated by summing the plaquette values over
* all plaquettes in the lattice, including the space-like ones at time N0
* if SF or open-SF boundary conditions are chosen, and dividing the sum by
* 6*N0*N1*N2*N3.
*
* Independently of the machine, the export functions write the data to the
* output file in little-endian byte order. Integers and double-precision
* numbers on the output file occupy 4 and 8 bytes, respectively, the latter
* being formatted according to the IEEE-754 standard. The import function
* assumes the data on the input file to be little endian and converts them
* to big-endian order if the machine is big endian. Exported configurations
* can thus be safely exchanged between different machines.
*
* Periodic and/or anti-periodic extensions of an imported field require the
* following conditions to be satisfied:
*
* - The current lattice sizes N0,..,N3 are multiples (even multiples in
*   the directions with anti-periodic extension) of the lattice sizes
*   nl[0],..,nl[3] read from the configuration file.
*
* - The lattice sizes nl[0],..,nl[3] are multiples of the local lattice
*   sizes L0,..,L3.
*
* - An extension in time is only possible if the current and the previous
*   boundary conditions are periodic.
*
* - Extensions in space are incompatible with SF and open-SF boundary
*   conditions unless all boundary link variables are equal to unity.
*
* Compatibility of a configuration with the chosen boundary conditions is
* established by calling check_bc() [lattice/bcnds.c], with a tolerance on
* the boundary link variables of 64.0*DBL_EPSILON, and by checking that the
* average plaquette coincides with the value read from the configuration
* file. On exit the programs read_cnfg(), import_cnfg() and blk_import_cnfg()
* set the boundary values of the field (if any) to the ones stored in the
* parameter data base so as to guarantee that they are bit-identical to the
* latter.
*
* All programs in this module may involve global communications and must be
* called simultaneously on all processes.
*
*******************************************************************************/

#define ARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3fcts.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static const int nsz[4]={N0,N1,N2,N3};
static const int lsz[4]={L0,L1,L2,L3};
static const int diy[4]={L1*L2*L3,L2*L3,L3,1};

static int my_rank,tag0,tag1,nios=0,nrmx=0;
static int endian,ns,nd,*ranks,*state=NULL;
static su3_dble *ubuf=NULL,*vbuf,*udb;


static void alloc_state(void)
{
   int n;

   ns=rlxs_size();
   nd=rlxd_size();

   if (ns<nd)
      n=nd;
   else
      n=ns;

   state=malloc(n*sizeof(int));
   error(state==NULL,1,"alloc_state [archive.c]",
         "Unable to allocate auxiliary array");
}


void set_nio_streams(int nio)
{
   int iprms[1];

   error_root((nio<1)||(nio>NPROC),1,"set_nio_streams [archive.c]",
            "Parameter nio is out of range");

   if (NPROC>1)
   {
      iprms[0]=nio;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=nio,1,"set_nio_streams [archive.c]",
            "Parameter is not global");
   }

   nios=nio;
}


int nio_streams(void)
{
   return nios;
}


void write_cnfg(char *out)
{
   int n,i,iw,ldat[16];
   double plaq;
   FILE *fout;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (state==NULL)
      alloc_state();

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

   error_root((nios<1)||(NPROC%nios),1,"write_cnfg [archive.c]",
              "Improper number of parallel I/O streams");
   n=NPROC/nios;

   error_root(query_flags(UD_PHASE_SET)==1,1,"write_cnfg [archive.c]",
              "Attempt to write phase-modified gauge field");
   udb=udfld();
   plaq=plaq_sum_dble(0);

   for (i=0;i<n;i++)
   {
      if (i==(my_rank%n))
      {
         fout=fopen(out,"wb");
         error_loc(fout==NULL,1,"write_cnfg [archive.c]",
                   "Unable to open output file");

         iw=fwrite(ldat,sizeof(int),16,fout);
         rlxs_get(state);
         iw+=fwrite(state,sizeof(int),ns,fout);
         rlxd_get(state);
         iw+=fwrite(state,sizeof(int),nd,fout);
         iw+=fwrite(&plaq,sizeof(double),1,fout);
         iw+=fwrite(udb,sizeof(su3_dble),4*VOLUME,fout);

         error_loc(iw!=(17+ns+nd+4*VOLUME),1,"write_cnfg [archive.c]",
                   "Incorrect write count");
         fclose(fout);
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }
}


void read_cnfg(char *in)
{
   int n,i,ir,ie,ldat[16];
   double nplaq,plaq0,plaq1,eps;
   FILE *fin;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (state==NULL)
      alloc_state();

   error_root((nios<1)||(NPROC%nios),1,"read_cnfg [archive.c]",
              "Improper number of parallel I/O streams");
   n=NPROC/nios;
   udb=udfld();
   set_flags(UNSET_UD_PHASE);
   set_bc();
   plaq0=0.0;

   for (i=0;i<n;i++)
   {
      if (i==(my_rank%n))
      {
         fin=fopen(in,"rb");
         error_loc(fin==NULL,1,"read_cnfg [archive.c]",
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
         error_loc(ie!=0,1,"read_cnfg [archive.c]",
                   "Unexpected process grid or lattice data");

         ir+=fread(state,sizeof(int),ns,fin);
         rlxs_reset(state);
         ir+=fread(state,sizeof(int),nd,fin);
         rlxd_reset(state);
         ir+=fread(&plaq0,sizeof(double),1,fin);
         ir+=fread(udb,sizeof(su3_dble),4*VOLUME,fin);

         error_loc(ir!=(17+ns+nd+4*VOLUME),1,"read_cnfg [archive.c]",
                   "Incorrect read count");
         fclose(fin);
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

   set_flags(UPDATED_UD);
   ie=check_bc(64.0*DBL_EPSILON);
   error_root(ie!=1,1,"read_cnfg [archive.c]",
              "Incompatible boundary conditions");

   nplaq=6.0*(double)(VOLUME);
   plaq1=plaq_sum_dble(0);
   eps=sqrt(64.0*nplaq)*DBL_EPSILON;
   eps+=64.0*(fabs(plaq0)+fabs(plaq1))*DBL_EPSILON;
   error(fabs(plaq1-plaq0)>eps,1,"read_cnfg [archive.c]",
         "Incorrect plaquette sum");
   set_bc();
}


static void check_lattice_size(int *nl,int mask)
{
   int mu,bc,ie;
   bc_parms_t bcp;

   ie=0x0;

   for (mu=0;mu<4;mu++)
   {
      if ((nl[mu]<1)||(nsz[mu]%nl[mu]))
         ie|=0x1;
      if (nl[mu]%lsz[mu])
         ie|=0x2;
      if (((mask>>mu)&0x1)&&(nl[mu]!=nsz[mu])&&((nsz[mu]/nl[mu])&0x1))
         ie|=0x4;
   }

   error_root((ie&0x1)!=0x0,1,"check_lattice_size [archive.c]",
              "Previous lattice sizes do not divide the current ones");
   error_root((ie&0x2)!=0x0,1,"check_lattice_size [archive.c]",
              "Previous lattice sizes are not multiples of the local ones");
   error_root((ie&0x4)!=0x0,1,"check_lattice_size [archive.c]",
              "Unsuitable lattice sizes for anti-periodic extension");

   bc=bc_type();

   if (bc!=3)
   {
      ie=(nl[0]<N0);

      if (((bc==1)||(bc==2))&&((nl[1]<N1)||(nl[2]<N2)||(nl[3]<N3)))
      {
         bcp=bc_parms();

         ie|=(bcp.phi[1][0]!=0.0);
         ie|=(bcp.phi[1][1]!=0.0);

         if (bc==1)
         {
            ie|=(bcp.phi[0][0]!=0.0);
            ie|=(bcp.phi[0][1]!=0.0);
         }
      }

      error_root(ie!=0x0,1,"check_lattice_size [archive.c]",
                 "Lattice extension is incompatible with boundary conditions");
   }
}


static void alloc_ubuf(void)
{
   int mu,nu,nv;

   nu=0;
   nv=0;

   for (mu=0;mu<4;mu++)
   {
      if (nu<lsz[mu])
         nu=lsz[mu];
      if (nv<nsz[mu])
         nv=nsz[mu];
   }

   ubuf=amalloc(4*nu*sizeof(*ubuf),ALIGN);
   vbuf=amalloc(4*nv*sizeof(*vbuf),ALIGN);
   error((ubuf==NULL)||(vbuf==NULL),1,"alloc_ubuf [archive.c]",
         "Unable to allocate auxiliary arrays");
   cm3x3_unity(4*nu,ubuf);
   cm3x3_unity(4*nv,vbuf);

   tag0=mpi_permanent_tag();
   tag1=mpi_permanent_tag();
   endian=endianness();
}


static void get_links(int mu,int iy)
{
   int iym,ifc;
   su3_dble *u,*v;

   v=ubuf;
   iym=iy+lsz[mu]*diy[mu];

   if (ipt[iy]<(VOLUME/2))
      iy+=diy[mu];

   for (;iy<iym;iy+=2*diy[mu])
   {
      u=udb+8*(ipt[iy]-(VOLUME/2));

      for (ifc=0;ifc<8;ifc++)
      {
         v[0]=u[0];
         v+=1;
         u+=1;
      }
   }
}


static void set_links(int mu,int iy)
{
   int iym,ifc;
   su3_dble *u,*v;

   v=ubuf;
   iym=iy+lsz[mu]*diy[mu];

   if (ipt[iy]<(VOLUME/2))
      iy+=diy[mu];

   for (;iy<iym;iy+=2*diy[mu])
   {
      u=udb+8*(ipt[iy]-(VOLUME/2));

      for (ifc=0;ifc<8;ifc++)
      {
         u[0]=v[0];
         u+=1;
         v+=1;
      }
   }
}


static void get_line(int ip0,int mu,int *x,int dx)
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

   kmn=np[mu];
   kmx=kmn+(dx/lsz[mu]);
   dmy=1;

   for (k=kmn;k<kmx;k++)
   {
      np[mu]=k;
      ip1=ipr_global(np);

      if (my_rank==ip1)
         get_links(mu,iy);

      if (ip1!=ip0)
      {
         if (my_rank==ip0)
         {
            MPI_Send(&dmy,1,MPI_INT,ip1,tag0,MPI_COMM_WORLD);
            MPI_Recv(vbuf+k*4*lsz[mu],4*lsz[mu]*18,MPI_DOUBLE,ip1,tag1,
                     MPI_COMM_WORLD,&stat);
         }
         else if (my_rank==ip1)
         {
            MPI_Recv(&dmy,1,MPI_INT,ip0,tag0,MPI_COMM_WORLD,&stat);
            MPI_Send(ubuf,4*lsz[mu]*18,MPI_DOUBLE,ip0,tag1,
                     MPI_COMM_WORLD);
         }
      }
      else if (my_rank==ip0)
         cm3x3_assign(4*lsz[mu],ubuf,vbuf+k*4*lsz[mu]);
   }
}


static void set_line(int ip0,int mu,int *x,int dx)
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

   kmn=np[mu];
   kmx=kmn+(dx/lsz[mu]);
   dmy=1;

   for (k=kmn;k<kmx;k++)
   {
      np[mu]=k;
      ip1=ipr_global(np);

      if (ip1!=ip0)
      {
         if (my_rank==ip0)
         {
            MPI_Send(vbuf+k*4*lsz[mu],4*lsz[mu]*18,MPI_DOUBLE,ip1,tag1,
                     MPI_COMM_WORLD);
            MPI_Recv(&dmy,1,MPI_INT,ip1,tag0,MPI_COMM_WORLD,&stat);
         }
         else if (my_rank==ip1)
         {
            MPI_Recv(ubuf,4*lsz[mu]*18,MPI_DOUBLE,ip0,tag1,
                     MPI_COMM_WORLD,&stat);
            MPI_Send(&dmy,1,MPI_INT,ip0,tag0,MPI_COMM_WORLD);
         }
      }
      else if (my_rank==ip1)
         cm3x3_assign(4*lsz[mu],vbuf+k*4*lsz[mu],ubuf);

      if (my_rank==ip1)
         set_links(mu,iy);
   }
}


static void extend1e(int mu,int *nl,su3_dble *ud)
{
   int n,l,ifc;
   su3_dble *vd,*wd;

   n=4*nl[mu];
   wd=ud+n-8;
   vd=ud+2*n-8;

   for (ifc=0;ifc<8;ifc+=2)
   {
      if (ifc==(2*mu))
      {
         vd[ifc]=wd[ifc];
         cm3x3_dagger(wd+ifc,vd+ifc+1);
         cm3x3_dagger(wd+ifc+1,wd+ifc);
      }
      else
      {
         vd[ifc  ]=wd[ifc  ];
         vd[ifc+1]=wd[ifc+1];
      }
   }

   vd-=8;

   for (l=0;l<(n-8);l+=8)
   {
      for (ifc=0;ifc<8;ifc+=2)
      {
         if (ifc==(2*mu))
         {
            cm3x3_dagger(ud+ifc,vd+ifc+1);
            cm3x3_dagger(ud+ifc+1,vd+ifc);
         }
         else
         {
            vd[ifc  ]=ud[ifc  ];
            vd[ifc+1]=ud[ifc+1];
         }
      }

      ud+=8;
      vd-=8;
   }
}


static void extend1o(int mu,int *nl,su3_dble *ud)
{
   int n,l,ifc;
   su3_dble *vd;

   n=4*nl[mu];
   vd=ud+2*n-8;

   for (l=0;l<n;l+=8)
   {
      for (ifc=0;ifc<8;ifc+=2)
      {
         if (ifc==(2*mu))
         {
            cm3x3_dagger(ud+ifc,vd+ifc+1);
            cm3x3_dagger(ud+ifc+1,vd+ifc);
         }
         else
         {
            vd[ifc  ]=ud[ifc  ];
            vd[ifc+1]=ud[ifc+1];
         }
      }

      ud+=8;
      vd-=8;
   }
}


static void aperiodic_extend(int *nl,int mask)
{
   int mu,nu,ip,ip0;
   int np[4],x[4],y[4],ym[4];

   for (mu=0;mu<4;mu++)
   {
      if ((nl[mu]!=nsz[mu])&&((mask>>mu)&0x1))
      {
         ip=0;

         for (nu=0;nu<4;nu++)
         {
            if (nu==mu)
            {
               ip|=((cpr[nu]*lsz[nu])>=(2*nl[nu]));
               np[nu]=0;
               ym[nu]=1;
            }
            else
            {
               ip|=((cpr[nu]*lsz[nu])>=nl[nu]);
               np[nu]=cpr[nu];
               ym[nu]=lsz[nu];
            }
         }

         if (ip==0)
         {
            ip0=ipr_global(np);

            for (y[0]=0;y[0]<ym[0];y[0]++)
            {
               for (y[1]=0;y[1]<ym[1];y[1]++)
               {
                  for (y[2]=0;y[2]<ym[2];y[2]++)
                  {
                     for (y[3]=0;y[3]<ym[3];y[3]++)
                     {
                        for (nu=0;nu<4;nu++)
                        {
                           if (nu==mu)
                              x[nu]=0;
                           else
                              x[nu]=cpr[nu]*lsz[nu]+y[nu];
                        }

                        get_line(ip0,mu,x,nl[mu]);

                        if ((x[0]+x[1]+x[2]+x[3])&0x1)
                        {
                           if (my_rank==ip0)
                              extend1o(mu,nl,vbuf);
                           x[mu]=nl[mu];
                           set_line(ip0,mu,x,nl[mu]);
                        }
                        else
                        {
                           if (my_rank==ip0)
                              extend1e(mu,nl,vbuf);
                           x[mu]=nl[mu]-lsz[mu];
                           set_line(ip0,mu,x,lsz[mu]+nl[mu]);
                        }
                     }
                  }
               }
            }
         }

         nl[mu]*=2;
      }
   }
}


static void alloc_ranks(int n)
{
   if (n>nrmx)
   {
      if (nrmx>0)
         free(ranks);
      ranks=malloc(n*sizeof(*ranks));
      error_loc(ranks==NULL,1,"alloc_ranks [archive.c]",
                "Unable to allocate ranks array");
      nrmx=n;
   }
}


static void periodic_extend(int *nl)
{
   int n,mu,np[4],nr[4],nc[4];
   MPI_Group world,group;
   MPI_Comm comm;

   if ((nl[0]<N0)||(nl[1]<N1)||(nl[2]<N2)||(nl[3]<N3))
   {
      for (mu=0;mu<4;mu++)
      {
         np[mu]=nl[mu]/lsz[mu];
         nr[mu]=cpr[mu]%np[mu];
      }

      alloc_ranks(NPROC/(np[0]*np[1]*np[2]*np[3]));
      n=0;

      for (nc[0]=nr[0];nc[0]<NPROC0;nc[0]+=np[0])
      {
         for (nc[1]=nr[1];nc[1]<NPROC1;nc[1]+=np[1])
         {
            for (nc[2]=nr[2];nc[2]<NPROC2;nc[2]+=np[2])
            {
               for (nc[3]=nr[3];nc[3]<NPROC3;nc[3]+=np[3])
               {
                  ranks[n]=ipr_global(nc);
                  n+=1;
               }
            }
         }
      }

      MPI_Comm_group(MPI_COMM_WORLD,&world);
      MPI_Group_incl(world,n,ranks,&group);
      MPI_Comm_create(MPI_COMM_WORLD,group,&comm);

      MPI_Bcast(udb,4*VOLUME*18,MPI_DOUBLE,0,comm);

      MPI_Comm_free(&comm);
      MPI_Group_free(&group);
      MPI_Group_free(&world);
   }
}


void lat_sizes(char *in,int *nl)
{
   int ir;
   stdint_t lsize[4];
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (ubuf==NULL)
      alloc_ubuf();

   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"lat_sizes [archive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      error_root(ir!=4,1,"lat_sizes [archive.c]",
                 "Incorrect read count");
      fclose(fin);

      if (endian==BIG_ENDIAN)
         bswap_int(4,lsize);

      nl[0]=(int)(lsize[0]);
      nl[1]=(int)(lsize[1]);
      nl[2]=(int)(lsize[2]);
      nl[3]=(int)(lsize[3]);
   }

   MPI_Bcast(nl,4,MPI_INT,0,MPI_COMM_WORLD);
}


void export_cnfg(char *out)
{
   int iw,iwa,x[4];
   stdint_t lsize[4];
   double nplaq,plaq;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   error_root(query_flags(UD_PHASE_SET)==1,1,"export_cnfg [archive.c]",
              "Attempt to export phase-modified gauge field");

   if (ubuf==NULL)
      alloc_ubuf();

   udb=udfld();
   nplaq=(double)(6*N0*N1)*(double)(N2*N3);
   plaq=plaq_sum_dble(1)/nplaq;

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_cnfg [archive.c]",
                 "Unable to open output file");

      lsize[0]=(stdint_t)(N0);
      lsize[1]=(stdint_t)(N1);
      lsize[2]=(stdint_t)(N2);
      lsize[3]=(stdint_t)(N3);

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&plaq);
      }

      iw=fwrite(lsize,sizeof(stdint_t),4,fout);
      iw+=fwrite(&plaq,sizeof(double),1,fout);

      error_root(iw!=5,1,"export_cnfg [archive.c]","Incorrect write count");
   }

   iwa=0;
   x[3]=0;

   for (x[0]=0;x[0]<N0;x[0]++)
   {
      for (x[1]=0;x[1]<N1;x[1]++)
      {
         for (x[2]=0;x[2]<N2;x[2]++)
         {
            get_line(0,3,x,N3);

            if (my_rank==0)
            {
               if (endian==BIG_ENDIAN)
                  bswap_double(4*N3*18,vbuf);

               iw=fwrite(vbuf,sizeof(su3_dble),4*N3,fout);
               iwa|=(iw!=(4*N3));
            }
         }
      }
   }

   if (my_rank==0)
   {
      error_root(iwa!=0,1,"export_cnfg [archive.c]",
                 "Incorrect write count");
      fclose(fout);
   }
}


void import_cnfg(char *in,int mask)
{
   int ir,ira,ie,nl[4],x[4],iprms[1];
   stdint_t lsize[4];
   double nplaq,plaq0,plaq1,eps;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (NPROC>1)
   {
      iprms[0]=mask;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=mask,1,"import_cnfg [archive.c]",
            "Parameters are not global");
   }

   if (ubuf==NULL)
      alloc_ubuf();

   udb=udfld();
   set_flags(UNSET_UD_PHASE);
   set_bc();

   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_cnfg [archive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&plaq0,sizeof(double),1,fin);
      error_root(ir!=5,1,"import_cnfg [archive.c]",
                 "Incorrect read count");

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&plaq0);
      }

      nl[0]=(int)(lsize[0]);
      nl[1]=(int)(lsize[1]);
      nl[2]=(int)(lsize[2]);
      nl[3]=(int)(lsize[3]);

      check_lattice_size(nl,mask);
   }

   MPI_Bcast(nl,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&plaq0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   ira=0;
   x[3]=0;

   for (x[0]=0;x[0]<nl[0];x[0]++)
   {
      for (x[1]=0;x[1]<nl[1];x[1]++)
      {
         for (x[2]=0;x[2]<nl[2];x[2]++)
         {
            if (my_rank==0)
            {
               ir=fread(vbuf,sizeof(su3_dble),4*nl[3],fin);
               ira|=(ir!=(4*nl[3]));

               if (endian==BIG_ENDIAN)
                  bswap_double(4*nl[3]*18,vbuf);
            }

            set_line(0,3,x,nl[3]);
         }
      }
   }

   if (my_rank==0)
   {
      error_root(ira!=0,1,"import_cnfg [archive.c]",
                 "Incorrect read count");
      fclose(fin);
   }

   aperiodic_extend(nl,mask);
   periodic_extend(nl);

   set_flags(UPDATED_UD);
   ie=check_bc(64.0*DBL_EPSILON);
   error_root(ie!=1,1,"import_cnfg [archive.c]",
              "Incompatible boundary conditions");

   nplaq=(double)(6*N0*N1)*(double)(N2*N3);
   plaq0*=nplaq;
   plaq1=plaq_sum_dble(1);
   eps=sqrt(64.0*nplaq)*DBL_EPSILON;
   eps+=64.0*(fabs(plaq0)+fabs(plaq1))*DBL_EPSILON;
   error_root(fabs(plaq1-plaq0)>eps,1,"import_cnfg [archive.c]",
              "Incorrect average plaquette");
   set_bc();
}


void blk_sizes(char *in,int *nl,int *bs)
{
   int mu,ip,ir;
   stdint_t lsize[8];
   FILE *fin;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (mu=0;mu<4;mu++)
      nl[mu]=0;
   ip=ipr_global(nl);

   if (my_rank==ip)
   {
      fin=fopen(in,"rb");
      error_loc(fin==NULL,1,"blk_sizes [archive.c]",
                "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),8,fin);
      error_loc(ir!=8,1,"blk_sizes [archive.c]",
                "Incorrect read count");

      if (endian==BIG_ENDIAN)
         bswap_int(8,lsize);

      for (mu=0;mu<4;mu++)
      {
         nl[mu]=(int)(lsize[  mu]);
         bs[mu]=(int)(lsize[4+mu]);
      }

      fclose(fin);
   }

   MPI_Bcast(nl,4,MPI_INT,ip,MPI_COMM_WORLD);
   MPI_Bcast(bs,4,MPI_INT,ip,MPI_COMM_WORLD);
}


int blk_index(int *nl,int *bs,int *nb)
{
   int mu,ie,n[4],m[4];

   error_loc(iup[0][0]==0,1,"blk_index [archive.c]",
             "Geometry arrays are not set");

   ie=0x0;

   for (mu=0;mu<4;mu++)
   {
      if ((bs[mu]<=0)||(bs[mu]%lsz[mu]))
         ie|=0x1;
      if (nl[mu]%bs[mu])
         ie|=0x2;
   }

   error_loc((ie&0x1)!=0x0,1,"blk_index [archive.c]",
             "Block sizes are not multiples of the local lattice sizes");
   error_loc((ie&0x2)!=0x0,1,"blk_index [archive.c]",
             "Block sizes do not divide the lattice sizes");

   for (mu=0;mu<4;mu++)
   {
      m[mu]=nl[mu]/bs[mu];
      n[mu]=(cpr[mu]*lsz[mu])/bs[mu];
   }

   (*nb)=m[0]*m[1]*m[2]*m[3];

   if ((n[0]<m[0])&&(n[1]<m[1])&&(n[2]<m[2])&&(n[3]<m[3]))
      return n[3]+n[2]*m[3]+n[1]*m[2]*m[3]+n[0]*m[1]*m[2]*m[3];
   else
      return (*nb);
}


int blk_root_process(int *nl,int *bs,int *bo,int *nb,int *ib)
{
   int mu,n[4];

   (*ib)=blk_index(nl,bs,nb);

   for (mu=0;mu<4;mu++)
   {
      bo[mu]=cpr[mu]*lsz[mu];
      bo[mu]-=bo[mu]%bs[mu];
      n[mu]=bo[mu]/lsz[mu];
   }

   return ipr_global(n);
}


void blk_export_cnfg(int *bs,char *out)
{
   int ip0,iw,iwa;
   int nl[4],bo[4];
   int mu,x[4],y[4];
   int n,i,nb,ib,iprms[4];
   stdint_t lsize[12];
   double nplaq,plaq;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (NPROC>1)
   {
      for (mu=0;mu<4;mu++)
         iprms[mu]=bs[mu];

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=bs[0])||(iprms[1]!=bs[1])||(iprms[2]!=bs[2])||
            (iprms[3]!=bs[3]),1,"blk_export_cnfg [archive.c]",
            "Parameters are not global");
   }

   if (ubuf==NULL)
      alloc_ubuf();

   udb=udfld();
   nplaq=(double)(6*N0*N1)*(double)(N2*N3);
   plaq=plaq_sum_dble(1)/nplaq;
   error_root(query_flags(UD_PHASE_SET)==1,1,"blk_export_cnfg [archive.c]",
              "Attempt to export phase-modified gauge field");

   for (mu=0;mu<4;mu++)
      nl[mu]=nsz[mu];
   ip0=blk_root_process(nl,bs,bo,&nb,&ib);
   error_root((nios<1)||((nb%nios)!=0),1,"blk_export_cnfg [archive.c]",
              "Improper number of parallel I/O streams");
   n=nb/nios;

   for (i=0;i<n;i++)
   {
      if (i==(ib%n))
      {
         if (my_rank==ip0)
         {
            fout=fopen(out,"wb");
            error_loc(fout==NULL,1,"blk_export_cnfg [archive.c]",
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
               bswap_double(1,&plaq);
            }

            iw=fwrite(lsize,sizeof(stdint_t),12,fout);
            iw+=fwrite(&plaq,sizeof(double),1,fout);

            error_loc(iw!=13,1,"blk_export_cnfg [archive.c]",
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

                  get_line(ip0,3,x,bs[3]);

                  if (my_rank==ip0)
                  {
                     if (endian==BIG_ENDIAN)
                        bswap_double(4*bs[3]*18,vbuf+4*bo[3]);
                     iw=fwrite(vbuf+4*bo[3],sizeof(su3_dble),4*bs[3],fout);
                     iwa|=(iw!=(4*bs[3]));
                  }
               }
            }
         }

         if (my_rank==ip0)
         {
            error_loc(iwa!=0,1,"blk_export_cnfg [archive.c]",
                      "Incorrect write count");
            fclose(fout);
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }
}


void blk_import_cnfg(char *in,int mask)
{
   int ip0,ir,ira,ie;
   int nl[4],bs[4],bo[4];
   int mu,x[4],y[4];
   int n,i,nb,ib,iprms[1];
   stdint_t lsize[12];
   double nplaq,plaq0,plaq1,eps;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (NPROC>1)
   {
      iprms[0]=mask;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=mask,1,"blk_import_cnfg [archive.c]",
            "Parameters are not global");
   }

   blk_sizes(in,nl,bs);

   if (my_rank==0)
      check_lattice_size(nl,mask);

   if (ubuf==NULL)
      alloc_ubuf();

   udb=udfld();
   set_flags(UNSET_UD_PHASE);
   set_bc();
   ip0=blk_root_process(nl,bs,bo,&nb,&ib);
   error_root((nios<1)||((nb%nios)!=0),1,"blk_import_cnfg [archive.c]",
              "Improper number of parallel I/O streams");
   n=nb/nios;
   plaq0=0.0;

   for (i=0;i<n;i++)
   {
      if ((i==(ib%n))&&(ib<nb))
      {
         if (my_rank==ip0)
         {
            fin=fopen(in,"rb");
            error_loc(fin==NULL,1,"blk_import_cnfg [archive.c]",
                      "Unable to open input file");

            ir=fread(lsize,sizeof(stdint_t),12,fin);
            ir+=fread(&plaq0,sizeof(double),1,fin);
            error_loc(ir!=13,1,"blk_import_cnfg [archive.c]",
                      "Incorrect read count");

            if (endian==BIG_ENDIAN)
            {
               bswap_int(12,lsize);
               bswap_double(1,&plaq0);
            }

            ie=0;

            for (mu=0;mu<4;mu++)
            {
               ie|=(nl[mu]!=(int)(lsize[  mu]));
               ie|=(bs[mu]!=(int)(lsize[4+mu]));
               ie|=(bo[mu]!=(int)(lsize[8+mu]));
            }

            error_loc(ie!=0,1,"blk_import_cnfg [archive.c]",
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
                     ir=fread(vbuf+4*bo[3],sizeof(su3_dble),4*bs[3],fin);
                     ira|=(ir!=(4*bs[3]));

                     if (endian==BIG_ENDIAN)
                        bswap_double(4*bs[3]*18,vbuf+4*bo[3]);
                  }

                  set_line(ip0,3,x,bs[3]);
               }
            }
         }

         if (my_rank==ip0)
         {
            error_loc(ira!=0,1,"blk_import_cnfg [archive.c]",
                      "Incorrect read count");
            fclose(fin);
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

   aperiodic_extend(nl,mask);
   periodic_extend(nl);

   set_flags(UPDATED_UD);
   ie=check_bc(64.0*DBL_EPSILON);
   error_root(ie!=1,1,"blk_import_cnfg [archive.c]",
              "Incompatible boundary conditions");

   nplaq=(double)(6*N0*N1)*(double)(N2*N3);
   plaq0*=nplaq;
   plaq1=plaq_sum_dble(1);
   eps=sqrt(64.0*nplaq)*DBL_EPSILON;
   eps+=64.0*(fabs(plaq0)+fabs(plaq1))*DBL_EPSILON;
   error((ib<nb)&&(my_rank==ip0)&&(fabs(plaq1-plaq0)>eps),1,
         "blk_import_cnfg [archive.c]","Incorrect average plaquette");
   set_bc();
}
