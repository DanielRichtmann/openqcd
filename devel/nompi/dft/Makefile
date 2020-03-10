################################################################################
#
# Makefile to compile and link C programs
#
# Version valid for Linux machines
#
# "make" compiles and links the specified main programs and modules
# using the specified libraries (if any), and produces the executables
#
# "make clean" removes all files created by "make"
#
# Dependencies on included files are automatically taken care of
#
################################################################################

all: rmxeq mkdep mkxeq
.PHONY: all


# main programs and required modules

MAIN = check1 check2 check3

DFT = dft_shuf small_dft

RANDOM = ranlxd ranlxs gauss

UTILS = endian mutils qsum utils

MODULES = $(DFT) $(RANDOM) $(UTILS)


# search path for modules

MDIR = ../../../modules
NDIR = $(MDIR)/nompi

VPATH = .:$(MDIR)/dft:$(MDIR)/random:$(NDIR)/utils:$(MDIR)/utils:


# additional include directories

INCPATH = ../../../include/nompi ../../../include


# additional libraries to be included

LIBS = m gmp

LIBPATH =


# scheduling and optimization options

CFLAGS = -std=c89 -pedantic -fstrict-aliasing -Wall \
         -Wno-format-overflow -Wstrict-prototypes -Werror \
         -O -mfpmath=sse -mno-avx -DAVX -DFMA3 -DPM


############################## do not change ###################################

SHELL=/bin/bash

CC=$(GCC)

PGMS= $(MAIN) $(MODULES)

INCDIRS = $(addprefix -I,$(INCPATH))

OBJECTS = $(addsuffix .o,$(MODULES))

LDFLAGS = $(addprefix -L,$(LIBPATH)) $(addprefix -l,$(LIBS))

-include $(addsuffix .d,$(PGMS))


# rule to make dependencies

$(addsuffix .d,$(PGMS)): %.d: %.c Makefile
	@ $(CC) -MM -ansi $(INCDIRS) $< -o $@


# rule to compile source programs

$(addsuffix .o,$(PGMS)): %.o: %.c Makefile
	$(CC) $< -c $(CFLAGS) $(INCDIRS) -o $@


# rule to link object files

$(MAIN): %: %.o $(OBJECTS) Makefile
	$(CC) $< $(OBJECTS) $(CFLAGS) $(LDFLAGS) -o $@


# produce executables

mkxeq: $(MAIN)


# remove old executables and old error log file

rmxeq:
	@ -rm -f $(MAIN); \
        echo "delete old executables"


# make dependencies

mkdep:  $(addsuffix .d,$(PGMS))
	@ echo "generate tables of dependencies"


# clean directory

clean:
	@ -rm -rf *.d *.o .tmp $(MAIN)
.PHONY: clean

################################################################################
