# ----------------------------------------------------------------
# environment
CC		= mpifccpx
FC		= 

# ----------------------------------------------------------------
# options

CFLAGS		= -Kfast -Koptmsg=2
FFLAGS		= 

# ----------------------------------------------------------------
# sources and objects

C_SRC		= src/main.c src/solver.c src/matrix.c src/vector.c src/mmio.c
F_SRC		= 

C_OBJ		= $(C_SRC:.c=.o)
F_OBJ		= $(F_SRC:.f=.o)

# ----------------------------------------------------------------
# executables

EXEC		= solver

all:		$(EXEC)

$(EXEC):	$(C_OBJ)
	$(CC) -o $@ $(CFLAGS) $(C_OBJ) -lm

# ----------------------------------------------------------------
# rules

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.f.o:
	$(FC) $(FFLAGS) -c $< -o $@

# ----------------------------------------------------------------
# clean up

clean:
	/bin/rm -f $(EXEC) $(C_OBJ) $(F_OBJ)

# ----------------------------------------------------------------
# End of Makefile
