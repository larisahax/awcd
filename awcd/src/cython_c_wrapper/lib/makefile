CC = gcc

default: libsparsematrixdot.a

libsparsematrixdot.a: sparsematrixdot.o
	ar rcs $@ $^
    
sparsematrixdot.o: sparsematrixdot.c sparsematrixdot.h
	$(CC) -c $<

clean:
	rm *.o *.a
