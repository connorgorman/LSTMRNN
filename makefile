CC = g++
DEBUG = -g
CFLAGS = -std=c++11
COPT = -O3
OBJS = main.cpp

aout : main.cpp
	$(CC) $(CFLAGS) $(COPT) main.cpp -o run

test: main.cpp
	$(CC)   -Wl,--start-group ${MKLROOT}/lib/mic/libmkl_intel_lp64.a ${MKLROOT}/lib/mic/libmkl_core.a ${MKLROOT}/lib/mic/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm $(CFLAGS) main.cpp -o run1

clean:
	rm run
