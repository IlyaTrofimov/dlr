all: dlr test_allreduce

test_allreduce: accumulate.cc accumulate.h allreduce.cc allreduce.h test_allreduce.cpp
	g++ test_allreduce.cpp accumulate.cc allreduce.cc -o test_allreduce -O0 -g

dlr: dlr.cpp accumulate.cc allreduce.cc asa047.c
	g++ -std=c++0x dlr.cpp accumulate.cc allreduce.cc asa047.c -o dlr -g -O3 -l boost_program_options
