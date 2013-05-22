# Copyright (c) 2013 The University of Edinburgh.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for OpenACC Benchmarks

ifeq ($(COMP),pgi)
CC = pgcc
CFLAGS = -acc -Minfo=accel -mcmodel=medium
LFLAGS = 
else ifeq ($(COMP),craypgi)
CC = cc
CFLAGS = -acc -Minfo=accel -mcmodel=medium
LFLAGS = 
else ifeq ($(COMP),hmpp)
CC = hmpp --hdpp-off --color gcc
CFLAGS = 
LFLAGS = -fopenmp -lm
else ifeq ($(COMP),cray)
CC = cc
CFLAGS = -hlist=a -h acc_model=auto_async_none
LFLAGS = 
endif

objects = common.o main.o level0.o level1.o 27stencil.o le_core.o himeno.o

default: oa

%.o : %.c
	$(CC) $(CFLAGS) -c $<

oa : $(objects)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

.PHONY: clean

clean:
	rm -f *.o oa __hmpp* *.lst *.cub *.ptx