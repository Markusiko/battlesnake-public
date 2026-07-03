.POSIX:
.SUFFIXES:
CC=gcc
CFLAGS=-O3 -Wall -Wextra -Wpedantic -std=c99 -flto

all: bin/move
bin/:; mkdir bin/
clean:; rm -rf bin/

bin/move:  bin/ vendor/jsonw.h vendor/jsonw.c move.c;  $(CC) $(CFLAGS) -o $@ vendor/jsonw.c move.c  -Wno-sign-compare -Wno-parentheses -Wno-unused-value -Wno-missing-field-initializers
