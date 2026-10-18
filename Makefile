
#TARGETS := q1 q2 q3
TARGETS := q1 q2 q3 q4

all: $(TARGETS)

q1: hw3-v1.cpp utils.h
	nvcc -arch=native -x cu hw3-v1.cpp -o $@

q2: hw3-v2.cpp utils.h
	nvcc -arch=native -x cu hw3-v2.cpp -o $@
#
q3: hw3-v3.cpp utils.h
	nvcc -arch=native -x cu hw3-v3.cpp -o $@

q4: hw3-v4.cpp utils.h
	nvcc -arch=native -x cu hw3-v4.cpp -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean