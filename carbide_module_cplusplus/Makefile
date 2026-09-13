CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
LDFLAGS  := -pthread

.PHONY: all run verify clean

all: carbide

carbide: main.cpp *.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) main.cpp -o carbide

run: carbide
	./carbide

# builds and runs both verification harnesses
verify: gradcheck_test model_smoke_test
	./gradcheck_test
	./model_smoke_test

gradcheck_test: gradcheck_test.cpp *.hpp
	$(CXX) $(CXXFLAGS) gradcheck_test.cpp -o gradcheck_test

model_smoke_test: model_smoke_test.cpp *.hpp
	$(CXX) $(CXXFLAGS) model_smoke_test.cpp -o model_smoke_test

clean:
	rm -f carbide gradcheck_test model_smoke_test
