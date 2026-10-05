CXX = clang++
CXXFLAGS = -std=c++23 -Wall -Wextra -g -O0
TARGET = out
SRCS = main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET) $(TARGET).dSYM

.PHONY: all run clean
