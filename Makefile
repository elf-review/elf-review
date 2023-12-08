# Makefile

# Compiler settings
CXX = g++
CXXFLAGS = -O3 -std=c++17 -pthread

# Target executable name
TARGET = elf_pthread

# Source files
SOURCES = elf_pthread.cpp

# Rule to create the executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)

