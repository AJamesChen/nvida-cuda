NVCC ?= nvcc
PYTHON ?= python
GCC12 := $(shell command -v gcc-12 2>/dev/null)
NVCCFLAGS ?= $(if $(GCC12),-ccbin $(GCC12),)

TARGET := vector_operations
SOURCE := main.cu
TEST_DIR := tests

.PHONY: all test run clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $< -o $@

test:
	$(PYTHON) -m unittest discover -s $(TEST_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)
