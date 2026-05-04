NVCC ?= nvcc
PYTHON ?= python
NVCCFLAGS ?=

TARGET := test_prime
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
