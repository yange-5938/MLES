CC = gcc
CFLAGS = -lm
SRCS = test_main.c ../src/simple_neural_networks.c
TARGET = test

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(SRCS) -o $(TARGET) $(CFLAGS) -lcheck -lpthread -lsubunit

.PHONY: clean

clean:
	rm -f $(TARGET)
