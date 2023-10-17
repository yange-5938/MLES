CC = gcc
CFLAGS = -lm
SRCS = src/main.c src/simple_neural_networks.c
TARGET = output

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(SRCS) -o $(TARGET) $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(TARGET)
