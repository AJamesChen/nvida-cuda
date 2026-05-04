## Output
```bash
(base) jamesc@jamesc-MS-7E13:~/skills/nvida-cuda/gpcc-ex4-3$ make
nvcc  main.cu -o vector_operations
(base) jamesc@jamesc-MS-7E13:~/skills/nvida-cuda/gpcc-ex4-3$ ./vector_operations 
input1: 12.000 9.000 -4.000 7.500 0.000 25.000
input2: 5.000 3.000 2.000 -2.500 4.000 6.000
----------------------------
multiply: 60.000 27.000 -8.000 -18.750 0.000 150.000
divide: 2.400 3.000 -2.000 -3.000 0.000 4.167
absolute difference: 7.000 6.000 6.000 10.000 4.000 19.000
maximum: 12.000 9.000 2.000 7.500 4.000 25.000
minimum: 5.000 3.000 -4.000 -2.500 0.000 6.000
modulus: 2.000 0.000 -0.000 0.000 0.000 1.000
```