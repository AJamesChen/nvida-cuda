import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "main.cu"


class VectorOperationsCudaProgramTest(unittest.TestCase):
    def test_kernels_calculate_element_wise_vector_results(self):
        executable = self._build_test_executable()

        result = subprocess.run(
            [str(executable)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        if result.returncode == 77:
            self.skipTest(result.stderr.strip())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def _build_test_executable(self):
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")

        nvcc_command = [nvcc]
        gcc_12 = shutil.which("gcc-12")
        if gcc_12 is not None:
            nvcc_command.extend(["-ccbin", gcc_12])

        build_dir = tempfile.TemporaryDirectory()
        self.addCleanup(build_dir.cleanup)
        test_source = Path(build_dir.name) / "test_vector_operations.cu"
        test_source.write_text(
            f"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNIT_TEST
#include "{SOURCE}"

#define CUDA_CHECK(call) do {{ \\
    cudaError_t err = (call); \\
    if (err != cudaSuccess) {{ \\
        fprintf(stderr, "%s failed: %s\\n", #call, cudaGetErrorString(err)); \\
        return EXIT_FAILURE; \\
    }} \\
}} while (0)

static int check_values(const char *label, const float *actual, const float *expected, int count) {{
    for (int i = 0; i < count; ++i) {{
        if (fabsf(actual[i] - expected[i]) > 0.00001f) {{
            fprintf(stderr, "%s[%d] was %f, expected %f\\n", label, i, actual[i], expected[i]);
            return EXIT_FAILURE;
        }}
    }}

    return EXIT_SUCCESS;
}}

int main() {{
    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {{
        fprintf(stderr, "no CUDA-capable device is available\\n");
        return 77;
    }}

    const int count = 6;
    const float input_1[count] = {{12.0f, 9.0f, -4.0f, 7.5f, 0.0f, 25.0f}};
    const float input_2[count] = {{5.0f, 3.0f, 2.0f, -2.5f, 4.0f, 6.0f}};
    const float expected_multiply[count] = {{60.0f, 27.0f, -8.0f, -18.75f, 0.0f, 150.0f}};
    const float expected_divide[count] = {{2.4f, 3.0f, -2.0f, -3.0f, 0.0f, 4.1666665f}};
    const float expected_abs_diff[count] = {{7.0f, 6.0f, 6.0f, 10.0f, 4.0f, 19.0f}};
    const float expected_maximum[count] = {{12.0f, 9.0f, 2.0f, 7.5f, 4.0f, 25.0f}};
    const float expected_minimum[count] = {{5.0f, 3.0f, -4.0f, -2.5f, 0.0f, 6.0f}};
    const float expected_modulus[count] = {{2.0f, 0.0f, -0.0f, 0.0f, 0.0f, 1.0f}};
    float multiply[count] = {{0.0f}};
    float divide[count] = {{0.0f}};
    float abs_diff[count] = {{0.0f}};
    float maximum[count] = {{0.0f}};
    float minimum[count] = {{0.0f}};
    float modulus[count] = {{0.0f}};
    float *d_input_1 = NULL;
    float *d_input_2 = NULL;
    float *d_multiply = NULL;
    float *d_divide = NULL;
    float *d_abs_diff = NULL;
    float *d_maximum = NULL;
    float *d_minimum = NULL;
    float *d_modulus = NULL;
    const size_t bytes = count * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_1, bytes));
    CUDA_CHECK(cudaMalloc(&d_input_2, bytes));
    CUDA_CHECK(cudaMalloc(&d_multiply, bytes));
    CUDA_CHECK(cudaMalloc(&d_divide, bytes));
    CUDA_CHECK(cudaMalloc(&d_abs_diff, bytes));
    CUDA_CHECK(cudaMalloc(&d_maximum, bytes));
    CUDA_CHECK(cudaMalloc(&d_minimum, bytes));
    CUDA_CHECK(cudaMalloc(&d_modulus, bytes));
    CUDA_CHECK(cudaMemcpy(d_input_1, input_1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_2, input_2, bytes, cudaMemcpyHostToDevice));

    vector_multiply<<<2, 4>>>(d_input_1, d_input_2, d_multiply, count);
    vector_divide<<<2, 4>>>(d_input_1, d_input_2, d_divide, count);
    vector_absolute_difference<<<2, 4>>>(d_input_1, d_input_2, d_abs_diff, count);
    vector_maximum<<<2, 4>>>(d_input_1, d_input_2, d_maximum, count);
    vector_minimum<<<2, 4>>>(d_input_1, d_input_2, d_minimum, count);
    vector_modulus<<<2, 4>>>(d_input_1, d_input_2, d_modulus, count);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err == cudaErrorUnsupportedPtxVersion) {{
        fprintf(stderr, "CUDA driver cannot run PTX produced by this toolkit\\n");
        return 77;
    }}
    CUDA_CHECK(kernel_err);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(multiply, d_multiply, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(divide, d_divide, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(abs_diff, d_abs_diff, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(maximum, d_maximum, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(minimum, d_minimum, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(modulus, d_modulus, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input_1));
    CUDA_CHECK(cudaFree(d_input_2));
    CUDA_CHECK(cudaFree(d_multiply));
    CUDA_CHECK(cudaFree(d_divide));
    CUDA_CHECK(cudaFree(d_abs_diff));
    CUDA_CHECK(cudaFree(d_maximum));
    CUDA_CHECK(cudaFree(d_minimum));
    CUDA_CHECK(cudaFree(d_modulus));

    if (check_values("multiply", multiply, expected_multiply, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}
    if (check_values("divide", divide, expected_divide, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}
    if (check_values("absolute_difference", abs_diff, expected_abs_diff, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}
    if (check_values("maximum", maximum, expected_maximum, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}
    if (check_values("minimum", minimum, expected_minimum, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}
    if (check_values("modulus", modulus, expected_modulus, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}

    return EXIT_SUCCESS;
}}
""",
            encoding="utf-8",
        )
        executable = Path(build_dir.name) / "test_vector_operations"
        subprocess.run(
            nvcc_command + [str(test_source), "-o", str(executable)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return executable


if __name__ == "__main__":
    unittest.main()
