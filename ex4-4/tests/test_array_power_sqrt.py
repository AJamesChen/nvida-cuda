import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "main.cu"


class ArrayPowerSqrtCudaProgramTest(unittest.TestCase):
    def test_kernels_calculate_square_and_square_root_results(self):
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

        build_dir = tempfile.TemporaryDirectory()
        self.addCleanup(build_dir.cleanup)
        test_source = Path(build_dir.name) / "test_array_power_sqrt.cu"
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
    const float input[count] = {{0.0f, 1.0f, 2.25f, 4.0f, 9.0f, 16.0f}};
    const float expected_square[count] = {{0.0f, 1.0f, 5.0625f, 16.0f, 81.0f, 256.0f}};
    const float expected_square_root[count] = {{0.0f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f}};
    float square[count] = {{0.0f}};
    float square_root[count] = {{0.0f}};
    float *d_input = NULL;
    float *d_square = NULL;
    float *d_square_root = NULL;
    const size_t bytes = count * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_square, bytes));
    CUDA_CHECK(cudaMalloc(&d_square_root, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice));

    array_square<<<2, 4>>>(d_input, d_square, count);
    array_square_root<<<2, 4>>>(d_input, d_square_root, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(square, d_square, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(square_root, d_square_root, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_square));
    CUDA_CHECK(cudaFree(d_square_root));

    if (check_values("square", square, expected_square, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}
    if (check_values("square_root", square_root, expected_square_root, count) != EXIT_SUCCESS) {{
        return EXIT_FAILURE;
    }}

    return EXIT_SUCCESS;
}}
""",
            encoding="utf-8",
        )
        executable = Path(build_dir.name) / "test_array_power_sqrt"
        subprocess.run(
            [nvcc, str(test_source), "-o", str(executable)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return executable


if __name__ == "__main__":
    unittest.main()
