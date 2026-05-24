import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "main.cu"


class MultiplyArrayCudaProgramTest(unittest.TestCase):
    def test_kernel_multiplies_each_array_element_by_scalar(self):
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
        test_source = Path(build_dir.name) / "test_multiply_array.cu"
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

int main() {{
    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {{
        fprintf(stderr, "no CUDA-capable device is available\\n");
        return 77;
    }}

    const int count = 6;
    const float scalar = 2.5f;
    const float input[count] = {{-4.0f, -1.5f, 0.0f, 1.0f, 3.25f, 10.0f}};
    const float expected[count] = {{-10.0f, -3.75f, 0.0f, 2.5f, 8.125f, 25.0f}};
    float output[count] = {{0.0f}};
    float *d_input = NULL;
    float *d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input, count * sizeof(float), cudaMemcpyHostToDevice));

    multiply_array<<<2, 4>>>(scalar, d_input, d_output, count);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err == cudaErrorUnsupportedPtxVersion) {{
        fprintf(stderr, "CUDA driver cannot run PTX produced by this toolkit\\n");
        return 77;
    }}
    CUDA_CHECK(kernel_err);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_output, count * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    for (int i = 0; i < count; ++i) {{
        if (fabsf(output[i] - expected[i]) > 0.00001f) {{
            fprintf(stderr, "output[%d] was %f, expected %f\\n", i, output[i], expected[i]);
            return EXIT_FAILURE;
        }}
    }}

    return EXIT_SUCCESS;
}}
""",
            encoding="utf-8",
        )
        executable = Path(build_dir.name) / "test_multiply_array"
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
