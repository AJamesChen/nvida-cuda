import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "main.cu"


class PrimeCudaProgramTest(unittest.TestCase):
    def test_prime_predicate_classifies_inputs(self):
        executable = self._build_test_executable()

        result = subprocess.run(
            [str(executable)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertEqual(result.stdout, "")

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
        test_source = Path(build_dir.name) / "test_prime.cu"
        test_source.write_text(
            f"""
#include <stdlib.h>
#include <limits.h>

#define UNIT_TEST
#include "{SOURCE}"

int main() {{
    const int inputs[] = {{-3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25, 29, 2147483646, INT_MAX}};
    const int expected[] = {{0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1}};
    const int count = sizeof(inputs) / sizeof(inputs[0]);

    for (int i = 0; i < count; ++i) {{
        if (is_prime_number(inputs[i]) != expected[i]) {{
            return EXIT_FAILURE;
        }}
    }}

    return EXIT_SUCCESS;
}}
""",
            encoding="utf-8",
        )
        executable = Path(build_dir.name) / "test_prime"
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
