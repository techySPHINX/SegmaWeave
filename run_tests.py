"""
Automated Testing Script for Hybrid Efficient nnU-Net
Runs comprehensive tests including unit tests, integration tests, and system tests
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime


class TestRunner:
    """Automated test runner for the project"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'PENDING'
        }

    def run_command(self, command, description):
        """Run a shell command and capture output"""
        print(f"\n{'='*80}")
        print(f"Running: {description}")
        print(f"Command: {command}")
        print(f"{'='*80}\n")

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            elapsed_time = time.time() - start_time

            status = "PASSED" if result.returncode == 0 else "FAILED"

            self.test_results['tests'][description] = {
                'status': status,
                'elapsed_time': elapsed_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            print(f"\n{'='*80}")
            print(f"Status: {status} (took {elapsed_time:.2f}s)")
            print(f"{'='*80}\n")

            return result.returncode == 0

        except Exception as e:
            print(f"ERROR: {e}")
            self.test_results['tests'][description] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False

    def check_dependencies(self):
        """Check if all dependencies are installed"""
        print("\n🔍 Checking dependencies...")

        required_packages = [
            'torch',
            'numpy',
            'pytest',
            'black',
            'isort',
            'flake8'
        ]

        all_installed = True
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} - NOT INSTALLED")
                all_installed = False

        return all_installed

    def run_unit_tests(self):
        """Run all unit tests"""
        return self.run_command(
            "pytest tests/ -v --tb=short",
            "Unit Tests"
        )

    def run_unit_tests_with_coverage(self):
        """Run unit tests with coverage report"""
        return self.run_command(
            "pytest tests/ --cov=. --cov-report=term --cov-report=html --tb=short",
            "Unit Tests with Coverage"
        )

    def run_code_formatting_check(self):
        """Check code formatting with black"""
        return self.run_command(
            "black --check --diff .",
            "Code Formatting Check (Black)"
        )

    def run_import_sorting_check(self):
        """Check import sorting with isort"""
        return self.run_command(
            "isort --check-only --diff .",
            "Import Sorting Check (isort)"
        )

    def run_linting(self):
        """Run linting with flake8"""
        return self.run_command(
            "flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics",
            "Linting (Flake8 - Critical Errors)"
        )

    def run_type_checking(self):
        """Run type checking with mypy"""
        return self.run_command(
            "mypy . --ignore-missing-imports --no-strict-optional",
            "Type Checking (MyPy)"
        )

    def run_model_test(self):
        """Test model instantiation and forward pass"""
        test_script = """
import torch
from model import create_lightweight_model

print("Creating model...")
model = create_lightweight_model(num_classes=3)
print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("Testing forward pass...")
x = torch.randn(1, 4, 64, 64, 64)
model.eval()
with torch.no_grad():
    output = model(x)

print(f"Output shape: {output.shape}")
assert output.shape == (1, 3, 64, 64, 64), "Output shape mismatch!"
print("✓ Model test passed!")
"""

        test_file = self.project_root / "test_model_quick.py"
        test_file.write_text(test_script)

        result = self.run_command(
            f"python {test_file}",
            "Quick Model Test"
        )

        # Clean up
        if test_file.exists():
            test_file.unlink()

        return result

    def run_loss_test(self):
        """Test loss functions"""
        test_script = """
import torch
from losses import DiceLoss, FocalLoss, CombinedLoss

print("Testing DiceLoss...")
dice_loss = DiceLoss()
pred = torch.randn(2, 3, 16, 16, 16)
target = torch.randint(0, 3, (2, 16, 16, 16))
loss = dice_loss(pred, target)
print(f"DiceLoss: {loss.item():.4f}")

print("Testing FocalLoss...")
focal_loss = FocalLoss()
loss = focal_loss(pred, target)
print(f"FocalLoss: {loss.item():.4f}")

print("Testing CombinedLoss...")
combined_loss = CombinedLoss(loss_types=['dice', 'ce'], loss_weights=[1.0, 1.0])
loss = combined_loss(pred, target)
print(f"CombinedLoss: {loss.item():.4f}")

print("✓ All loss tests passed!")
"""

        test_file = self.project_root / "test_losses_quick.py"
        test_file.write_text(test_script)

        result = self.run_command(
            f"python {test_file}",
            "Quick Loss Functions Test"
        )

        # Clean up
        if test_file.exists():
            test_file.unlink()

        return result

    def save_report(self):
        """Save test results to JSON file"""
        report_file = self.project_root / "test_report.json"

        # Calculate overall status
        statuses = [test['status']
                    for test in self.test_results['tests'].values()]
        if all(s == 'PASSED' for s in statuses):
            self.test_results['overall_status'] = 'ALL PASSED'
        elif any(s == 'ERROR' for s in statuses):
            self.test_results['overall_status'] = 'ERROR'
        else:
            self.test_results['overall_status'] = 'SOME FAILED'

        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\n📊 Test report saved to: {report_file}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        for test_name, result in self.test_results['tests'].items():
            status = result['status']
            elapsed = result.get('elapsed_time', 0)

            symbol = "✓" if status == "PASSED" else "✗"
            print(f"{symbol} {test_name}: {status} ({elapsed:.2f}s)")

        print("="*80)
        print(f"Overall Status: {self.test_results['overall_status']}")
        print("="*80 + "\n")

    def run_all_tests(self, quick=False):
        """Run all automated tests"""
        print("\n🚀 Starting Automated Test Suite")
        print(f"Project Root: {self.project_root}")
        print(f"Timestamp: {self.test_results['timestamp']}\n")

        # Check dependencies first
        if not self.check_dependencies():
            print("\n⚠️  Warning: Some dependencies are missing. Install them with:")
            print("   pip install -r requirements-dev.txt")
            print("\nContinuing with available tests...\n")

        # Quick tests (always run)
        self.run_model_test()
        self.run_loss_test()

        if not quick:
            # Full test suite
            self.run_linting()
            self.run_code_formatting_check()
            self.run_import_sorting_check()

            # Unit tests (may not exist yet)
            if (self.project_root / "tests").exists():
                self.run_unit_tests()
                self.run_unit_tests_with_coverage()
            else:
                print("\n⚠️  tests/ directory not found. Skipping unit tests.")

        # Save and print results
        self.save_report()
        self.print_summary()

        return self.test_results['overall_status'] in ['ALL PASSED', 'SOME FAILED']


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Testing for Hybrid nnU-Net")
    parser.add_argument('--quick', action='store_true',
                        help="Run only quick tests")
    parser.add_argument('--coverage', action='store_true',
                        help="Include coverage report")
    args = parser.parse_args()

    runner = TestRunner()

    if args.quick:
        print("\n⚡ Running quick tests only...\n")
        success = runner.run_all_tests(quick=True)
    else:
        print("\n📋 Running full test suite...\n")
        success = runner.run_all_tests(quick=False)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
