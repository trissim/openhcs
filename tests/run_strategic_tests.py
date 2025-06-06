"""
Strategic Test Runner - DNA-Guided Test Execution.

Executes the complete strategic test suite with mathematical validation
and comprehensive reporting based on DNA analysis.

🔬 Execution Strategy:
- Priority-based test execution
- Mathematical validation
- Comprehensive coverage analysis
- Production-grade reporting
"""
import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework.mathematical_test_framework import test_framework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategicTestRunner:
    """
    DNA-guided strategic test runner.
    
    Executes tests in priority order based on mathematical analysis
    and provides comprehensive validation and reporting.
    """
    
    def __init__(self):
        self.test_suites = {
            # Services Layer (Highest Priority - Core Business Logic)
            'services': {
                'priority': 1,
                'tests': [
                    'tests/test_services/test_plate_manager_service.py',
                    'tests/test_services/test_command_service.py'
                ],
                'description': 'Core business logic services'
            },
            
            # Controllers Layer (High Priority - Coordination Logic)
            'controllers': {
                'priority': 2,
                'tests': [
                    'tests/test_controllers/test_application_controller.py'
                ],
                'description': 'UI coordination controllers'
            },
            
            # Commands Layer (Medium Priority - User Actions)
            'commands': {
                'priority': 3,
                'tests': [
                    'tests/test_commands/test_simplified_commands.py'
                ],
                'description': 'User command implementations'
            },
            
            # Integration Layer (Critical Priority - Component Interactions)
            'integration': {
                'priority': 4,
                'tests': [
                    'tests/test_integration/test_tui_integration.py'
                ],
                'description': 'Component integration tests'
            }
        }
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all strategic tests in priority order.
        
        Returns:
            Comprehensive test results with mathematical validation
        """
        logger.info("🧬 Starting Strategic DNA-Guided Test Execution")
        self.start_time = time.time()
        
        # Sort test suites by priority
        sorted_suites = sorted(
            self.test_suites.items(),
            key=lambda x: x[1]['priority']
        )
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for suite_name, suite_config in sorted_suites:
            logger.info(f"🔬 Executing {suite_name} tests: {suite_config['description']}")
            
            suite_results = await self._run_test_suite(suite_name, suite_config)
            self.results[suite_name] = suite_results
            
            total_tests += suite_results['total_tests']
            total_passed += suite_results['passed_tests']
            total_failed += suite_results['failed_tests']
            
            logger.info(f"✅ {suite_name}: {suite_results['passed_tests']}/{suite_results['total_tests']} passed")
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        final_report = await self._generate_final_report(total_tests, total_passed, total_failed)
        
        logger.info(f"🎯 Strategic Test Execution Complete: {total_passed}/{total_tests} passed")
        
        return final_report
    
    async def _run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test suite."""
        suite_start_time = time.time()
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        test_details = []
        
        for test_file in suite_config['tests']:
            logger.info(f"  📋 Running {test_file}")
            
            try:
                # Run pytest on the specific file
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', 
                    test_file, 
                    '-v', 
                    '--tb=short',
                    '--no-header'
                ], 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent
                )
                
                # Parse pytest output
                test_result = self._parse_pytest_output(result.stdout, result.stderr, result.returncode)
                test_details.append({
                    'file': test_file,
                    'result': test_result
                })
                
                total_tests += test_result['total']
                passed_tests += test_result['passed']
                failed_tests += test_result['failed']
                
                if test_result['failed'] > 0:
                    logger.warning(f"    ⚠️  {test_result['failed']} tests failed in {test_file}")
                else:
                    logger.info(f"    ✅ All {test_result['passed']} tests passed in {test_file}")
                
            except Exception as e:
                logger.error(f"    ❌ Error running {test_file}: {e}")
                test_details.append({
                    'file': test_file,
                    'result': {'total': 0, 'passed': 0, 'failed': 1, 'error': str(e)}
                })
                failed_tests += 1
        
        suite_end_time = time.time()
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'execution_time': suite_end_time - suite_start_time,
            'test_details': test_details
        }
    
    def _parse_pytest_output(self, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        lines = stdout.split('\n')
        
        total = 0
        passed = 0
        failed = 0
        
        # Look for test result lines
        for line in lines:
            if '::test_' in line:
                total += 1
                if 'PASSED' in line:
                    passed += 1
                elif 'FAILED' in line or 'ERROR' in line:
                    failed += 1
        
        # If no individual test results found, try to parse summary
        if total == 0:
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # Try to extract from summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed':
                            try:
                                passed = int(parts[i-1])
                            except (ValueError, IndexError):
                                pass
                        elif part == 'failed':
                            try:
                                failed = int(parts[i-1])
                            except (ValueError, IndexError):
                                pass
                    total = passed + failed
                    break
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': returncode
        }
    
    async def _generate_final_report(self, total_tests: int, total_passed: int, total_failed: int) -> Dict[str, Any]:
        """Generate comprehensive final report with mathematical validation."""
        execution_time = self.end_time - self.start_time
        
        # Generate mathematical test framework report
        framework_report = test_framework.generate_test_report()
        
        # Calculate strategic metrics
        pass_rate = total_passed / total_tests if total_tests > 0 else 0
        test_efficiency = total_passed / execution_time if execution_time > 0 else 0
        
        # DNA-based validation
        dna_validation = await self._perform_dna_validation()
        
        final_report = {
            'execution_summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'pass_rate': pass_rate,
                'execution_time': execution_time,
                'test_efficiency': test_efficiency
            },
            'suite_results': self.results,
            'mathematical_framework': framework_report,
            'dna_validation': dna_validation,
            'strategic_assessment': self._generate_strategic_assessment(pass_rate, dna_validation),
            'recommendations': self._generate_recommendations(total_failed, dna_validation)
        }
        
        # Save report to file
        await self._save_report(final_report)
        
        return final_report
    
    async def _perform_dna_validation(self) -> Dict[str, Any]:
        """Perform DNA-based validation of test coverage."""
        try:
            # Run DNA analysis on TUI module
            result = subprocess.run([
                'dna', 'openhcs/tui', '--detailed'
            ], 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                # Parse DNA output for validation metrics
                lines = result.stdout.split('\n')
                
                entropy_total = 0
                issue_count = 0
                file_count = 0
                
                for line in lines:
                    if 'Entropy: Total=' in line:
                        try:
                            entropy_part = line.split('Total=')[1].split(',')[0]
                            entropy_total = float(entropy_part)
                        except (IndexError, ValueError):
                            pass
                    elif 'Found' in line and 'issues' in line:
                        try:
                            issue_count = int(line.split('Found ')[1].split(' ')[0])
                        except (IndexError, ValueError):
                            pass
                    elif 'Analyzed' in line and 'files' in line:
                        try:
                            file_count = int(line.split('Analyzed ')[1].split(' ')[0])
                        except (IndexError, ValueError):
                            pass
                
                return {
                    'entropy_total': entropy_total,
                    'issue_count': issue_count,
                    'file_count': file_count,
                    'entropy_per_file': entropy_total / file_count if file_count > 0 else 0,
                    'issues_per_file': issue_count / file_count if file_count > 0 else 0,
                    'validation_status': 'success'
                }
            else:
                return {
                    'validation_status': 'failed',
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'validation_status': 'error',
                'error': str(e)
            }
    
    def _generate_strategic_assessment(self, pass_rate: float, dna_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic assessment based on test results and DNA analysis."""
        # Calculate overall quality score
        quality_score = pass_rate * 0.6  # 60% weight on test pass rate
        
        if dna_validation.get('validation_status') == 'success':
            # Add DNA-based quality factors
            entropy_factor = max(0, 1 - (dna_validation.get('entropy_per_file', 111) / 150))  # Normalize entropy
            issue_factor = max(0, 1 - (dna_validation.get('issues_per_file', 17) / 50))  # Normalize issues
            
            quality_score += entropy_factor * 0.2  # 20% weight on entropy
            quality_score += issue_factor * 0.2    # 20% weight on issues
        
        # Determine quality level
        if quality_score >= 0.9:
            quality_level = 'EXCELLENT'
        elif quality_score >= 0.8:
            quality_level = 'GOOD'
        elif quality_score >= 0.7:
            quality_level = 'ACCEPTABLE'
        elif quality_score >= 0.6:
            quality_level = 'NEEDS_IMPROVEMENT'
        else:
            quality_level = 'CRITICAL'
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'test_coverage_assessment': 'COMPREHENSIVE' if pass_rate >= 0.95 else 'PARTIAL',
            'architectural_stability': 'STABLE' if dna_validation.get('entropy_per_file', 111) <= 115 else 'UNSTABLE',
            'production_readiness': quality_level in ['EXCELLENT', 'GOOD']
        }
    
    def _generate_recommendations(self, failed_tests: int, dna_validation: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        if failed_tests > 0:
            recommendations.append(f"🔧 Fix {failed_tests} failing tests to improve reliability")
        
        if dna_validation.get('validation_status') == 'success':
            entropy_per_file = dna_validation.get('entropy_per_file', 0)
            issues_per_file = dna_validation.get('issues_per_file', 0)
            
            if entropy_per_file > 115:
                recommendations.append("🧬 Reduce code entropy through further refactoring")
            
            if issues_per_file > 20:
                recommendations.append("⚠️  Address remaining architectural issues")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Test suite is comprehensive and codebase is production-ready")
        
        return recommendations
    
    async def _save_report(self, report: Dict[str, Any]):
        """Save comprehensive report to file."""
        report_dir = Path(__file__).parent / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'strategic_test_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive report saved to: {report_file}")


async def main():
    """Main entry point for strategic test execution."""
    runner = StrategicTestRunner()
    
    try:
        report = await runner.run_all_tests()
        
        # Print summary
        print("\n" + "="*80)
        print("🧬 STRATEGIC DNA-GUIDED TEST EXECUTION COMPLETE")
        print("="*80)
        
        summary = report['execution_summary']
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Pass Rate: {summary['pass_rate']:.1%}")
        print(f"⏱️  Execution Time: {summary['execution_time']:.2f}s")
        print(f"🚀 Test Efficiency: {summary['test_efficiency']:.2f} tests/sec")
        
        assessment = report['strategic_assessment']
        print(f"\n🎯 Quality Score: {assessment['quality_score']:.3f}")
        print(f"🏆 Quality Level: {assessment['quality_level']}")
        print(f"🔬 Production Ready: {assessment['production_readiness']}")
        
        print("\n📋 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "="*80)
        
        # Exit with appropriate code
        if summary['failed_tests'] == 0:
            print("🎉 ALL TESTS PASSED - PRODUCTION READY!")
            sys.exit(0)
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Strategic test execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
