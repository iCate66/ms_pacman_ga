import unittest
import sys
import logging
from test_variables import TestVariableDefinitions

def run_tests():
    """Run all tests and provide feedback"""
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Run tests
    logger.info("Starting variable definition tests...")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVariableDefinitions)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Process results
    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed!")
        logger.error("\nFailed tests:")
        for failure in result.failures:
            logger.error(f"\n{failure[0]}")
            logger.error(f"Error: {failure[1]}")
        for error in result.errors:
            logger.error(f"\n{error[0]}")
            logger.error(f"Error: {error[1]}")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests())