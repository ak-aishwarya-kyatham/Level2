import os
import sys
import importlib.util
import traceback

def run_all_tests():
    print("Starting test discovery...")
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(tests_dir)) # Add backend root to PYTHONPATH
    
    test_files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]
    
    passed_tests = 0
    failed_tests = 0
    
    for file_name in test_files:
        module_name = file_name[:-3]
        file_path = os.path.join(tests_dir, file_name)
        
        print(f"\nRunning tests in module: {module_name}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find functions starting with test_
        test_functions = [getattr(module, name) for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))]
        
        for func in test_functions:
            func_name = func.__name__
            try:
                # Call test function
                func()
                print(f"  [PASS] {func_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  [FAIL] {func_name} (Assertion Failed)")
                traceback.print_exc(limit=2)
                failed_tests += 1
            except Exception as e:
                print(f"  [ERROR] {func_name}: {e}")
                traceback.print_exc(limit=2)
                failed_tests += 1
                
    print(f"\n=========================================")
    print(f"Test Summary: {passed_tests} Passed, {failed_tests} Failed")
    print(f"=========================================")
    
    if failed_tests > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
