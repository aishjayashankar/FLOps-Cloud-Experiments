import os
import logging
import time

# Set environment variable for log file
log_file = "test_logging.log"
os.environ["LOG_FILE"] = log_file

# Remove existing log file if it exists
if os.path.exists(log_file):
    os.remove(log_file)

print(f"Setting LOG_FILE to {log_file}")

# Import modules to trigger logging configuration
# Note: We need to import them inside the script to ensure env var is picked up
# But since python imports are cached, if we were running this in a REPL it might be an issue.
# As a standalone script, it's fine.

try:
    import flops_infra_drift.client_app
    import flops_infra_drift.server_app
    print("Imported client_app and server_app")
except ImportError as e:
    print(f"Import failed: {e}")
    # Try to add current directory to path if needed, though it should be fine
    import sys
    sys.path.append(os.getcwd())
    try:
        import flops_infra_drift.client_app
        import flops_infra_drift.server_app
        print("Imported client_app and server_app after adding cwd to path")
    except ImportError as e:
        print(f"Import failed again: {e}")
        exit(1)

# Generate some logs
logging.info("This is a test log message from verification script.")

# Check if file exists and has content
if os.path.exists(log_file):
    print(f"Log file {log_file} created successfully.")
    with open(log_file, "r") as f:
        content = f.read()
        print("Log file content:")
        print(content)
        
        if "Logging configured" in content:
            print("SUCCESS: Logging configuration message found.")
        else:
            print("FAILURE: Logging configuration message NOT found.")
            
        if "This is a test log message" in content:
            print("SUCCESS: Test log message found.")
        else:
            print("FAILURE: Test log message NOT found.")
else:
    print(f"FAILURE: Log file {log_file} was NOT created.")
