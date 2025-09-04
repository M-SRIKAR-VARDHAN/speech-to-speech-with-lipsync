import sys
import os
import inspect
from collections import defaultdict

# We need to import the main function from your conversion script.
# MODIFIED: Changed the filename to match your script 'my_convert.py'
try:
    from my_convert import main as rvc_main
except ImportError:
    print("❌ ERROR: Could not import the main function from 'my_convert.py'.")
    print("   Please ensure 'my_convert.py' is in the same directory.")
    sys.exit(1)

# --- Configuration ---
OUTPUT_LOG_FILE = "trace_log.txt"

# --- Global variables to store trace data ---
call_trace = []
site_packages_path = ""
for path in sys.path:
    if "site-packages" in path:
        site_packages_path = path
        break

def tracer(frame, event, arg):
    """
    This function is called by Python for every function call, return, etc.
    We are interested in the 'call' event.
    """
    if event == 'call':
        # Get information about the function being called
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename

        # We only care about functions coming from third-party libraries
        if site_packages_path in filename:
            # Get the full module path (e.g., 'librosa.core.audio')
            module = inspect.getmodule(frame)
            if module:
                full_func_name = f"{module.__name__}.{func_name}"
                
                # Check who called this function
                caller_frame = frame.f_back
                caller_info = "Called from an unknown external source"
                if caller_frame:
                    caller_code = caller_frame.f_code
                    caller_module = inspect.getmodule(caller_frame)
                    if caller_module:
                         caller_info = f"Called from: {caller_module.__name__}.{caller_code.co_name}"

                call_trace.append((full_func_name, caller_info))
    return tracer

def run_with_trace():
    """
    Sets up and runs the RVC script under the tracer's supervision.
    """
    print("=" * 60)
    print("🕵️  STARTING DEEP DEPENDENCY TRACE...")
    print(f"   The full function call log will be saved to '{OUTPUT_LOG_FILE}'")
    print("=" * 60)

    # --- IMPORTANT ---
    # Set up the command-line arguments for my_convert.py here
    # This simulates running it from the terminal.
    # Example: python my_convert.py --model ... --input ...
    args_for_rvc = [
        "my_convert.py",  # The script name is always the first argument
        "--model", "weights/modi.pth",
        "--index", "weights/model.index",
        "--input", "test.mp3",
        "--output", "result.wav",
        # Add any other arguments you use, like --pitch, etc.
    ]

    # Temporarily replace Python's arguments so rvc_main can parse them
    original_argv = sys.argv
    sys.argv = args_for_rvc

    # --- Start the tracer ---
    sys.setprofile(tracer)

    try:
        # --- Run the RVC main function ---
        rvc_main()
    finally:
        # --- Stop the tracer and restore arguments ---
        sys.setprofile(None)
        sys.argv = original_argv
        print("\n🕵️  DEEP DEPENDENCY TRACE FINISHED.")
        print("=" * 60)


def generate_report():
    """
    Processes the collected trace data and writes a structured report.
    """
    if not call_trace:
        print("No third-party function calls were traced.")
        return

    # Use a dictionary to count calls and store callers
    function_usage = defaultdict(lambda: {'count': 0, 'callers': set()})
    
    for func, caller in call_trace:
        function_usage[func]['count'] += 1
        function_usage[func]['callers'].add(caller)

    # Sort the functions by library
    sorted_functions = sorted(function_usage.items())
    
    current_library = ""
    with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("             FUNCTION CALL TRACE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report shows every function called from third-party libraries\n")
        f.write("during the execution of the script.\n\n")

        for func, data in sorted_functions:
            library = func.split('.')[0]
            if library != current_library:
                f.write("\n" + "-" * 60 + "\n")
                f.write(f"LIBRARY: {library}\n")
                f.write("-" * 60 + "\n")
                current_library = library
            
            f.write(f"  -> Function: {func}\n")
            f.write(f"     - Called {data['count']} time(s)\n")
            # f.write(f"     - Called by:\n")
            # for caller in sorted(list(data['callers'])):
            #     f.write(f"       - {caller}\n")

    print(f"✅ Report successfully generated: {OUTPUT_LOG_FILE}")


if __name__ == "__main__":
    # 1. Run the RVC script with tracing enabled
    run_with_trace()
    # 2. Generate the report from the traced data
    generate_report()

