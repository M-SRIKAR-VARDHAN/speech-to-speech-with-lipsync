import os

def list_directory_tree(path, indent=""):
    """Recursively prints files and folders in a tree structure"""
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(f"{indent}📂 {os.path.basename(path)} [Permission Denied]")
        return

    for i, item in enumerate(items):
        full_path = os.path.join(path, item)
        is_last = (i == len(items) - 1)

        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}{item}")

        if os.path.isdir(full_path):
            new_indent = indent + ("    " if is_last else "│   ")
            list_directory_tree(full_path, new_indent)

if __name__ == "__main__":
    root_dir = "D:\SN_BOSE_INTERNSHIP\demo"  # Change to any directory you want
    print(f"📂 {os.path.abspath(root_dir)}")
    list_directory_tree(root_dir)
