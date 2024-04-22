import os
def change_dir_back(this_path):
    """Decorator to ensure the working directory is changed back after function call."""
    def outer_decorator(func):
        def inner_decorator(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                os.chdir(this_path)
            
        return inner_decorator
    return outer_decorator