import concurrent.futures
import functools
from collections.abc import Callable
from typing import Any


def run_until_timeout(func: Callable, timeout: float, timeout_error_type: type[Exception], *args, **kwargs) -> Any:
    """Runs the function in a separate thread and raises a exception if it exceeds the timeout.

    Args:
        func (Callable): The function to be executed.
        timeout (float): The maximum time the function is allowed to run.
        timeout_error_type (Type[Exception]): The type of exception to raise in case of a timeout.
        *args: Variable length argument list to pass to the function.
        **kwargs: Arbitrary keyword arguments to pass to the function.

    Raises:
        timeout_error_type: If the function exceeds the specified timeout.

    Returns:
        The result of the function.
    """

    # Define a function that wraps the original function and raises a timeout error in case of timeout
    def func_with_timeout(*args: Any, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except timeout_error_type:
            raise timeout_error_type(f"Function {func.__name__} took too long to respond")

    # Create a partial function with the provided arguments
    wrapped_func = functools.partial(func_with_timeout, *args, **kwargs)

    # Run the function in a separate thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(wrapped_func)
        try:
            # Wait for the function to complete, raise a timeout error if it exceeds the specified timeout
            result = future.result(timeout=timeout)
        except (Exception, timeout_error_type):
            raise timeout_error_type(f"Function {func.__name__} took too long to respond")
    return result
