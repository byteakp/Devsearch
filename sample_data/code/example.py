# search_project/sample_data/code/example.py
class AdvancedMath:
    """
    A class for performing advanced mathematical operations.
    Includes methods for Fibonacci sequence generation.
    """
    def fibonacci(self, n: int) -> list[int]:
        """
        Generates the Fibonacci sequence up to n numbers.
        Args:
            n: The number of Fibonacci numbers to generate.
        Returns:
            A list containing the Fibonacci sequence.
        Raises:
            ValueError: If n is not a positive integer.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Input must be a positive integer for Fibonacci sequence length.")
        elif n == 1:
            return [0]
        
        sequence = [0, 1]
        while len(sequence) < n:
            next_val = sequence[-1] + sequence[-2]
            sequence.append(next_val)
        return sequence

def factorial(num: int) -> int:
    """
    Computes the factorial of a non-negative integer recursively.
    Args:
        num: The number to compute factorial for.
    Returns:
        The factorial of num.
    Raises:
        ValueError: If num is negative.
        TypeError: If num is not an integer.
    """
    if not isinstance(num, int):
        raise TypeError("Input must be an integer for factorial calculation.")
    if num < 0:
        raise ValueError("Factorial not defined for negative numbers.")
    elif num == 0:
        return 1
    else:
        # Basic recursive factorial
        return num * factorial(num - 1)

# Example of a potential error source for testing search
# This part might be commented out or be part of a test suite usually.
# def cause_index_error_example():
#     my_empty_list = []
#     print("This line will cause an error if uncommented and called:")
#     print("Accessing element from empty list:", my_empty_list[0]) # This will cause an IndexError

if __name__ == '__main__':
    math_ops = AdvancedMath()
    print(f"Fibonacci sequence (up to 10 numbers): {math_ops.fibonacci(10)}")
    print(f"Factorial of 5: {factorial(5)}")

    try:
        print(f"Attempting Factorial of -2: {factorial(-2)}")
    except ValueError as e:
        print(f"Caught expected error (ValueError) calculating factorial: {e}")
    
    try:
        print(f"Attempting Factorial of 3.5: {factorial(3.5)}") # type: ignore
    except TypeError as e:
        print(f"Caught expected error (TypeError) calculating factorial: {e}")

    # Uncomment to test IndexError logging if you have such an error document:
    # try:
    #     cause_index_error_example()
    # except IndexError as e:
    #     print(f"Caught expected error (IndexError): {e}")