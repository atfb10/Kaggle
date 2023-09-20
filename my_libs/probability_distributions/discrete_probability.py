'''
Adam Forestier
September 2023
This file contains functions to solve discrete probability distribution questions
'''

import math

EULER = math.e

def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer 'n'.

    The factorial of a non-negative integer 'n' is the product of all positive
    integers from 1 to 'n'. For example, the factorial of 5 (5!) is 5 x 4 x 3 x 2 x 1 = 120.

    Parameters:
        n (int): The non-negative integer for which to calculate the factorial.

    Returns:
        int: The factorial of 'n'.

    Raises:
        ValueError: If 'n' is a negative integer.

    Examples:
        >>> factorial(0)
        1
        >>> factorial(5)
        120
        >>> factorial(10)
        3628800

    Note:
        - The function handles non-negative integers only. It will raise a ValueError
          if 'n' is negative.
        - The factorial of 0 is defined as 1 by convention.
    """
    if n < 0:
        raise ValueError("Integer must be non-negative")
    
    # Calculate the factorial using a loop
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


def binomial_distribution(p: float, x: int, n: int) -> float:
    """
    Calculate the probability mass function (PMF) of a binomial distribution.

    This function computes the probability of getting exactly 'x' successes
    in 'n' independent Bernoulli trials, each with a probability of success 'p'.

    Parameters:
        p (float): The probability of success in each trial (0 <= p <= 1).
        x (int): The number of successes to calculate the PMF for (0 <= x <= n).
        n (int): The total number of trials (n >= 0).

    Returns:
        float: The probability of getting 'x' successes in 'n' trials.

    Raises:
        ValueError: If any of the input arguments are invalid.
    """
    if p < 0 or p > 1:
        raise ValueError('Probability must be inclusively between 0 and 1')
    if x > n:
        raise ValueError('Total number of trials must be greater than or equal to number of successes')
    n_sub_x = n - x
    return round(factorial(n=n) / (factorial(n=x) * factorial(n=n_sub_x)) * (p ** x) * ((1 - p) ** n_sub_x), 4)

def poisson_distribution(l: float, x: int) -> float:
    """
    Calculate the probability mass function (PMF) of the Poisson distribution.

    The Poisson distribution models the number of events occurring in a fixed
    interval of time or space, given a known average rate of occurrence 'l'.

    Parameters:
        l (float): The average rate of occurrence (lambda) of events. Must be >= 0.
        x (int): The number of events to calculate the PMF for. Must be >= 0.

    Returns:
        float: The probability of observing 'x' events in the given interval.

    Raises:
        ValueError: If 'l' or 'x' is less than 0.

    Note:
        - The Poisson distribution is commonly used to model rare events.
        - The PMF is calculated using the formula:
          PMF(x; l) = (e^(-l) * l^x) / x! where 'e' is Euler's number.
        - The result is rounded to 4 decimal places for readability.
    """
    if l < 0:
        raise ValueError("Lambda (average) must be greater than or equal to 0")
    if x < 0:
        raise ValueError("X must be greater than or equal to 0")

    result = (EULER ** (-1 * l)) * (l ** x) / factorial(x)
    return round(result, 4)

