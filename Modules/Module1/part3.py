## TODO: Define the goes_outside method here
def goes_outside(temperature):
    """
    Compute if chad will go outside
    (More than 60 but less than 80 degrees)

    Parameters
    ----------
    temperature: int
        the temperature outside
    """
    ret = False
    if temperature > 60 and temperature < 80:
        ret = True
    return ret




# Run some tests on the method
print(goes_outside(50), end='.')
print(goes_outside(65), end='.')
print(goes_outside(72), end='.')
print(goes_outside(83), end='.')
print(goes_outside(90), end='.')