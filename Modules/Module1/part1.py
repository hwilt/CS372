import math
def print_bus_vans(bus_capacity, van_capacity, num_people):
    """
    Compute the bus and van capacity
    Parameters
    ----------
    bus_capacity: int
     Number of people a bus can hold
    van_capacity: int
     Number of people a van can hold
    num_people: int
     Number of people on the trip
    """
    ## TODO: Change these two variables to reflect the actual
    ## number needed
    num_buses = 0
    num_vans = 0
    num_buses = math.floor(num_people/bus_capacity)
    num_people -= num_buses * bus_capacity
    num_vans = math.ceil(num_people/van_capacity)
    print("{} Buses, {} Vans".format(num_buses, num_vans), end='.')


# Run some tests on the method
print_bus_vans(10, 4, 87)
print_bus_vans(20, 7, 87)
print_bus_vans(20, 8, 5)