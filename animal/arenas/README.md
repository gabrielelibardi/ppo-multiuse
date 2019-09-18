# Create a basic test set

RUN python create_test_set.py --target-dir /path/to/target/directory

to create a basic test set that a successfully trained agent is expected to complete.
It includes:

    c1 arenas: solve environment with green, orange and red balls of equal size.
    c2 arenas: solve environment with green, orange and red balls of different size.
    c3 arenas: solve environment with multiple objects.
    c4 arenas: solve environment with red and orange zones.
    c5 arenas: solve environment were climbing up ramps is required to get goals.
    c6 arenas: solve environment with multiple objects and random colors.
    c7 arenas: solve environment with multiple objects and blackouts.
