# Create a basic test set

RUN python create_test_set.py --target-dir /path/to/target/directory

to create a basic test set that a successfully trained agent is expected to complete.
It includes:

    - 10 c1 arenas: solve environment with green, orange and red balls of equal size.
    - 10 c2 arenas: solve environment with green, orange and red balls of different size.
    - 10 c3 arenas: solve environment with multiple objects.
    - 10 c4 arenas: solve environment with red and orange zones.
    - 10 c5 arenas: solve environment were climbing up ramps is required to get goals.
    - 10 c6 arenas: solve environment with multiple objects and random colors.
    - 10 c7 arenas: solve environment with multiple objects and blackouts.
