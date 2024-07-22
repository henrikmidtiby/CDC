# Reference - Multiprocessing

This document will explain why it has been choosen to not multiprocess, the main function beint called and the seperate tiles.

This is because when testing the program a tile size of 500 x 500 was used. 
When testing with multiprocessing it was found that it increased the amount of time it took the system to run, as each process had to be created and destroyed, which took longer then to process the tile.
