
C++ output and PyTorch outputs are not the same


I have applied various methods to make it aligned,


    1. Manual seeding on both implementations.

    
    2. uniform initialization of weights and bias.

    
    3. same normalization method on both sides.

    
    4. The difference in datatypes between C++ and Python is also addressed. This needs to be evaluated.

    

The difference between the datatypes of Pytorch and C++ can be a problem.
