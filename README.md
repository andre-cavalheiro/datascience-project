Project for Datascience course - 2019/2010 semester
<<<<<<< .mine

This last mode needs a little more experimentation to be at full potential but it's function for the moment. The abstraction made to the code enables us to very simply use optimization techniques. The one enabled so far is [Baysian Optimization](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf). 

Jarvis uses the Optuna package to do it. You need to configure it via the Puppet's configuration file ```src/args.py``` depending on the type of variable. (See example in Puppet Class).

You can configure the optimization process in Jarvis' configurations with:

```
  numTrials: 500
  numJobs: 1
```

To better understand this you can look up [optuna's documentation](https://optuna.readthedocs.io/en/stable/tutorial/configurations.html).

=======














>>>>>>> .theirs
