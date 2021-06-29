from matplotlib import pyplot
from numpy import linspace
from scipy.integrate import odeint
from math import log
import random

class SSA:
    """Container for SSAs"""

    def __init__(self, model, seed=1234):
        """Initialize container with model and pseudorandom number generator"""
        self.model = model
        random.seed(seed)

    def direct(self):
        """Indefinite generator of direct-method trajectories"""
        while True:
            while not self.model.exit():

                # evaluate weights and partition
                weights = [
                    (rxn, sto, pro(self.model))
                    for (rxn, sto, pro) in self.model.reactions
                ]
                partition = sum(w[-1] for w in weights)

                # evaluate sojourn time (MC step 1)
                sojourn = log(1.0 / random.random()) / partition
                self.model["time"].append(self.model["time"][-1] + sojourn)

                # evaluate the reaction (MC step 2)
                partition = partition * random.random()
                while partition >= 0.0:
                    rxn, sto, pro = weights.pop(0)
                    partition -= pro
                for species, delta in sto.items():
                    self.model[species].append(self.model[species][-1] + delta)

                self.model.curate()
            yield self.model
            self.model.reset()

    def first_reaction(self):
        """Indefinite generator of 1st-reaction trajectories"""
        while True:
            while not self.model.exit():

                # evaluate next reaction times
                times = [
                    (
                        log(
                            1.0 / random.random()
                        ) / pro(self.model),
                        sto
                    )
                    for (rxn, sto, pro) in self.model.reactions
                ]
                times.sort()

                # evaluate reaction time
                self.model["time"].append(
                    self.model["time"][-1] + times[0][0]
                )

                # evaluate reaction
                for species, delta in times[0][1].items():
                    self.model[species].append(
                        self.model[species][-1] + delta
                    )

                self.model.curate()
            yield self.model
            self.model.reset()


    """Container for SSA model"""
class SSAModel(dict):
    def __init__(
        self, initial_conditions, propensities, stoichiometry
    ):
        """
        Initialize model with a dictionary of initial conditions (each
     
        """
        super().__init__(**initial_conditions)
        self.reactions = list()
        self.excluded_reactions = list()
        for reaction,propensity in propensities.items():
            if propensity(self) == 0.0:
                self.excluded_reactions.append(
                    (
                        reaction,
                        stoichiometry[reaction],
                        propensity
                    )
                )
            else:
                self.reactions.append(
                    (
                        reaction,
                        stoichiometry[reaction],
                        propensity
                    )
                )

    def exit(self):
        """Return True to break out of trajectory"""

        # return True if no more reactions
        if len(self.reactions) == 0: return True

        # return False if there are more reactions
        else: return False

    def curate(self):
        """Validate and invalidate model reactions"""
        
        # evaulate possible reactions
        reactions = []
        while len(self.reactions) > 0:
            reaction = self.reactions.pop()
            if reaction[2](self) == 0:
                self.excluded_reactions.append(reaction)
            else:
                reactions.append(reaction)
        reactions.sort()
        self.reactions = reactions

        # evaluate impossible reactions
        excluded_reactions = []
        while len(self.excluded_reactions) > 0:
            reaction = self.excluded_reactions.pop()
            if reaction[2](self) > 0:
                self.reactions.append(reaction)
            else:
                excluded_reactions.append(reaction)
        excluded_reactions.sort()
        self.excluded_reactions = excluded_reactions

    def reset(self):
        """Clear the trajectory"""

        # reset species to initial conditions
        for key in self: del self[key][1:]

        # reset reactions per initial conditions
        self.curate()

# initial species counts and sojourn times
initital_conditions = {
    "s": [48000],
    "i": [2000],
    "r": [0],
    "time": [0.0],
}

# propensity functions
propensities = {
    0: lambda d: 2.0 * d["s"][-1] * d["i"][-1] / 50000,
    1: lambda d: 1.0 * d["i"][-1],
}

# change in species for each propensity
stoichiometry = {
    0: {"s": -1, "i": 1, "r": 0},
    1: {"s": 0, "i": -1, "r": 1},
}

# instantiate the epidemic SSA model container
epidemic = SSAModel(
    initital_conditions,
    propensities,
    stoichiometry
)

# instantiate the SSA container with model
epidemic_generator = SSA(epidemic)

# make a nice, big figure
pyplot.figure(figsize=(10,10), dpi=100)

# make a subplot for the susceptible, infected and recovered individuals
axes_s = pyplot.subplot(311)
axes_s.set_ylabel("susceptible individuals")

axes_i = pyplot.subplot(312)
axes_i.set_ylabel("infected individuals")

axes_r = pyplot.subplot(313)
axes_r.set_ylabel("Incidence")
axes_r.set_xlabel("time (arbitrary units)")

# simulate and plot 30 trajectories
trajectories = 0
# for trajectory in epidemic_generator.direct():
#     axes_s.plot(trajectory["time"], trajectory["s"], color="orange")
#     axes_i.plot(trajectory["time"], trajectory["i"], color="orange")
#     axes_r.plot(trajectory["time"], trajectory["r"], color="orange")
#     trajectories += 1
#     if trajectories == 30:
#         break

# numerical solution using an ordinary differential equation solversir
t = linspace(0, 14, num=2000)
y0 = (2499, 1, 0)

z = [3.0,2.7,2.4,2.1,1.8]

for j in z:
    alpha = 18.0
    beta = 18.0 / j

    def differential_SIR(n_SIR, t, alpha, beta):
        dS_dt = -alpha * n_SIR[0] * n_SIR[1] / 2500
        dI_dt = ((alpha * n_SIR[0] / 2500) - beta) * n_SIR[1]
        dR_dt = beta * n_SIR[1]
        return dS_dt, dI_dt, dR_dt

    solution = odeint(differential_SIR, y0, t, args=(alpha, beta))
    solution = [[row[i] for row in solution] for i in range(3)]

    sol_inc = [0]

    for i in range(len(solution[0])):
        if i > 0:
            sol_inc.append(solution[0][i-1]-solution[0][i])

    # plot numerical solution
    axes_s.plot(t, solution[0])
    axes_i.plot(t, solution[1])
    axes_r.plot(t, sol_inc)



pyplot.show()