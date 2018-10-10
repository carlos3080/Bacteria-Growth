# Problem Set 4: Simulating the Spread of Disease and Bacteria Population Dynamics
# Name: Carlos Hernandez
# Collaborators (Discussion): None
# Time: 5 hours
# Difficult Sections:

import math
import numpy as np
import pylab
import random
import matplotlib.pyplot as plt


##########################
# End helper code
##########################

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacterium
    and ResistantBacterium classes to indicate that a bacteria cell does not
    reproduce. You should use NoChildException as is; you do not need to
    modify it or add any code.
    """

##########################
# PROBLEM 1
##########################

class SimpleBacterium(object):
    """A simple bacteria cell with no antibiotic resistance"""

    def __init__(self, birth_prob, death_prob):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob

    def is_killed(self):
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        return random.random() < self.death_prob

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        birth_prob * (1 - population density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacterium (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacterium: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        if random.random() < self.birth_prob * (1 - pop_density):
            return SimpleBacterium(self.birth_prob, self.death_prob)
        raise NoChildException


class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria (list of SimpleBacterium): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        self.bacteria = bacteria
        self.max_pop = max_pop

    def get_total_pop(self):
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determine whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassign the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        new_bacteria_list = []
        for bacteria in self.bacteria:
            if not bacteria.is_killed():
                new_bacteria_list.append(bacteria)
        pop_density = len(new_bacteria_list) / self.max_pop
        offspring_list = []
        for bacteria in new_bacteria_list:
            try:
                offspring = bacteria.reproduce(pop_density)
                offspring_list.append(offspring)
            except NoChildException:
                pass
        self.bacteria = new_bacteria_list + offspring_list
        return len(self.bacteria)





##########################
# PROBLEM 2
##########################

def simulation_without_antibiotic(num_bacteria,
                                  max_pop,
                                  birth_prob,
                                  death_prob,
                                  num_trials):
    """
    Run the simulation. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacterium
        * instantiate a Patient using the list of SimpleBacterium
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. Note
          that the first time step should contain the starting number of
          bacteria in the patient

    Args:
        num_bacteria (int): number of SimpleBacterium to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    populations = []
    for trial in range(num_trials):

        # create new list of bacteria for each trial
        bacteria_list = []

        # create new patient with new list of bacteria
        for i in range(num_bacteria):
            bacteria_list.append(SimpleBacterium(birth_prob, death_prob))
        patient = Patient(bacteria_list, max_pop)

        # add a sublist representing the trial
        populations.append([])
        for timeStep in range(300):

            # add to the population at each timeStep to the sublist (trial) and update the population
            populations[trial].append(patient.get_total_pop())
            patient.update()
    return populations

# When you are ready to run the simulation, uncomment the next line
# populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 50)
# print(populations)
# print(len(populations))
# print(len(populations[7]))

##########################
# PROBLEM 3
##########################

def calc_pop_avg(populations, n):
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
        n (int): time step
    Returns:
        float: The average bacteria population size at time step n
    """
    # stores the population sizes at a timeStep n from each trial
    pop_sizes = []
    for trial in populations:
        pop_sizes.append(trial[n])

    # return the average value in pop_sizes
    return sum(pop_sizes) / len(pop_sizes)
    
def calc_pop_std(populations, t):
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root

    You may not use third-party functions that calculate standard deviation,
    such as numpy.std. Other built-in or third-party functions that do not
    calculate standard deviation may be used.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    average = calc_pop_avg(populations, t)
    pop_sizes_minus_avg_squared = []
    for trial in populations:
        pop_sizes_minus_avg_squared.append((trial[t] - average)**2)
    return (sum(pop_sizes_minus_avg_squared) / len(pop_sizes_minus_avg_squared))**(1/2)



def calc_95_ci(populations, t):
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        I.e., you should return a tuple containing (mean, width)
    """
    mean = calc_pop_avg(populations, t)
    standard_dev = calc_pop_std(populations, t)
    sem = standard_dev / len(populations)**(1/2)
    return mean, 1.96 * sem

def plot_simulation_without_antibiotic(populations):
    """
    Makes a plot of the bacteria population with error bars representing the
    95% confidence interval around the average bacteria population.
    Axis and plot title should be present. Use the code at the bottom of this
    document to test your implementation.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    x_values = []  # time steps
    y_values = []  # number of bacteria with error
    yerr = []  # list of different errors for each time step
    for x in range(300):
        x_values.append(x)
        y_values.append(calc_95_ci(populations, x)[0])
        yerr.append(calc_95_ci(populations, x)[1])
    plt.errorbar(x_values, y_values, yerr=yerr)
    plt.title("Simulation Without Antibiotic")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Bacteria")
    plt.show()

##########################
# PROBLEM 4
##########################

class ResistantBacterium(SimpleBacterium):
    """A bacteria cell that can have antibiotic resistance."""

    def __init__(self, birth_prob, death_prob, resistant, mut_prob):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        SimpleBacterium.__init__(self, birth_prob, death_prob)
        self.resistant = resistant
        self.mut_prob = mut_prob

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""
        return self.resistant

    def is_killed(self):
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability / 4. If not resistant,
        the bacteria dies with the regular death probability, making the
        probability of death higher.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        if self.resistant:
            return random.random() < self.death_prob / 4
        # return SimpleBacterium.is_killed(self)
        return random.random() < self.death_prob

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacterium, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacterium: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        if random.random() < self.birth_prob * (1 - pop_density):
            if self.resistant:
                return ResistantBacterium(self.birth_prob, self.death_prob, self.resistant, self.mut_prob)
            return ResistantBacterium(self.birth_prob, self.death_prob, random.random() < self.mut_prob * (1 - pop_density), self.mut_prob)
        raise NoChildException


class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function should initialize self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.

        Don't forget to call Patient's __init__ method at the start of this
        method.
        """
        Patient.__init__(self, bacteria, max_pop)
        self.on_antibiotic = False

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotic = True

    def get_resistant_pop(self):
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        population = 0
        for bacteria in self.bacteria:
            if type(bacteria) is ResistantBacterium:
                if bacteria.get_resistant():
                    population += 1
        return population

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Use the same calculation as in Patient

        4. Based on this value of population density, determine whether each
           surviving bacteria cell should reproduce and add offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        new_bacteria_list = []
        for bacteria in self.bacteria:
            if not bacteria.is_killed():
                new_bacteria_list.append(bacteria)
        if self.on_antibiotic:
            resistant_bacteria = []
            for bacteria in new_bacteria_list:
                if bacteria.get_resistant:
                    resistant_bacteria.append(bacteria)
            new_bacteria_list = resistant_bacteria
        pop_density = len(new_bacteria_list) / self.max_pop
        offspring_list = []
        for bacteria in new_bacteria_list:
            try:
                offspring = bacteria.reproduce(pop_density)
                offspring_list.append(offspring)
            except NoChildException:
                pass
        self.bacteria = new_bacteria_list + offspring_list
        return len(self.bacteria)


##########################
# PROBLEM 5
##########################

def simulation_with_antibiotic(num_bacteria,
                               max_pop,
                               birth_prob,
                               death_prob,
                               resistant,
                               mut_prob,
                               num_trials):
    """
    Runs simulations and plots graphs for problem 4.

    For each of num_trials trials:
        * instantiate a list of ResistantBacterium
        * instantiate a patient
        * run a simulation for 150 timesteps, add the antibiotic, and run the
          simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot. You might find
    the helper function make_two_curve_plot helpful

    Args:
        num_bacteria (int): number of ResistantBacterium to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacterium cells
        num_trials (int): number of simulation runs to execute

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    populations = []
    resistant_population = []
    for trial in range(num_trials):

        # create new list of bacteria for each trial
        bacteria_list = []

        # create new patient with new list of bacteria
        for i in range(num_bacteria):
            bacteria_list.append(ResistantBacterium(birth_prob, death_prob, resistant, mut_prob))
        patient = TreatedPatient(bacteria_list, max_pop)

        # add a sublist representing the trial
        populations.append([])
        resistant_population.append([])
        for timeStep in range(400):
            # add to the population at each timeStep to the sublist (trial) and update the population
            populations[trial].append(patient.get_total_pop())
            resistant_population[trial].append(patient.get_resistant_pop())
            patient.update()
            if timeStep == 150:
                patient.set_on_antibiotic()
    return populations, resistant_population

def plot_simulation_with_antibiotic(populations, resistant_pop):
    """
    Makes a plot with two curves on it. One curve depicts the bacteria
    population and the second curve depicts the resistant population.
    Both curves should include error bars representing the 95% confidence
    interval around the average populations. Include a title, labels, and
    a legend.

    Args:
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    x_values = []  # time steps
    y_values = []  # number of bacteria with error
    y_values_r = []  # number of resistant bacteria with error
    yerr = []  # list of different errors for each time step for total population
    yerr_r = []  # list of different errors for each time step for resistant population
    for x in range(400):
        x_values.append(x)
        y_values.append(calc_95_ci(populations, x)[0])
        y_values_r.append(calc_95_ci(resistant_pop, x)[0])
        yerr.append(calc_95_ci(populations, x)[1])
        yerr_r.append(calc_95_ci(resistant_pop, x)[1])
    plt.errorbar(x_values, y_values, yerr=yerr, label="Total Bacteria")
    plt.errorbar(x_values, y_values_r, yerr=yerr_r, label="Resistant Bacteria")
    plt.title("Simulation With Antibiotic")
    plt.xlabel("Time Step")
    plt.ylabel("Bacteria Population")
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    pass
    # When you are ready to run the simulations, uncomment the next lines one
    # at a time

    ############################
    # Problem 3
    ############################

    populations = simulation_without_antibiotic(num_bacteria=100,
                                                max_pop=1000,
                                                birth_prob=0.5,
                                                 death_prob=0.3,
                                                num_trials=50)
    plot_simulation_without_antibiotic(populations)
    ############################
    # Problem 5
    ############################
    ## EX1
    total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
                                                           max_pop=800,
                                                           birth_prob=0.25,
                                                           death_prob=0.15,
                                                           resistant=False,
                                                           mut_prob=0.1,
                                                           num_trials=50)
    plot_simulation_with_antibiotic(total_pop, resistant_pop)
    
    ## EX2
    total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
                                                           max_pop=800,
                                                           birth_prob=0.08,
                                                           death_prob=0.2,
                                                           resistant=False,
                                                           mut_prob=0.8,
                                                           num_trials=50)
    plot_simulation_with_antibiotic(total_pop, resistant_pop)
