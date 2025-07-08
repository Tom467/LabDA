**IMPORTANT**

There may be a delay in loading the online interface as it will go to sleep if it has not been used for a certain period of time and then need a minute or two to wake up, please be patient whilst it boots.



**LabDA**
This GUI automatically processes experimently collected data using Buckingham Pi Theory in order to display in plot format correlations for experimental data.  The dimensionless groups are also printed.

1. **Overview**

**Experimental Research**
Every experiment has two types of variables: independent and dependent. The goal of the experimenter is to derive a relationship whereby they can determine the dependent variable by controlling independent variables. Sometimes this relationship can be directly derived from physics and governing equations, but other times the relationship is harder to uncover. The simplest of procedures for finding this relationship, or even validating a derived relationship, would look something like the following:

Define dependent variable
Define and list possible independent variables
Setup an experiment where the independent variables are varied
Measure the dependent variable

While this process is very simple and can produce powerful results, it turns out to be an "expensive" process in time, and resources. This is due to the exponential increase in combination with the increase of independent variables. For example, let's say we want to find a correlation between how many times a ball on a string will swing back and forth in a set amount of time. The number of swings is the dependent variable (what we don't control) and the set time is the independent variable (what we do control). We can run ten experiments changing how long we let the ball swing for each time and record the number of oscillations. We would unsurprisingly find that the longer we let the ball swing the more oscillations it will make. This is simple enough, but now suppose we also want to know how the mass of the ball, length of the string, release angle, and force of gravity affect the number of oscillations. If we were to follow the same approach and try all the different combinations, we would have to run 100,000 experiments (10 raised to the power of the number of independent variables). This many trial runs even for a simple experiment is clearly problematic, let alone for a more complex setup.

What if we could reduce the number of experiments we have to run by taking advantage of connections between the physical natures of the dependent variable: e.g. the effect due to the mass of the ball and the force of gravity seem likely to be related. These relationships can be defined with dimensional analysis which is done with the units of the variables (kg for mass, m/s^2 for gravity, etc.). One well know method of dimensional analysis is the Buckingham Pi Theorem.

**Using the Buckingham Pi Theorem for Simplified Analysis**
The Buckingham Pi Theorem outlines five steps for reducing the complexity of an experiment through dimensional analysis:

Write down the units of all the independent and dependent variables
Select the appropriate number of repeating variables:
The number of repeating variable depends on the number of base units included the units of the variables
Every base unit mass be included in at least one repeating variable
The units of the repeating variables must be linearly independent
Create a "Pi Group" by taking a non-repeating variable and cancelling its units by multiplying it with the repeating variables raised to exponential powers
Repeat the previous step until one pi group has been made for each non-repeating variable
The pi groups can now be used to find the correlation between the dependent and independent variable
Returning to our example of the ball on the string we would get the following:

Number of oscillations (has no units), Swing time (s), mass (kg), string length (m), release angle (radians), gravity (m/s^2)
We have three base units (kg, m, s) so we need three repeating variables. There are multiple possible combinations of repeating variables that would be acceptable, and we will return to a discussion on this later but for now let us select the period, gravity, and mass.
The length of the string has units of 'm' which can be cancelled out by period^-2 and gravity^1
As it turns out both the oscillations and the release angle are already dimensionless so this step is complete
Our final pi groups are Oscillations, release angle, and string length divided by the period squared times gravity
It is apparent now that instead of describing the experiment with one dependent and five independent variables, we can describe it with three pi groups. What originally needed 100,000 experiments can now be done with 100 experiments, a much more reasonable amount.

Returning to the topic of repeating variables, it was noted that there generally are several acceptable options. While any set of repeating variables would have the same effect in respect to simplification of the experiment, they will not result in the same relationships between pi groups. THe different repeating variable sets can be thought of as different coordinate systems. And just like how the equation for a unit circle is simpler in polar coordinates (r=1) than rectangular coordinates (x^2+y^2=1), so to some repeating variables give a simpler relationship than others.

**The Benefit of Automation**
Dimensional analysis has traditionally been done by hand by the experimenter. The process can be laborious and algebraic errors can occur in rushed calculations. It can be extra tedious exploring all the different combinations of repeating variables. The algorithmic nature of dimensional analysis suggests that a computer can aid in the process, and that is the goal of this project. This package seeks to take experimentally collected data and display all possible relations using dimensional analysis for the user to then interpret. Incredible data is useless if it is displayed ineffectively.


2. **Instructions for Use**

A data table is uploaded to the online GUI application. The data table should be uploaded in the below form, with the variables in the first row and then numerical values in all the following rows:

rho,mu,D,v,F
1,2,3,4,5
....
....
  
This generates a preview of the data and the user has to choose a basis (MLKT or FL).  If the user is unsure, most conventional data can be analysed using MLKT.

The user then assigns base dimensions for each quantity and the GUI automatically finds all the dimensionless Pi groups.  The Pi groups are plotted against each other and plots are also produced for the reciprocals of the Pi groups plotted against each other.

Each plot can then be viewed and saved as needed.
