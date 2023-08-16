---
title: Writing MATLAB Scripts
teaching: 35
exercises: 0
---

::::::::::::::::::::::::::::::::::::::: objectives

- "Write and save MATLAB scripts."
- "Save MATLAB plots to disk."

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- "How can I save and re-use my programs?"

::::::::::::::::::::::::::::::::::::::::::::::::::


So far, we've typed in commands one-by-one on the command line
to get MATLAB to do things for us. But what if we want to repeat
our analysis? Sure, it's only a handful of commands,
and typing them in shouldn't take
us more than a few minutes. But if we forget a step or make a mistake,
we'll waste time rewriting commands. Also, we'll quickly find ourselves
doing more complex analyses, and we'll need our results to
be more easily reproducible.

In addition to running MATLAB commands one-by-one on the command line,
we can also write several commands in a *script*.
A MATLAB script is just a text file with a `.m` extension.
We've written commands to load data from a `.csv` file,
compute statistics of the data and display some plots about that data.
Let's put those commands in a script called `patient_analysis.m`,
which we'll save in the `src` directory in our current folder,`matlab-novice-inflammation`.

To create a new script we can click the "New script" button on the top left, or use the command:
```matlab
edit src/patient_analysis.m
```

Matlab will create a file called `patient_analysis.m` in the `src` folder.
It is important that we let matlab know that we want it to find stuff in this folder.
To do this, right click on the folder icon in the file browser and select "Add to Path".

:::::::::::::::::::::::::::::::::::::::::  callout

## The MATLAB path

MATLAB knows about files in the current directory, but if we want to
run a script saved in a different location, we need to make sure that
this file is visible to MATLAB.
We do this by adding directories to the MATLAB **path**.
The *path* is a list of directories MATLAB will search through to locate
files.

To add a directory to the MATLAB path,
we go to the `Home` tab,
click on `Set Path`,
and then on `Add with Subfolders...`.
We navigate to the directory and
add it to the path to tell MATLAB where to look for our files. When you refer
to a file (either code or data), MATLAB will search all the directories in the path
to find it. Alternatively, for data files, we can provide the relative or
absolute file path.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::  callout

## GNU Octave

Octave has only recently gained a MATLAB-like user interface. To change the
path in any version of Octave, including command-line-only installations, use
`addpath('path/to/directory')`

::::::::::::::::::::::::::::::::::::::::::::::::::

We can now type the contents of the script:

```matlab
% Load patient data
patient_data = readmatrix('data/inflammation-01.csv');

% Compute global statistics
g_mean = mean(patient_data(:));
g_max = max(patient_data(:));
g_min = min(patient_data(:));

% Compute patient statistics
p_mean = mean(patient_data(5,:));
p_max = max(patient_data(5,:));
p_min = min(patient_data(5,:));

% Compare patient vs global
disp('Patient 5:')
disp('High mean?')
disp(p_mean > g_mean)
disp('Highest max?')
disp(p_max == g_max)
disp('Lowest min?')
disp(p_min == g_min)
```

Now, before running this script lets clear our workplace so that we can see what is happening.
```matlab
>> clear
>> clc
```


If you now run the script by clicking "Run" on the graphical user interface,
pressing `F5` on the keyboard,
or typing the script's name `patient_analysis` on the command line (without extention),
you'll see a bunch of variables appear on the workspace and this output:
```matlab
patient_analysis
```
```output
Patient 5:
High mean?
   0
Highest max?
   0
Lowest min?
   1
```

Remember, we supressed most outputs with `;`, so the only lines printed are the ones with `disp`.

As you can see, the script ran every line of code in the script in order, and created any variable we asked for.
Having the code in the script makes it much easier to follow what we are doing, and also make changes.
For example, if we now want to look at patient 8, all we need to do is change the number in lines 10, 11 and 12.
We can actually do a bit better, and replace that number for a variable `patient_number`.
This variable needs to exist before it is used, so lets insert it before computing the patient statistics, like so:
```matlab
% Load patient data
patient_data = readmatrix('data/inflammation-01.csv');

% Compute global statistics
g_mean = mean(patient_data(:));
g_max = max(patient_data(:));
g_min = min(patient_data(:));

patient_number = 8;

% Compute patient statistics
p_mean = mean(patient_data(patient_number,:));
p_max = max(patient_data(patient_number,:));
p_min = min(patient_data(patient_number,:));

% Compare patient vs global
disp('Patient:')
disp(patient_number)
disp('High mean?')
disp(p_mean > g_mean)
disp('Highest max?')
disp(p_max == g_max)
disp('Lowest min?')
disp(p_min == g_min)
```

Note that we also changed the disp commands to show the right patient number.

Getting the results for whichever patient is now as simple as changing the value of `patient_number`.

For the case of patient 8, we get:
```matlab
patient_analysis
```
```output
Patient 8:
High mean?
   1
Highest max?
   1
Lowest min?
   1
```

:::::::::::::::::::::::::::::::::::::::::  callout

## Help text

You might have noticed that we described what we want
our code to do using the percent sign: `%`.
This is another plus of writing scripts: you can comment
your code to make it easier to understand when you come
back to it after a while.

A comment can appear on any line, but be aware that the first line
or block of comments in a script or function is used by MATLAB as the
**help text**.
When we use the `help` command, MATLAB returns the *help text*.
The first help text line (known as the **H1 line**)
typically includes the name of the program, and a brief description.
The `help` command works in just the same way for our own programs as for
built-in MATLAB functions.
You should write help text for all of your own scripts and functions.

::::::::::::::::::::::::::::::::::::::::::::::::::

Let's write an H1 line at the top of our script:

```matlab
% PATIENT_ANALYSIS   Computes mean, max and min of a patient and compares to global statistics.
```

We can then get help for our script by running

```matlab
help patient_analysis
```
```output
  patient_analysis   Computes mean, max and min of a patient and compares to global statistics.
```

## Script for plotting

Of course, our scripts can be as complicated as we like.
There were a lot of commands involved with plotting, so lets try and put that in a script.

Create a new script in the current directory called `plot_daily_average.m`
```matlab
edit src/plot_daily_average.m
```

In the script, lets recap what we need to do:

```matlab
% PLOT_DAILY_AVERAGE   Plots daily average inflammation accross patients.

% Load patient data
patient_data = readmatrix('data/inflammation-01.csv');

figure

% Plot average inflammation per day
plot(mean(patient_data, 1))
title('Daily average inflammation')
xlabel('Day of trial')
ylabel('Inflammation')
```

Note that we are explicitly creating a new figure window using the `figure` command.

Try this on the command line:

```matlab
figure
```

MATLAB's plotting commands only create a new figure window if one doesn't already exist:
the default behaviour is to reuse the current figure window as we saw in the previous episode.
Explicitly creating a new figure window in the script avoids any unexpected results from
plotting on top of existing figures.

Now lets run the script:
```matlab
plot_daily_average
```

You should see the figure appear.


Let's modify our `plot_daily_average` script so that it creates sub-plots,
rather than individual plots.

```matlab
% PLOT_DAILY_AVERAGE   Plots daily average, max and min inflammation accross patients.

% Load patient data
patient_data = readmatrix('data/inflammation-01.csv');

figure

% Plot average inflammation per day
subplot(1, 3, 1)
plot(mean(patient_data, 1))
title('Daily average inflammation')
xlabel('Day of trial')
ylabel('Inflammation')

% Plot max inflammation per day
subplot(1, 3, 2)
plot(max(patient_data, [], 1))
title('Max')
ylabel('Inflammation')
xlabel('Day of trial')

% Plot min inflammation per day
subplot(1, 3, 3)
plot(min(patient_data, [], 1))
title('Min')
ylabel('Inflammation')
xlabel('Day of trial')
```

The script now allows us to create all 3 plots with a single command: `plot_daily_average`.

We can ask matlab to save the image too using the `print` command.
In order to maintain an organised project we'll save the images
in the `results` directory:

```matlab
% Save plot in 'results' folder as png image:
print('results/daily_average_01','-dpng')
```

:::::::::::::::::::::::::::::::::::::::::  callout

**Note:**

If you have a printer configured on your computer the `print` command may send it to the printer.
You can instead use `saveas`, which requiere two arguments: a figure and a filename.
You can get the current figure with `gcf`, and specify the filename with an extension, like so:
```matlab
% Save plot in 'results' folder as png image:
saveas(gcf,'results/daily_average_01.png')
```

::::::::::::::::::::::::::::::::::::::::::::::::::

When saving plots to disk,
it's sometimes useful to turn off their visibility as MATLAB plots them.
For example, we might not want to view (or spend time closing) the figures in MATLAB, and
not displaying the figures could make the script run faster.

Let's add a couple of lines of code to do this.

We can ask MATLAB to create an empty figure window without
displaying it by setting its `'visible'` property to `'off'`, like so:

```matlab
figure('visible', 'off')
```

When we do this, we have to be careful to manually "close" the figure
after we are doing plotting on it - the same as we would "close"
an actual figure window if it were open:

```matlab
close()
```

Adding these two lines, our finished script looks like this:

```matlab
% PLOT_DAILY_AVERAGE   Saves plot of daily average, max and min inflammation accross patients.

% Load patient data
patient_data = readmatrix('data/inflammation-01.csv');

figure('visible', 'off')

% Plot average inflammation per day
subplot(1, 3, 1)
plot(mean(patient_data, 1))
title('Daily average inflammation')
xlabel('Day of trial')
ylabel('Inflammation')

% Plot max inflammation per day
subplot(1, 3, 2)
plot(max(patient_data, [], 1))
title('Max')
ylabel('Inflammation')
xlabel('Day of trial')

% Plot min inflammation per day
subplot(1, 3, 3)
plot(min(patient_data, [], 1))
title('Min')
ylabel('Inflammation')
xlabel('Day of trial')


% Save plot in 'results' folder as png image:
saveas(gcf,'results/daily_average_01.png')

close()
```

The scripts we've written make regenerating plots easier,
and looking at individual patient's data much simpler,
but we still need to open the script, change the patient number, save, and run.
In contrast, when we have used *functions* we can *provide* arguments,
which are then used to do something. So, can we create our own *functions*?




:::::::::::::::::::::::::::::::::::::::: keypoints

- "Save MATLAB code in files with a `.m` suffix."

::::::::::::::::::::::::::::::::::::::::::::::::::
