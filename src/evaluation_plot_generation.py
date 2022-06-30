"""
Contains code that is used to generate all analysis and evaluation plots.

Parameters are described at the top of the file, after the imports.
"""
import os
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda
from scipy.linalg import block_diag

from visualization_utils import visualize_scenario_data_batch
from iae import TrackerIAE
from memekf import TrackerMEMEKF
from metrics import gwd_full_state
from models.multi_channel_elliptical_nn import MultiChannelEllipticalNN
from nn_ett_tracker import TrackerNN
from scenario_generation import get_slow_turn_data
from matplotlib import rcParams

# ========================
# Figure Saving options
rcParams["savefig.format"] = 'png'

# Font Styling if desired
rcParams.update({'font.size': 15})

# ========================
# PARAMETERS FOR THIS FILE
# ========================
# Output:
# Bool: Whether to save to file
SAVE_PLOTS_TO_FILE = True
# Str: Path to folder in which plots are saved (if SAVE_PLOTS_TO_FILE=True)
PATH_TO_OUTPUT_FOLDER = "../output/plot_generation/"

# Bool: Whether the CNN should be run on GPU
RUN_CNN_ON_GPU = torch.cuda.is_available()

# select which plots should be generated when running the file:
#   simply comment out any lines you don't want to generate
PLOTS_TO_GENERATE = [
    "example trajectory",
    "grid based diff",
    "scaling over object sizes",
    "error over time steps",
    "runtime over settings",
    "error over init noise"
    "error over outliers"
]

# every general "default" parameter should be taken from this dict:
GENERAL_PARAMETERS = {
    "R": np.eye(2) * 0.5,
    "lambda": 8,
    "time_step_length": 1,
    "gw_ix": 0,
    "n_steps_in_scenario": 50,
    "number_of_runs_per_setting": 100,
    "obj_length": 4.7,
    "obj_width": 1.7
}


# ------------------------
def plot_example_trajectory(tracker, R, poisson_lambda, RNG=None, time_step_length=None):
    """Run for the given parameters and create a plot of an example trajectory of one method"""
    RNG = np.random.default_rng() if RNG is None else RNG

    time_step_length = time_step_length if time_step_length is not None else GENERAL_PARAMETERS["time_step_length"]

    data = get_slow_turn_data(time_step_length=time_step_length,
                              poisson_lambda=poisson_lambda,
                              R=R,
                              initial_velocity=5,
                              min_measurements=1,
                              RNG=RNG,
                              length=GENERAL_PARAMETERS["obj_length"],
                              width=GENERAL_PARAMETERS["obj_width"],
                              n_steps=50)
    n_steps = len(data)

    tracker = deepcopy(tracker)

    # run tracker
    states = []
    times_per_time_step = []
    for i in range(n_steps):
        Z = data[i]["measurements"]
        start_time = time.time()
        tracker.predict()
        tracker.update(Z)
        times_per_time_step.append(time.time() - start_time)
        states.append(tracker.get_state())
    print("Time analysis:")
    print(f"\tAverage Processing Time: {np.average(times_per_time_step):.4f}s")
    print(f"\tMinimum Processing Time: {np.min(times_per_time_step):.4f}s")
    print(f"\tMaximum Processing Time: {np.max(times_per_time_step):.4f}s")
    # reformat states
    states = np.vstack(states)

    # calculate squared Gauss-Wasserstein error
    gwd_list = []
    for i in range(n_steps):
        gwd_list.append(gwd_full_state(deepcopy(states[i]), deepcopy(data[i]["gt"]), return_parts=True))
    gwd_list = np.array(gwd_list)

    # visualize on entire batch of data
    plt.clf()
    visualize_scenario_data_batch(data=data, track=states, inset_lower_step_id=0, inset_upper_step_id=49,
                                  show_plot=not SAVE_PLOTS_TO_FILE, disable_title=True, show_centers=False,
                                  alpha_measurements=0.75, s_marker=1, s_marker_zoom=5, show_inset_titles=False)
    if SAVE_PLOTS_TO_FILE:
        plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER, f"trajectory and zoomed time steps"),
                    bbox_inches='tight')
        plt.clf()
        print("Saved 'trajectory and zoomed time steps'")
    return gwd_list


def grid_based_diff(ref_tracker,
                    trackers,
                    noise_scaling_parameters,
                    poisson_lambda_options,
                    number_of_runs_per_setting=None,
                    n_steps_in_scenario=None,
                    RNG=None,
                    show_individual_performance_plots=False):
    """
    Will show individual images for trackers["ref_tracker"] vs all other trackers.
    ref_tracker must be a key to the dict "trackers"
    """
    if number_of_runs_per_setting is None:
        number_of_runs_per_setting = GENERAL_PARAMETERS["number_of_runs_per_setting"]
    if n_steps_in_scenario is None:
        n_steps_in_scenario = GENERAL_PARAMETERS["n_steps_in_scenario"]
    RNG = RNG if RNG is not None else np.random.default_rng()

    # Index for GW dist: 0 = full GW // 1 = loc part // 2 = shape part
    gw_ix = GENERAL_PARAMETERS["gw_ix"]

    data_list = np.full((len(noise_scaling_parameters), len(poisson_lambda_options), number_of_runs_per_setting), {})
    print("Beginning Data Generation")
    for uid_noise_factor, next_noise_factor in enumerate(noise_scaling_parameters):
        next_R = np.eye(2) * next_noise_factor
        for uid_lambda, next_poisson_lambda in enumerate(poisson_lambda_options):
            for run_number in range(number_of_runs_per_setting):
                data_list[uid_noise_factor, uid_lambda, run_number] = get_slow_turn_data(time_step_length=1,
                                                                                         poisson_lambda=next_poisson_lambda,
                                                                                         R=next_R,
                                                                                         initial_velocity=5,
                                                                                         min_measurements=3,
                                                                                         RNG=RNG,
                                                                                         length=GENERAL_PARAMETERS[
                                                                                             "obj_length"],
                                                                                         width=GENERAL_PARAMETERS[
                                                                                             "obj_width"],
                                                                                         n_steps=n_steps_in_scenario)
    print("Finished Data Generation")

    # prep tracker performance dict
    tracker_names = trackers.keys()
    tracker_performances = {}
    for tracker_id in tracker_names:
        print(f"Running for tracker '{tracker_id}'")
        # tracker performances are 4D arrays:
        #   dim1: which noise scaling parameter was used
        #   dim2: which poisson lambda choice was used
        #   dim3: which monte carlo run this was recorded in
        #   dim4: which step in the individual run this error belonged to
        tracker_performances[tracker_id] = np.zeros((len(noise_scaling_parameters),
                                                     len(poisson_lambda_options),
                                                     number_of_runs_per_setting,
                                                     n_steps_in_scenario))

        # run over all possible noise factors
        for uid_noise_factor, next_noise_factor in enumerate(noise_scaling_parameters):
            next_R = np.eye(2) * next_noise_factor
            for uid_lambda, next_poisson_lambda in enumerate(poisson_lambda_options):
                for run_number in range(number_of_runs_per_setting):
                    data = data_list[uid_noise_factor, uid_lambda, run_number]
                    n_steps = len(data)
                    # Run tracker on data
                    tracker = trackers[tracker_id]()
                    tracker.set_R(next_R)
                    # run tracker
                    states = []
                    for i in range(n_steps):
                        Z = data[i]["measurements"]
                        tracker.predict()
                        tracker.update(Z)
                        states.append(tracker.get_state())
                    # reformat states
                    states = np.vstack(states)
                    # calculate squared Gauss-Wasserstein error
                    gwd_list = []
                    for i in range(n_steps):
                        gwd_list.append(gwd_full_state(deepcopy(states[i]), deepcopy(data[i]["gt"]), return_parts=True))
                    gwd_list = np.array(gwd_list).astype(float)
                    tracker_performances[tracker_id][uid_noise_factor, uid_lambda, run_number, :] += gwd_list[:, gw_ix]

    # evaluation plots
    error_per_setting = {}
    # calc and plot each individual result image
    for tracker_ix, tracker_id in enumerate(tracker_names):
        # first: average on a per-step basis over all runs (removing axis 2)
        performance_average_over_runs = np.average(tracker_performances[tracker_id], axis=2)
        # second: average over all steps in the scenario (removing axis 2, which was previously 3)
        error_per_setting[tracker_id] = np.average(performance_average_over_runs, axis=2)

    maximum_overall_error = np.max([np.max(error_per_setting[tracker_id]) for tracker_id in tracker_names])
    if show_individual_performance_plots:
        for tracker_ix, tracker_id in enumerate(tracker_names):
            # plot this
            plt.clf()
            # plt.imshow(error_per_setting[tracker_id], vmin=0, vmax=maximum_overall_error)
            plt.imshow(error_per_setting[tracker_id])
            plt.title(f"{tracker_id} Performance")
            plt.xlabel("$\lambda$ (Measurement Rate)")
            plt.xticks(np.arange(start=0, stop=len(poisson_lambda_options)))
            plt.gca().set_xticklabels(poisson_lambda_options)
            plt.ylabel("Noise $r$ (for $\mathbf{R} = \mathbf{I} \cdot r)$")
            plt.yticks(np.arange(start=0, stop=len(noise_scaling_parameters)))
            plt.gca().set_yticklabels(noise_scaling_parameters)
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.ax.get_yaxis().labelpad = 5
            cbar.ax.set_ylabel("Squared GWD / m$^2$", rotation=90)
            if SAVE_PLOTS_TO_FILE:
                plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER, f"grid performance {str(tracker_id).replace('*', '')}"),
                            bbox_inches='tight')
                plt.clf()
            else:
                plt.show()

    # next: plot ref_tracker vs all other trackers individually
    for tracker_ix, tracker_id in enumerate(tracker_names):
        if tracker_id == ref_tracker:
            # skip reference tracker (can't compare with it self)
            continue
        plt.clf()
        error_comparison = error_per_setting[ref_tracker] - error_per_setting[tracker_id]
        cmapval = np.max(np.abs(error_comparison))
        plt.imshow(error_comparison, cmap="seismic", vmin=-cmapval, vmax=cmapval)
        plt.title(f"[{ref_tracker}] - [{tracker_id}]")
        plt.xlabel("$\lambda$ (Measurement Rate)")
        plt.xticks(np.arange(start=0, stop=len(poisson_lambda_options)))
        plt.gca().set_xticklabels(poisson_lambda_options)
        plt.ylabel("Noise $r$ (for $\mathbf{R} = \mathbf{I} \cdot r)$")
        plt.yticks(np.arange(start=0, stop=len(noise_scaling_parameters)))
        plt.gca().set_yticklabels(noise_scaling_parameters)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().labelpad = 5
        cbar.ax.set_ylabel(f"Difference in squared GWD / m$^2$\nHigh Values: {tracker_id} better", rotation=90)
        # show
        if SAVE_PLOTS_TO_FILE:
            plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER,
                                     f"grid diff [{ref_tracker}] - [{str(tracker_id).replace('*', '')}]"),
                        bbox_inches='tight')
            plt.clf()
        else:
            plt.show()


def evaluate_object_sizes(tracker_base,
                          n_runs_per=5,
                          sizes=None,
                          use_flips=True,
                          poisson_lambda=8,
                          R=None,
                          seed=None,
                          image_max_sizes=None):
    """Run experiment evaluating different object sizes for for different scaling factors, i.e., for different
    image_max_sizes."""
    RNG = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    image_max_sizes = image_max_sizes if image_max_sizes is not None else [10]
    gw_ix = 0
    n_steps = 50
    # ensure sizes is defined
    if sizes is None:
        sizes = [[3, 1], [4.7, 1.7], [6, 2.8]]

    R = R if R is not None else GENERAL_PARAMETERS["R"]

    # add in flipped version of each size
    if use_flips:
        doubled_sizes = []
        for size in sizes:
            doubled_sizes.append(size)
            doubled_sizes.append(size[::-1])
        sizes = doubled_sizes

    # run
    plt.clf()
    worst_performance = -1
    for image_max_size in image_max_sizes:
        performances = np.zeros((len(sizes), n_runs_per, n_steps))
        print(f"---Next Scaling Factor ?/{len(image_max_sizes)}")
        for uid_size, next_size in enumerate(sizes):
            obj_length, obj_width = next_size
            print(f"\tRunning for size {uid_size+1}/{len(sizes)}")
            for run_number in range(n_runs_per):
                # generate data and tracker
                tracker: TrackerNN = deepcopy(tracker_base)
                tracker._maximum_scaleable_distance = image_max_size
                data = get_slow_turn_data(time_step_length=1,
                                          poisson_lambda=poisson_lambda,
                                          R=R,
                                          initial_velocity=5,
                                          min_measurements=3,
                                          RNG=RNG,
                                          length=obj_length,
                                          width=obj_width,
                                          n_steps=n_steps)
                tracker.set_R(R)

                # run tracker
                states = []
                for i in range(n_steps):
                    Z = data[i]["measurements"]
                    tracker.predict()
                    tracker.update(Z)
                    states.append(tracker.get_state())
                # reformat states
                states = np.vstack(states)
                # calculate squared Gauss-Wasserstein error
                gwd_list = []
                for i in range(n_steps):
                    gwd_list.append(gwd_full_state(deepcopy(states[i]), deepcopy(data[i]["gt"]), return_parts=True))
                gwd_list = np.array(gwd_list)
                performances[uid_size, run_number, :] += gwd_list[:, gw_ix]

        # analyse and plot
        # average across all runs
        performances = np.average(performances, axis=1)
        # average across all steps
        performances = np.average(performances, axis=1)

        plt.plot(performances, label=f"Scaling Factor = {image_max_size}", marker='x', linestyle=':')
        worst_performance = max(worst_performance, np.max(performances))

    # summarize plot
    plt.legend()
    plt.xlabel("Object Dimensions / m")
    plt.ylabel("Average squared GWD / m$^2$")
    plt.ylim(0, worst_performance + 0.2)
    plt.xticks(ticks=list(range(len(performances))), labels=[f"{s[0]:.1f}x{s[1]:.1f}" for s in sizes],
               rotation="vertical")
    if SAVE_PLOTS_TO_FILE:
        plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER, f"scaling factor errors over sizes"),
                    bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def error_over_time(trackers, R, poisson_lambda, n_runs=None, RNG=None):
    """Plot the (average) error of all trackers over time steps in the example trajectory."""
    n_steps_in_scenario = GENERAL_PARAMETERS["n_steps_in_scenario"]
    n_runs = n_runs if n_runs is not None else GENERAL_PARAMETERS["number_of_runs_per_setting"]
    RNG = RNG if RNG is not None else np.random.default_rng()
    data_list = []
    for i in range(n_runs):
        data_list.append(get_slow_turn_data(time_step_length=GENERAL_PARAMETERS["time_step_length"],
                                            poisson_lambda=poisson_lambda,
                                            R=R,
                                            initial_velocity=5,
                                            min_measurements=1,
                                            RNG=RNG,
                                            length=GENERAL_PARAMETERS["obj_length"],
                                            width=GENERAL_PARAMETERS["obj_width"],
                                            n_steps=n_steps_in_scenario)
                         )
    tracker_names = list(trackers.keys())
    tracker_performances = {}
    for tracker_id in tracker_names:
        print(f"Running for tracker '{tracker_id}'")
        tracker_performances[tracker_id] = np.zeros((n_runs,
                                                     n_steps_in_scenario))
        times_per_time_step = []
        for run_number, data in enumerate(data_list):
            tracker = trackers[tracker_id]()
            tracker.set_R(R)
            n_steps = len(data)
            # run tracker
            states = []

            # actual run:

            for i in range(n_steps):
                start_time = time.time()
                Z = data[i]["measurements"]
                tracker.predict()
                tracker.update(Z)
                states.append(tracker.get_state())
                times_per_time_step.append(time.time() - start_time)

            # reformat states
            states = np.vstack(states)
            # calculate squared Gauss-Wasserstein error
            gwd_list = []
            for i in range(n_steps):
                gwd_list.append(gwd_full_state(deepcopy(states[i]), deepcopy(data[i]["gt"]), return_parts=True))
            gwd_list = np.array(gwd_list).astype(float)
            tracker_performances[tracker_id][run_number, :] += gwd_list[:, GENERAL_PARAMETERS["gw_ix"]]

        # average over runs
        tracker_performances[tracker_id] = np.average(tracker_performances[tracker_id], axis=0)

        # also: quickly print Information about timings
        print(f"\tAverage Processing Time: {np.average(times_per_time_step):.4f}s")
        print(f"\tMinimum Processing Time: {np.min(times_per_time_step):.4f}s")
        print(f"\tMaximum Processing Time: {np.max(times_per_time_step):.4f}s")
    # PLOT
    plt.clf()
    for tracker_id in tracker_names:
        plt.plot(tracker_performances[tracker_id], label=tracker_id)
    plt.legend(loc="upper right")
    plt.xlabel("Time Step")
    plt.ylabel("Average squared GWD / m$^2$")
    if SAVE_PLOTS_TO_FILE:
        plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER, f"error over time steps"),
                    bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def error_over_init_noise(trackers, R, poisson_lambda, P_init_scaling, n_runs=None, RNG=None):
    """Plot different errors over initial noise choices for all trackers"""
    n_scaling_options = len(P_init_scaling)
    n_steps_in_scenario = GENERAL_PARAMETERS["n_steps_in_scenario"]
    n_runs = n_runs if n_runs is not None else GENERAL_PARAMETERS["number_of_runs_per_setting"]
    RNG = RNG if RNG is not None else np.random.default_rng()
    data_list = []
    for i in range(n_runs):
        next_list = []
        for j in range(n_scaling_options):
            next_list.append(get_slow_turn_data(time_step_length=GENERAL_PARAMETERS["time_step_length"],
                                                poisson_lambda=poisson_lambda,
                                                R=R,
                                                initial_velocity=5,
                                                min_measurements=1,
                                                RNG=RNG,
                                                length=GENERAL_PARAMETERS["obj_length"],
                                                width=GENERAL_PARAMETERS["obj_width"],
                                                n_steps=n_steps_in_scenario)
                             )
        data_list.append(next_list)
    tracker_names = list(trackers.keys())
    tracker_performances = {}
    for tracker_id in tracker_names:
        print(f"Running for tracker '{tracker_id}'")
        tracker_performances[tracker_id] = np.zeros((n_scaling_options,
                                                     n_runs,
                                                     n_steps_in_scenario))
        times_per_time_step = []
        for scaling_number, p_scale in enumerate(P_init_scaling):
            for run_number in range(n_runs):
                data = data_list[run_number][scaling_number]
                tracker = trackers[tracker_id](P_scale=p_scale)
                tracker.set_R(R)
                n_steps = len(data)
                # run tracker
                states = []

                # actual run:

                for i in range(n_steps):
                    start_time = time.time()
                    Z = data[i]["measurements"]
                    tracker.predict()
                    tracker.update(Z)
                    states.append(tracker.get_state())
                    times_per_time_step.append(time.time() - start_time)

                # reformat states
                states = np.vstack(states)
                # calculate squared Gauss-Wasserstein error
                gwd_list = []
                for i in range(n_steps):
                    gwd_list.append(gwd_full_state(deepcopy(states[i]), deepcopy(data[i]["gt"]), return_parts=True))
                gwd_list = np.array(gwd_list).astype(float)
                tracker_performances[tracker_id][scaling_number, run_number, :] += gwd_list[:,
                                                                                   GENERAL_PARAMETERS["gw_ix"]]

        # average over steps
        tracker_performances[tracker_id] = np.average(tracker_performances[tracker_id], axis=2)

        # average over runs
        tracker_performances[tracker_id] = np.average(tracker_performances[tracker_id], axis=1)

    # PLOT
    plt.clf()
    for tracker_id in tracker_names:
        plt.plot(P_init_scaling, tracker_performances[tracker_id], label=tracker_id, marker='x', linestyle='--')
    plt.legend(loc="upper right")
    plt.xlabel("$p$ for $\mathbf{P} = \mathbf{I} \cdot p$")
    plt.ylabel("Average squared GWD / m$^2$")
    if SAVE_PLOTS_TO_FILE:
        plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER, f"error over init noise"),
                    bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def error_over_outliers(trackers, R, poisson_lambda, n_outliers, n_runs=None, RNG=None):
    """Plot experiment testing the effect that increasing numbers of outliers have on measurement performance"""
    n_outlier_option_count = len(n_outliers)
    n_steps_in_scenario = GENERAL_PARAMETERS["n_steps_in_scenario"]
    n_runs = n_runs if n_runs is not None else GENERAL_PARAMETERS["number_of_runs_per_setting"]
    RNG = RNG if RNG is not None else np.random.default_rng()
    data_list = []
    for i in range(n_runs):
        next_list = []
        for j in range(n_outlier_option_count):
            next_list.append(get_slow_turn_data(time_step_length=GENERAL_PARAMETERS["time_step_length"],
                                                poisson_lambda=poisson_lambda,
                                                R=R,
                                                initial_velocity=5,
                                                min_measurements=1,
                                                RNG=RNG,
                                                length=GENERAL_PARAMETERS["obj_length"],
                                                width=GENERAL_PARAMETERS["obj_width"],
                                                n_steps=n_steps_in_scenario,
                                                n_outliers=n_outliers[j])
                             )
        data_list.append(next_list)
    tracker_names = list(trackers.keys())
    tracker_performances = {}
    for tracker_id in tracker_names:
        print(f"Running for tracker '{tracker_id}'")
        tracker_performances[tracker_id] = np.zeros((n_outlier_option_count,
                                                     n_runs,
                                                     n_steps_in_scenario))
        times_per_time_step = []
        for outlier_id in range(n_outlier_option_count):
            for run_number in range(n_runs):
                data = data_list[run_number][outlier_id]
                tracker = trackers[tracker_id]()
                tracker.set_R(R)
                n_steps = len(data)
                # run tracker
                states = []

                # actual run:

                for i in range(n_steps):
                    start_time = time.time()
                    Z = data[i]["measurements"]
                    tracker.predict()
                    tracker.update(Z)
                    states.append(tracker.get_state())
                    times_per_time_step.append(time.time() - start_time)

                # reformat states
                states = np.vstack(states)
                # calculate squared Gauss-Wasserstein error
                gwd_list = []
                for i in range(n_steps):
                    gwd_list.append(gwd_full_state(deepcopy(states[i]), deepcopy(data[i]["gt"]), return_parts=True))
                gwd_list = np.array(gwd_list).astype(float)
                tracker_performances[tracker_id][outlier_id, run_number, :] += gwd_list[:, GENERAL_PARAMETERS["gw_ix"]]

        # average over steps
        tracker_performances[tracker_id] = np.average(tracker_performances[tracker_id], axis=2)

        # average over runs
        tracker_performances[tracker_id] = np.average(tracker_performances[tracker_id], axis=1)

    # PLOT
    plt.clf()
    for tracker_id in tracker_names:
        plt.plot(n_outliers, tracker_performances[tracker_id], label=tracker_id, marker='x', linestyle='--')
    plt.legend(loc="upper left")
    plt.xlabel("Number of Outliers")
    plt.ylabel("Average squared GWD / m$^2$")
    if SAVE_PLOTS_TO_FILE:
        plt.savefig(os.path.join(PATH_TO_OUTPUT_FOLDER, f"error over outliers"),
                    bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def runtime_over_settings(trackers, poisson_lambda_options=None, RNG=None):
    """Evaluate the runtime for a variety of measurement rates across multiple trackers"""
    number_of_runs_per_setting = GENERAL_PARAMETERS["number_of_runs_per_setting"]
    if poisson_lambda_options is None:
        poisson_lambda_options = [3, 8, 50]

    RNG = RNG if RNG is not None else np.random.default_rng()
    R = GENERAL_PARAMETERS["R"]
    n_steps_in_scenario = GENERAL_PARAMETERS["n_steps_in_scenario"]

    data_list = np.full((len(poisson_lambda_options), number_of_runs_per_setting), {})
    for uid_lambda, next_poisson_lambda in enumerate(poisson_lambda_options):
        for run_number in range(number_of_runs_per_setting):
            data_list[uid_lambda, run_number] = get_slow_turn_data(time_step_length=1,
                                                                   poisson_lambda=next_poisson_lambda,
                                                                   R=R,
                                                                   initial_velocity=5,
                                                                   min_measurements=3,
                                                                   RNG=RNG,
                                                                   length=GENERAL_PARAMETERS[
                                                                       "obj_length"],
                                                                   width=GENERAL_PARAMETERS[
                                                                       "obj_width"],
                                                                   n_steps=n_steps_in_scenario)

    # prep tracker performance dict
    tracker_names = trackers.keys()
    tracker_performances = {}
    for tracker_id in tracker_names:
        print(f"Running for tracker '{tracker_id}'")
        # tracker performances are 4D arrays:
        #   dim1: which noise scaling parameter was used
        #   dim2: which poisson lambda choice was used
        #   dim3: which monte carlo run this was recorded in
        #   dim4: which step in the individual run this error belonged to
        tracker_performances[tracker_id] = np.zeros((len(poisson_lambda_options),
                                                     number_of_runs_per_setting,
                                                     n_steps_in_scenario))

        # run over all possible noise factors
        for uid_lambda, next_poisson_lambda in enumerate(poisson_lambda_options):
            for run_number in range(number_of_runs_per_setting):
                data = data_list[uid_lambda, run_number]
                n_steps = len(data)
                # Run tracker on data
                tracker = trackers[tracker_id]()
                tracker.set_R(R)
                # run tracker
                for i in range(n_steps):
                    Z = data[i]["measurements"]
                    start_time = time.time()
                    tracker.predict()
                    tracker.update(Z)
                    tracker_performances[tracker_id][uid_lambda, run_number, i] = (time.time() - start_time) * 1000

    # nicely formatted print:
    print("\nGeneral Information:")
    print(f"\tSettings (lambdas): {poisson_lambda_options}")
    print(f"\tRuns per setting  : {number_of_runs_per_setting}")
    print(f"\tSteps per run     : {n_steps_in_scenario}")
    for uid_lambda, next_poisson_lambda in enumerate(poisson_lambda_options):
        print(f"Measurement Rate Lambda={next_poisson_lambda}")
        for tracker_id in tracker_names:
            perf = np.average(tracker_performances[tracker_id][uid_lambda, :, :])
            print(f"\tTracker {tracker_id:{max([len(x) for x in tracker_names])}s}: {perf:3.4f} / ms")


if __name__ == '__main__':
    RNG = np.random.default_rng()

    # --- SET UP ALL TRACKERS
    # set up initial values
    time_step_length = 1
    x_init = np.array([0, 0, 5, 0])
    P_init = block_diag(np.eye(2), np.eye(2))
    Q = np.diag([0.05, 0.05, 0.2, 0.2])
    alpha_init = 0
    alpha_var_init = np.pi / 2
    R_alpha = np.pi / 64
    Q_alpha = np.pi / 4

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # decay factor
    gamma = 0.95

    # neural network loading
    blurry_checkpoint_path = "../models/MultiChannelEllipticalNN/v1.ckpt"

    model_blurry = MultiChannelEllipticalNN.load_from_checkpoint(blurry_checkpoint_path)
    model_blurry.eval()
    # optionally, move model to GPU:
    if RUN_CNN_ON_GPU:
        model_blurry.move_to_cuda()

    # perform a single prediction on an empty image ONCE
    #   as the network is slow when used first
    model_blurry(np.zeros((2, 300, 300)))


    # The following functions can be collected in a dictionary to get freshly initialized trackers, optionally with
    # adapted initial covariance.
    # Each function returns a new instance of the corresponding tracker, initialized with noise P_init * P_scale.
    # If P_scale = None, the tracker is initialized using P_init
    # P_init should be defined in the beginning of this "main" block - e.g. set to np.eye(4)
    def get_nnett(P_scale=None):
        """Return a fresh NN-ETT tracker instance"""
        P_init_to_use = P_init * P_scale if P_scale is not None else P_init
        noisy_x = RNG.multivariate_normal(mean=x_init, cov=P_init_to_use)
        tracker_nnett = TrackerNN(x_init=noisy_x,
                                  P_init=P_init_to_use,
                                  R=GENERAL_PARAMETERS["R"],
                                  Q=Q,
                                  model=model_blurry,
                                  image_size=300,
                                  maximum_scaleable_distance=8,
                                  gamma=gamma,
                                  update_kinematics_from_statistics=True,
                                  T=time_step_length)
        return tracker_nnett


    def get_memekf(P_scale=None):
        """Return a fresh MEM-EKF* tracker instance"""
        m_init = [x_init[0],
                  x_init[1],
                  np.linalg.norm(x_init[2:])
                  ]
        memekf_Cm_init = P_init[:3, :3] * P_scale if P_scale is not None else P_init[:3, :3]

        noisy_m = RNG.multivariate_normal(mean=m_init, cov=memekf_Cm_init)
        memekf_p_init = [alpha_init, 2, 1]
        memekf_Cp_init = np.diag([0.01, 0.005, 0.005])
        memekf_tracker = TrackerMEMEKF(m_init=noisy_m,
                                       p_init=memekf_p_init,
                                       C_m_init=memekf_Cm_init,
                                       C_p_init=memekf_Cp_init,
                                       R=GENERAL_PARAMETERS["R"],
                                       Q=np.diag([2.5, 2.5, 0.75]),
                                       Q_extent=np.diag([0.05, 1e-8, 1e-8]))
        return memekf_tracker


    def get_iae(P_scale=None):
        """Return a fresh IAE tracker instance"""
        P_init_to_use = P_init * P_scale if P_scale is not None else P_init
        noisy_x = RNG.multivariate_normal(mean=x_init, cov=P_init_to_use)
        iae_tracker = TrackerIAE(x_init=noisy_x,
                                 P_init=P_init_to_use,
                                 l_init=[2, 1],
                                 c_init=[5, 5],
                                 R=GENERAL_PARAMETERS["R"],
                                 Q=Q)
        return iae_tracker

    # create overall dict, mapping IDs to the previously defined functions
    tracker_dict = {
        "Proposed Method": get_nnett,
        "IAE": get_iae,
        "MEM-EKF*": get_memekf,
    }

    # ---
    # RUN ALL FUNCTIONS (that were included in the PLOTS_TO_GENERATE settings)
    if "example trajectory" in PLOTS_TO_GENERATE:
        key = [k for k in tracker_dict.keys()][0]
        print("\n---Plot Example Trajectory for method {key}:")
        plot_example_trajectory(tracker_dict[key](),
                                R=GENERAL_PARAMETERS["R"],
                                poisson_lambda=GENERAL_PARAMETERS["lambda"],
                                RNG=np.random.default_rng(1234567890))

    if "scaling over object sizes" in PLOTS_TO_GENERATE:
        print("\n---Object Sizes:")
        key = [k for k in tracker_dict.keys()][0]
        evaluate_object_sizes(tracker_base=tracker_dict[key](),
                              n_runs_per=GENERAL_PARAMETERS["number_of_runs_per_setting"],
                              use_flips=False,
                              sizes=[
                                  [0.3, 0.1], [0.7, 0.6], [1.3, 0.7], [2.5, 2.2],
                                  [4.7, 1.7], [6, 2], [8, 6], [10, 3], [12, 7]
                              ],
                              image_max_sizes=[5, 10])

    if "grid based diff" in PLOTS_TO_GENERATE:
        print("\n---Grid based Diff:")
        key = [k for k in tracker_dict.keys()][0]
        grid_based_diff(ref_tracker=key,
                        trackers=tracker_dict,
                        noise_scaling_parameters=[0.01, 0.25, 0.5, 1, 2, 4],
                        poisson_lambda_options=[1, 3, 8, 15, 50, 75],
                        number_of_runs_per_setting=GENERAL_PARAMETERS["number_of_runs_per_setting"],
                        show_individual_performance_plots=True)

    if "error over time steps" in PLOTS_TO_GENERATE:
        print("\n---Error over Time:")
        error_over_time(trackers=tracker_dict,
                        R=GENERAL_PARAMETERS["R"],
                        poisson_lambda=GENERAL_PARAMETERS["lambda"],
                        n_runs=GENERAL_PARAMETERS["number_of_runs_per_setting"])

    if "runtime over settings" in PLOTS_TO_GENERATE:
        print("\n---Runtime over Settings (print):")
        runtime_over_settings(trackers=tracker_dict,
                              poisson_lambda_options=[3, 20, 50, 75])

    if "error over init noise" in PLOTS_TO_GENERATE:
        print("\n---Error over Initialization Noise")
        error_over_init_noise(trackers=tracker_dict,
                              R=GENERAL_PARAMETERS["R"],
                              poisson_lambda=GENERAL_PARAMETERS["lambda"],
                              n_runs=GENERAL_PARAMETERS["number_of_runs_per_setting"],
                              P_init_scaling=[0.1, 0.5, 1, 3, 5, 10])

    if "error over outliers" in PLOTS_TO_GENERATE:
        print("\n---Error over Outlier Measurements")
        lambda_outlier = GENERAL_PARAMETERS["lambda"] * 1
        print(f"Using adapted poisson_lambda = {lambda_outlier}")
        error_over_outliers(trackers=tracker_dict,
                            R=GENERAL_PARAMETERS["R"],
                            poisson_lambda=lambda_outlier,
                            n_runs=GENERAL_PARAMETERS["number_of_runs_per_setting"],
                            n_outliers=[0, 2, 4, 8, 16])
    # -------------------
    # FINALIZE (as needed)
    print("")
