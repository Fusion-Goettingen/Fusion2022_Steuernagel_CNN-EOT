import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.animated_visualization_utils import VideoSaver
from src.utils import Rot


def plot_elliptic_extent(m, p, ax=None, est_color='b', color_alpha=1., label="Extent Estimate", linestyle=None,
                         show_center=True):
    """
    Add matplotlib ellipse patch based on location and extent information about vehicle
    :param m: Kinematic information as 4D array [x, y, velocity_x, velocity_y]
    :param p: extent information as 3D array [orientation, length, width]. Orientation in radians.
    :param ax: matplotlib axis to plot on or None (will use .gca() if None)
    :param est_color: Color to plot the ellipse and marker in
    :param color_alpha: Alpha value for plot
    :param label: Label to apply to plot or None to not add a label
    :param linestyle: Linestyle parameter passed to matplotlib
    :param show_center: If True, will additionally add an x for the center location
    """
    if ax is None:
        ax = plt.gca()
    alpha, l1, l2 = p
    alpha = np.rad2deg(alpha)
    # patches.Ellipse takes angle counter-clockwise
    el = patches.Ellipse(xy=m[:2], width=l1, height=l2, angle=alpha, fill=False, color=est_color, label=label,
                         alpha=color_alpha, linestyle=linestyle)
    if show_center:
        ax.scatter(m[0], m[1], c=est_color, marker='x')
    ax.add_patch(el)


def plot_rectangle_extent(m, p, ax=None, est_color='b', color_alpha=1., label="Extent Estimate", linestyle="-",
                          show_center=True):
    """
    Plots a box given as center-center-orientation-length-width
    :param m: Kinematic information as 4D array [x, y, velocity_x, velocity_y]
    :param p: extent information as 3D array [orientation, length, width]. Orientation in radians.
    :param ax: Axis or None. If None, uses the current axis.
    :param color: Color string passed to matplotlib
    :param label: label passed to matplotlib
    :param linestyle: linestyle passed to matplotlib
    """
    if m is None or p is None:
        return
    x, y = m[:2]
    theta, length, width = p
    length, width = length / 2, width / 2
    corners = [
        [-length, -width],
        [-length, width],
        [length, width],
        [length, -width]
    ]
    R = Rot(theta)
    corners = np.array([R @ np.array(c) + [x, y] for c in corners])
    ax = plt.gca() if ax is None else ax
    # only label the first edge
    ax.plot(corners[[0, 1], 0], corners[[0, 1], 1], c=est_color, label=label, linestyle=linestyle, alpha=color_alpha)
    # plot unlabeled edges
    ax.plot(corners[[1, 2], 0], corners[[1, 2], 1], c=est_color, linestyle=linestyle, alpha=color_alpha)
    ax.plot(corners[[2, 3], 0], corners[[2, 3], 1], c=est_color, linestyle=linestyle, alpha=color_alpha)
    ax.plot(corners[[3, 0], 0], corners[[3, 0], 1], c=est_color, linestyle=linestyle, alpha=color_alpha)
    if show_center:
        ax.scatter(x, y, c=est_color, marker='x')


def visualize_scenario_data_batch(data, gt_color='b', track=None, track_color='r', ellipse_alpha=0.7,
                                  disable_title=False, meas_color='r',
                                  inset_lower_step_id=None, inset_upper_step_id=None,
                                  show_plot=True, show_centers=True, alpha_measurements=0.5,
                                  s_marker=None, s_marker_zoom=None, show_inset_titles=True, as_rectangle=False):
    """
    Given measurement data across multiple time steps, visualize the data in batch for all steps
    :param data: List of dict. Each dict contains "time", 7-D "gt" object data and a N,2 shaped array of "measurements"
    :param gt_color: Color to be used for plotting the truth (loc + ellipse)
    :param track: If None, will be ignored. Else should be a (N,7) array of estimated object states
    :param track_color: If track is not None, it will be plotted in this color.
    :param ellipse_alpha: Alpha(transparency) value used for ellipses
    :param as_rectangle: If True, everything is plotted as rectangles rather than ellipses
    """
    plot_extent = plot_elliptic_extent if not as_rectangle else plot_rectangle_extent
    if track is not None:
        assert len(track) == len(data), "Track length ({}) and data length ({}) don't match".format(len(track),
                                                                                                    len(data))
        track = np.array(track)

    for i, scene in enumerate(data):
        gt = scene["gt"]
        measurements = scene["measurements"]
        meas_label = 'Measurements' if i == 0 else None
        extent_label = 'True Extent' if i == 0 else None

        plot_extent(m=gt[:4], p=gt[4:], est_color=gt_color, color_alpha=ellipse_alpha, label=extent_label,
                    show_center=show_centers)
        plt.scatter(measurements[:, 0], measurements[:, 1], alpha=alpha_measurements, marker='o', label=meas_label,
                    s=s_marker)

        if track is not None:
            track_label = "Track Estimate" if i == 0 else None
            plot_extent(m=track[i, :4], p=track[i, 4:],
                        est_color=track_color, color_alpha=ellipse_alpha, label=track_label,
                        show_center=show_centers)

    if not disable_title:
        plt.title("All locations and measurements after {} steps".format(i + 1))
    plt.legend(loc="upper left")
    plt.axis('equal')
    plt.xlabel("$m_1$ / m")
    plt.ylabel("$m_2$ / m")

    inset_size = 0.125

    if inset_lower_step_id is not None:
        inset_lower = plt.axes([.25, .2, inset_size, inset_size])
        scene = data[inset_lower_step_id]
        time = scene["time"]
        gt = scene["gt"]
        measurements = scene["measurements"]
        plot_extent(m=gt[:4], p=gt[4:], est_color=gt_color, color_alpha=ellipse_alpha, label="True Extent")
        plt.scatter(measurements[:, 0], measurements[:, 1], alpha=0.5, marker='o', c='blue', label='Measurements',
                    s=s_marker_zoom)

        if track is not None:
            plot_extent(m=track[inset_lower_step_id, :4], p=track[inset_lower_step_id, 4:],
                        est_color=track_color, color_alpha=ellipse_alpha, label='Est. Extent')

        if show_inset_titles:
            plt.title(f"Zoom: Time Step {inset_lower_step_id + 1}")
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')

    if inset_upper_step_id is not None:
        inset_upper = plt.axes([.55, .69, inset_size, inset_size])
        scene = data[inset_upper_step_id]
        time = scene["time"]
        gt = scene["gt"]
        measurements = scene["measurements"]
        plot_extent(m=gt[:4], p=gt[4:], est_color=gt_color, color_alpha=ellipse_alpha, label="True Extent")
        plt.scatter(measurements[:, 0], measurements[:, 1], alpha=0.5, marker='o', c='blue', label='Measurements',
                    s=s_marker_zoom)

        if track is not None:
            plot_extent(m=track[inset_upper_step_id, :4], p=track[inset_upper_step_id, 4:],
                        est_color=track_color, color_alpha=ellipse_alpha, label='Est. Extent')

        if show_inset_titles:
            plt.title(f"Zoom: Time Step {inset_upper_step_id + 1}")
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')

    if show_plot:
        plt.show()


def visualize_scenario_data_individual(data, gt_color='b', track=None, track_color='r', meas_color='r',
                                       ellipse_alpha=0.9, individual_save_file=None, title_prefix="",
                                       pause_interval=0.1, as_rectangle=False):
    """
    Given measurement data across multiple time steps, visualize each time step individually
    :param data: List of dict. Each dict contains "time", 7-D "gt" object data and a N,2 shaped array of "measurements"
    :param gt_color: Color to be used for plotting the true object state (loc + ellipse)
    :param meas_color: Color to be used for scattering the measurements
    :param individual_save_file: If not None and visualize_steps_individually==True, then a video of each individual
    step will be saved to this file.
    :param title_prefix: Prefix added to all titles
    :param as_rectangle: If True, everything is plotted as rectangles rather than ellipses
    """
    plot_extent = plot_elliptic_extent if not as_rectangle else plot_rectangle_extent
    if individual_save_file is not None:
        video_saver = VideoSaver(verbose=True)
        plt.clf()
        plt.text(0, 0, "Consider going into fullscreen\nin order to record the video\nin fullscreen", size=25)
        plt.pause(1)

    for i, scene in enumerate(data):
        plt.cla()
        time = scene["time"]
        gt = scene["gt"]
        measurements = scene["measurements"]
        plt.title(f"{title_prefix}Step #{i + 1} at Time: {time}s")
        plot_extent(m=gt[:4], p=gt[4:], est_color=gt_color, color_alpha=ellipse_alpha, label="True Extent")
        plt.scatter(measurements[:, 0], measurements[:, 1], alpha=0.5, marker='o', c=meas_color, label='Measurements')

        if track is not None:
            plot_extent(m=track[i, :4], p=track[i, 4:],
                        est_color=track_color, color_alpha=ellipse_alpha, label='Est. Extent', linestyle='--')

        # show plot
        plt.legend(loc="upper left")
        plt.axis('equal')
        plt.margins(2, 2)
        zoom = max(gt[5:]) * 2.5
        plt.xlim([gt[0] - zoom, gt[0] + zoom])
        plt.ylim([gt[1] - zoom, gt[1] + zoom])
        plt.draw()
        plt.pause(pause_interval)
        if individual_save_file is not None:
            video_saver.add_frame(plt.gcf())
    if individual_save_file is not None:
        video_saver.generate_video(filename=individual_save_file, fps=60, repeat_frame_count=40)


def visualize_scenario_data_individual_multi(data, gt_color='b', track_dict=None, meas_color='r',
                                             ellipse_alpha=0.9, individual_save_file=None, title_prefix="",
                                             tracker_color_dict=None, pause_interval=0.1, as_rectangle=False):
    """
    Given measurement data across multiple time steps, visualize each time step individually
    :param data: List of dict. Each dict contains "time", 7-D "gt" object data and a N,2 shaped array of "measurements"
    :param gt_color: Color to be used for plotting the true object state (loc + ellipse)
    :param meas_color: Color to be used for scattering the measurements
    :param individual_save_file: If not None and visualize_steps_individually==True, then a video of each individual
    step will be saved to this file.
    :param title_prefix: Prefix added to all titles
    :param as_rectangle: If True, everything is plotted as rectangles rather than ellipses
    """
    plot_extent = plot_elliptic_extent if not as_rectangle else plot_rectangle_extent
    if individual_save_file is not None:
        video_saver = VideoSaver(verbose=True)
        plt.clf()
        plt.text(0, 0, "Consider going into fullscreen\nin order to record the video\nin fullscreen", size=25)
        plt.pause(1)

    for i, scene in enumerate(data):
        plt.cla()
        time = scene["time"]
        gt = scene["gt"]
        measurements = scene["measurements"]
        plt.title(f"{title_prefix}Step #{i + 1} at Time: {time}s")
        plot_extent(m=gt[:4], p=gt[4:], est_color=gt_color, color_alpha=ellipse_alpha, label="True Extent")
        plt.scatter(measurements[:, 0], measurements[:, 1], alpha=0.5, marker='o', c=meas_color, label='Measurements')

        for tracker_id in track_dict:
            if track_dict[tracker_id] is not None:
                track = np.array(track_dict[tracker_id])
                if tracker_color_dict is not None:
                    track_color = tracker_color_dict[tracker_id]
                else:
                    track_color = 'red'
                plot_extent(m=track[i, :4], p=track[i, 4:],
                            est_color=track_color, color_alpha=ellipse_alpha, label=tracker_id,
                            linestyle='--')

        # show plot
        plt.legend(loc="upper left")
        plt.axis('equal')
        plt.margins(2, 2)
        zoom = max(gt[5:]) * 2.5
        plt.xlim([gt[0] - zoom, gt[0] + zoom])
        plt.ylim([gt[1] - zoom, gt[1] + zoom])
        plt.draw()
        plt.pause(pause_interval)
        if individual_save_file is not None:
            video_saver.add_frame(plt.gcf())
    if individual_save_file is not None:
        video_saver.generate_video(filename=individual_save_file, fps=60, repeat_frame_count=40)


def show_multi_channel_image(image, titles=None, disable_ticks=False):
    """
    Helper Function to plot multiple channels of one image
    :param image: Image to show. np array of shape (n_channels, x, y)
    :param titles: List of titles for individual images. Needs to have as many entries as the image channels
    """
    n_channels = image.shape[0]
    titles = [f"Channel {c}" for c in range(n_channels)] if titles is None else titles
    assert len(titles) == image.shape[0], f"Image has {image.shape[0]} channels but {len(titles)} were given!"
    fig, axs = plt.subplots(nrows=1, ncols=n_channels)
    for c in range(n_channels):
        plt.sca(axs[c])
        plt.imshow(image[c, :, :])
        if disable_ticks:
            plt.xticks([])
            plt.yticks([])
        plt.title(titles[c])
